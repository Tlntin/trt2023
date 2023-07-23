import numpy as np
import torch
from cuda import cudart
# --- add to sample --- #
from ldm.modules.diffusionmodules.util import (
    make_ddim_sampling_parameters,
    make_ddim_timesteps,
    noise_like,
)



class TRT_DDIMSampler(object):
    def __init__(
            self,
            device,
            engine: dict,
            events: dict,
            stream,
            cuda_stream,
            num_timesteps,
            scale_factor,
            alphas_cumprod,
            do_summarize: bool = False,
            use_cuda_graph: bool = False,
            schedule="linear", **kwargs
        ):
        super().__init__()
        self.device = device
        self.scale_factor = scale_factor
        self.alphas_cumprod = alphas_cumprod
        self.engine = engine
        self.events = events
        self.stream = stream
        self.cuda_stream = cuda_stream
        self.use_cuda_graph = use_cuda_graph
        self.do_summarize = do_summarize
        self.ddpm_num_timesteps = num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_eta=0.):
        c = self.ddpm_num_timesteps // ddim_num_steps
        self.ddim_timesteps = torch.arange(
            1, self.ddpm_num_timesteps + 1, c,
            dtype=torch.long,
            device=self.device
        )
        self.flip_ddim_timesteps = torch.flip(self.ddim_timesteps, [0])
        self.ddim_sampling_tensor = torch.stack(
            (self.flip_ddim_timesteps, self.flip_ddim_timesteps),
            1
        )
        assert self.alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        # ddim sampling parameters
        ddim_alphas = self.alphas_cumprod[self.ddim_timesteps]
        ddim_alphas_prev = torch.cat(
            (self.alphas_cumprod[:1], self.alphas_cumprod[self.ddim_timesteps[:-1]]),
            0
        )
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', torch.sqrt(1. - ddim_alphas))
        # according the the formula provided in https://arxiv.org/abs/2010.02502
        ddim_sigmas = ddim_eta * torch.sqrt(
            (1 - ddim_alphas_prev) / (1 - ddim_alphas) * \
            (1 - ddim_alphas / ddim_alphas_prev)
        )
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               batch_concat,
               batch_crossattn,
               # conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               # unconditional_conditioning=None, # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               dynamic_threshold=None,
               ucg_schedule=None,
               **kwargs
               ):
        self.make_schedule(ddim_num_steps=S, ddim_eta=eta)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        samples, intermediates = self.ddim_sampling(
            size,
            ddim_num_steps=S,
            batch_concat=batch_concat,
            batch_crossattn=batch_crossattn,
            callback=callback,
            img_callback=img_callback,
            quantize_denoised=quantize_x0,
            mask=mask, x0=x0,
            # ddim_use_original_steps=False,
            noise_dropout=noise_dropout,
            temperature=temperature,
            score_corrector=score_corrector,
            corrector_kwargs=corrector_kwargs,
            x_T=x_T,
            log_every_t=log_every_t,
            unconditional_guidance_scale=unconditional_guidance_scale,
            dynamic_threshold=dynamic_threshold,
            ucg_schedule=ucg_schedule,
        )
        
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(
        self,
        # cond,
        shape,
        ddim_num_steps,
        batch_concat,
        batch_crossattn,
        x_T=None,
        # ddim_use_original_steps=False,
        callback=None,
        timesteps=None,
        quantize_denoised=False,
        mask=None,
        x0=None,
        img_callback=None,
        log_every_t=100,
        temperature=1.,
        noise_dropout=0.,
        score_corrector=None,
        corrector_kwargs=None,
        unconditional_guidance_scale=1.,
        # unconditional_conditioning=None,
        dynamic_threshold=None,
        ucg_schedule=None,
    ):
        batch_size = shape[0]
        # if x_T is None:
        #     img = torch.randn(shape, device=device)
        # else:
        #     img = x_T
        img = torch.randn(shape, device=self.device)
        img = torch.cat((img, img), 0)

        # if timesteps is None:
        #     timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        # elif timesteps is not None and not ddim_use_original_steps:
        #     subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
        #     timesteps = self.ddim_timesteps[:subset_end]

        # intermediates = {'x_inter': [img], 'pred_x0': [img]}
        intermediates = {}
        # time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        # total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        # print(f"Running DDIM Sampling with {total_steps} timesteps")

        #iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        for i in range(ddim_num_steps):
            # index = total_steps - i - 1
            # ts = torch.full((2 * b, ), step, device=device, dtype=torch.long)
            index = ddim_num_steps - i - 1
            ts = self.ddim_sampling_tensor[i]
            img, _  = self.p_sample_ddim(
                img,
                ts,
                batch_concat=batch_concat,
                batch_crossattn=batch_crossattn,
                index=index,
                # use_original_steps=ddim_use_original_steps,
                quantize_denoised=quantize_denoised, temperature=temperature,
                noise_dropout=noise_dropout, score_corrector=score_corrector,
                corrector_kwargs=corrector_kwargs,
                unconditional_guidance_scale=unconditional_guidance_scale,
                # unconditional_conditioning=unconditional_conditioning,
                dynamic_threshold=dynamic_threshold
            )
        return img[:batch_size], intermediates
    
    def run_engine(self, model_name: str, feed_dict: dict):
        """
        run TensorRT engine
        :params model_name: model_name must in ["clip", "control_net", "unet", "vae"]
        :params feed_dict: the input params dict
        """
        engine = self.engine[model_name]
        result = engine.infer(
            feed_dict,
            self.stream,
            self.cuda_stream,
            use_cuda_graph=self.use_cuda_graph
        )
        return result
    
    def apply_model(self, x_noisy, t, c_concat_merge, c_crossattn_merge, *args, **kwargs):
        """
        need to replace pytorch with tensorRT
        """
        #assert isinstance(cond, dict)
        #cond_txt = torch.cat(cond['c_crossattn'], 1)

        # if cond['c_concat'] is None:
        #     eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
        # else:
        # control = self.control_model(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond_txt)
        control_dict = self.run_engine(
            "union_model",
            {
                "sample": x_noisy,
                #"hint": torch.cat(cond['c_concat'], 1),
                "hint": c_concat_merge,
                "timestep": t,
                "context": c_crossattn_merge
            }
        )
        # self.control_scalres = [1] * n, ignore it
        # control = [c * scale for c, scale in zip(control, self.control_scales)]
        # eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)
        # eps = control_dict["latent"].clone()
        eps = control_dict["latent"]
        return eps
    
    def decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        # if predict_cids:
        #     if z.dim() == 4:
        #         z = torch.argmax(z.exp(), dim=1).long()
        #     z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
        #     z = rearrange(z, 'b h w c -> b c h w').contiguous()

        z = 1. / self.scale_factor * z
        # return self.first_stage_model.decode(z)
        return self.run_engine(
            "vae",
            {"latent": z}
        )["images"]

    @torch.no_grad()
    def p_sample_ddim(
        self,
        x,
        t,
        batch_concat,
        batch_crossattn,
        index,
        repeat_noise=False,
        # use_original_steps=False,
        quantize_denoised=False,
        temperature=1.,
        noise_dropout=0.,
        score_corrector=None,
        corrector_kwargs=None,
        unconditional_guidance_scale=1.,
        unconditional_conditioning=None,
        dynamic_threshold=None
    ):
        b, *_, device = *x.shape, x.device
        if self.do_summarize:
            cudart.cudaEventRecord(self.events[f'union_model_{index}-start'], 0)
        model_uncond_merge = self.apply_model(x, t, batch_concat, batch_crossattn)
        #model_t = self.apply_model(x, t, c)
        #model_uncond = self.apply_model(x, t, unconditional_conditioning)
        if self.do_summarize:
            cudart.cudaEventRecord(self.events[f'union_model_{index}-stop'], 0)
        #model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)
        model_output = model_uncond_merge[1] + unconditional_guidance_scale * (model_uncond_merge[0] - model_uncond_merge[1])
        e_t = model_output
        alphas = self.ddim_alphas
        alphas_prev = self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas
        sigmas = self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0