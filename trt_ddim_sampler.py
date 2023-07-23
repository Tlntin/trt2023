import numpy as np
import torch
from cuda import cudart



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

    @torch.no_grad()
    def sample(
        self,
        ddim_num_steps,
        batch_size,
        shape,
        batch_concat,
        batch_crossattn,
        # conditioning=None,
        # callback=None,
        # normals_sequence=None,
        # img_callback=None,
        # quantize_x0=False,
        eta=0.,
        # mask=None,
        # x0=None,
        temperature=1.,
        score_corrector=None,
        corrector_kwargs=None,
        verbose=True,
        # x_T=None,
        # log_every_t=100,
        unconditional_guidance_scale=1.,
        # unconditional_conditioning=None, # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
        dynamic_threshold=None,
        do_summarize=False,
        # ucg_schedule=None,
        # **kwargs
    ):
        # --- copy from make schedule ---
        c = self.ddpm_num_timesteps // ddim_num_steps
        ddim_timesteps = torch.arange(
            1, self.ddpm_num_timesteps + 1, c,
            dtype=torch.long,
            device=self.device
        )
        flip_ddim_timesteps = torch.flip(ddim_timesteps, [0])
        ddim_sampling_tensor = torch.stack(
            (flip_ddim_timesteps, flip_ddim_timesteps),
            1
        )
        # ddim sampling parameters
        alphas = self.alphas_cumprod[ddim_timesteps]
        alphas = torch.stack(
            (alphas, alphas),
            1
        ).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        alphas_prev = torch.cat(
            (self.alphas_cumprod[:1], self.alphas_cumprod[ddim_timesteps[:-1]]),
            0
        )
        alphas_prev = torch.stack(
            (alphas_prev, alphas_prev),
            1
        ).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        sqrt_one_minus_alphas = torch.sqrt(1. - alphas)
        # according the the formula provided in https://arxiv.org/abs/2010.02502
        sigmas = eta * torch.sqrt(
            (1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev)
        )
        # --- copy end  ---
        
        # batch_size = shape[0]
        img = torch.randn(shape, device=self.device)
        img = torch.cat((img, img), 0)
        # becasuse seed, rand is pin
        rand_noise = torch.rand_like(img, device=self.device) * temperature
        intermediates = {}
        for i in range(ddim_num_steps):
            index = ddim_num_steps - i - 1
            ts = ddim_sampling_tensor[i]
            alphas_at = alphas[index]
            alphas_prev_at = alphas_prev[index]
            sqrt_one_minus_alphas_at = sqrt_one_minus_alphas[index]
            sigmas_at = sigmas[index]
            img, _  = self.p_sample_ddim(
                img,
                ts,
                batch_concat=batch_concat,
                batch_crossattn=batch_crossattn,
                index=index,
                alphas_at=alphas_at,
                alphas_prev_at=alphas_prev_at,
                sigmas_at=sigmas_at,
                sqrt_one_minus_alphas_at=sqrt_one_minus_alphas_at,
                rand_noise=rand_noise,
                score_corrector=score_corrector,
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
        control_dict = self.run_engine(
            "union_model",
            {
                "sample": x_noisy,
                "hint": c_concat_merge,
                "timestep": t,
                "context": c_crossattn_merge
            }
        )
        eps = control_dict["latent"]
        return eps
    
    def decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
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
        alphas_at,
        alphas_prev_at,
        sqrt_one_minus_alphas_at,
        sigmas_at,
        rand_noise,
        score_corrector=None,
        corrector_kwargs=None,
        unconditional_guidance_scale=1.,
        unconditional_conditioning=None,
        dynamic_threshold=None
    ):
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
        pred_x0 = (x - sqrt_one_minus_alphas_at * e_t) / alphas_at.sqrt()
        # direction pointing to x_t
        dir_xt = (1. - alphas_prev_at - sigmas_at ** 2).sqrt() * e_t
        # noise = sigmas_at * noise_like(x.shape, self.device, repeat_noise) * temperature
        # noise = sigmas_at * torch.rand(x.shape, device=self.device) * temperature
        noise = sigmas_at * rand_noise
        x_prev = alphas_prev_at.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0