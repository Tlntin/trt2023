import torch
import einops
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
        control,
        batch_crossattn,
        ddim_num_steps,
        eta=0.,
        temperature=1.,
        uncond_scale=1.,
        batch_size=1,
    ):
        h, w, c = control.shape
        device = control.device
        shape = (batch_size, 4, h // 8, w // 8)
        # make ddim_num_step % 4 == 0
        # ddim_num_steps = (ddim_num_steps + 3) // 4 * 4
        control = torch.stack(
            [control for _ in range(batch_size * 2)],
            dim=0
        )
        control = einops.rearrange(control, 'b h w c -> b c h w')
        # --- copy from make schedule ---
        c = self.ddpm_num_timesteps // ddim_num_steps
        ddim_timesteps = torch.arange(
            1, self.ddpm_num_timesteps + 1, c,
            dtype=torch.int32,
            device=device
        )
        ddim_sampling_tensor = ddim_timesteps\
            .unsqueeze(1).repeat(1, 2 * batch_size).view(-1)
        # ddim sampling parameters
        alphas = self.alphas_cumprod[ddim_timesteps]
        alphas_prev = torch.cat(
            (self.alphas_cumprod[:1], self.alphas_cumprod[ddim_timesteps[:-1]]),
            0
        )
        sqrt_one_minus_alphas = torch.sqrt(1. - alphas)
        # according the the formula provided in https://arxiv.org/abs/2010.02502
        sigmas = eta * torch.sqrt(
            (1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev)
        )
        # --- copy end  ---
        # ---optimizer code for forward -- #
        alphas = alphas.sqrt()
        temp_di = (1. - alphas_prev - sigmas ** 2).sqrt()
        alphas_prev = alphas_prev.sqrt()
        # noise = sigmas_at * rand_noise
        # batch_size = shape[0]
        img = torch.randn(shape, device=device)
        rand_noise = torch.rand_like(img, device=device) * temperature
        img = img.repeat(2 * batch_size, 1, 1, 1)
        # becasuse seed, rand is pin, use unsqueeze(0) to auto boradcast
        noise = sigmas.unsqueeze(1).unsqueeze(2).unsqueeze(3) * rand_noise
        for i in range(ddim_num_steps):
            index = ddim_num_steps - i - 1
            img = self.p_sample_ddim(
                img,
                hint=control,
                t=ddim_sampling_tensor[index * 2: index * 2 + 2],
                context=batch_crossattn,
                index=index,
                alphas_at=alphas[index],
                alphas_prev_at=alphas_prev[index],
                sqrt_one_minus_alphas_at=sqrt_one_minus_alphas[index],
                noise_at=noise[index],
                temp_di_at=temp_di[index],
                uncond_scale=uncond_scale,
            )
        return img[:batch_size]
    
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
    
    def apply_model(self, sample, hint, timestep, context):
        control_dict = self.run_engine(
            "union_model",
            {
                "sample": sample,
                "hint": hint,
                "timestep": timestep,
                "context": context,
            }
        )
        eps = control_dict["latent"]
        return eps
    
    def decode_first_stage(self, z):
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
        hint,
        t,
        context,
        index,
        alphas_at,
        alphas_prev_at,
        sqrt_one_minus_alphas_at,
        noise_at,
        temp_di_at,
        uncond_scale,
    ):
        if self.do_summarize:
            cudart.cudaEventRecord(self.events[f'union_model_{index}-start'], 0)
        b_latent = self.apply_model(x, hint, t, context)
        b = x.shape[0] // 2
        if self.do_summarize:
            cudart.cudaEventRecord(self.events[f'union_model_{index}-stop'], 0)
        e_t = b_latent[b:] + uncond_scale * (b_latent[:b] - b_latent[b:])
        pred_x0 = (x - sqrt_one_minus_alphas_at * e_t) / alphas_at
        # direction pointing to x_t
        dir_xt = temp_di_at * e_t
        x_prev = alphas_prev_at * pred_x0 + dir_xt + noise_at
        return x_prev