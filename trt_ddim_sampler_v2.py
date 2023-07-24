import torch
from cuda import cudart
import einops



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
        control: torch.Tensor,
        batch_crossattn: torch.Tensor,
        uncond_scale: torch.Tensor,
        ddim_num_steps = 20,
        eta = 0.,
        temperature=1.,
    ):
        h, w, c = control.shape
        device = control.device
        shape = (1, 4, h // 8, w // 8)
        # make ddim_num_step % 4 == 0
        ddim_num_steps = (ddim_num_steps + 3) // 4 * 4
        control = torch.stack(
            # [control for _ in range(num_samples * 2)],
            (control, control),
        dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w')
        # --- copy from make schedule ---
        c = self.ddpm_num_timesteps // ddim_num_steps
        ddim_timesteps = torch.arange(
            1, self.ddpm_num_timesteps + 1, c,
            dtype=torch.long,
            device=device
        )
        # flip_ddim_timesteps = torch.flip(ddim_timesteps, [0])
        ddim_sampling_tensor = torch.stack(
            # [flip_ddim_timesteps for _ in range(num_samples * 2)],
            (ddim_timesteps, ddim_timesteps),
            1
        )
        # ddim sampling parameters
        alphas = self.alphas_cumprod[ddim_timesteps]
        alphas = torch.stack(
            # [alphas for _ in range(num_samples * 2)],
            (alphas, alphas),
            1
        ).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        alphas_prev = torch.cat(
            (self.alphas_cumprod[:1], self.alphas_cumprod[ddim_timesteps[:-1]]),
            0
        )
        alphas_prev = torch.stack(
            # [alphas_prev for _ in range(num_samples * 2)],
            (alphas_prev, alphas_prev),
            1
        ).unsqueeze(2).unsqueeze(3).unsqueeze(4)
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
        img = torch.cat(
            # [img for _ in range(num_samples * 2)],
            (img, img),
            0
        )
        # becasuse seed, rand is pin, use unsqueeze(0) to auto boradcast
        rand_noise = torch.rand_like(img, device=device).unsqueeze(0) * temperature
        noise = sigmas * rand_noise
        # --optimizer code end -- #
        count = 0
        for i in range(0, ddim_num_steps, 4):
            index = ddim_num_steps - i
            if self.do_summarize:
                cudart.cudaEventRecord(self.events[f'union_model_v2_{count}-start'], 0)
            img  = self.p_sample_ddim(
                img,
                hint=control,
                timestep=ddim_sampling_tensor[index - 4: index],
                context=batch_crossattn,
                alphas=alphas[index - 4: index],
                alphas_prev=alphas_prev[index - 4: index],
                sqrt_one_minus_alphas=sqrt_one_minus_alphas[index - 4: index],
                noise=noise[index - 4: index],
                temp_di=temp_di[index -4: index],
                uncond_scale=uncond_scale,
            )
            if self.do_summarize:
                cudart.cudaEventRecord(self.events[f'union_model_v2_{count}-stop'], 0)
            count += 1
        return img[:1]
    
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
    
    def p_sample_ddim(
        self,
        sample,
        hint,
        timestep,
        context,
        alphas,
        alphas_prev,
        sqrt_one_minus_alphas,
        noise,
        temp_di,
        uncond_scale
    ):
        control_dict = self.run_engine(
            "union_model_v2",
            {
                "sample": sample,
                "hint": hint,
                "timestep": timestep,
                "context": context,
                "alphas": alphas,
                "alphas_prev": alphas_prev,
                "sqrt_one_minus_alphas": sqrt_one_minus_alphas,
                "noise": noise,
                "temp_di": temp_di,
                "uncond_scale": uncond_scale,
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