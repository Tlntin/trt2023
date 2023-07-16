from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random
from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
# ------------ new add ---------------
import os
from cuda import cudart
from models import (get_embedding_dim, CLIP, UNet, VAE)
from utilities import Engine


class hackathon():
    now_dir = os.path.dirname(os.path.abspath(__file__))
    def __init__(
        self,
        version="1.5",
        stages=["clip", "control_net", "unet", "vae"],
        max_batch_size=16,
        de_noising_steps=20,
        device='cuda',
        output_dir = os.path.join(now_dir, "output"),
        verbose=False,
        nvtx_frofile=False,
        use_cuda_graph=False
        ) -> None:
        """
        Initializes the hackathon pipeline.

        Args:
            version (str):
                The version of the pipeline. Should be one of [1.5, 2.1]
            stages (list):
                Ordered sequence of stages. Options: ['vae_encoder', 'clip','unet', 'control_net' ,'vae']
            max_batch_size (int):
                Maximum batch size for dynamic batch engine.
            de_noising_steps (int):
                The number of denoising steps.
                More denoising steps usually lead to a higher quality image at the expense of slower inference.
            device (str):
                PyTorch device to run inference. Default: 'cuda'
            output_dir (str):
                Output directory for log files and image artifacts
            verbose (bool):
                Enable verbose logging.
            nvtx_profile (bool):
                Insert NVTX profiling markers.
            use_cuda_graph (bool):
                Use CUDA graph to capture engine execution and then launch inference
        """
        self.de_noising_steps = de_noising_steps
        self.max_batch_size = max_batch_size
        _, free_mem, _ = cudart.cudaMemGetInfo()
        one_gb = 1 ** 30
        if free_mem > 12 * one_gb:
            activate_memory = 8 * one_gb
            self.max_workspace_size = free_mem - activate_memory
        else:
            self.max_workspace_size = 0
        self.version = version
        self.output_dir = output_dir 
        if not os.path.exists(output_dir):
            os.mkdir(self.output_dir)
        self.device = device
        self.verbose = verbose
        self.nvtx_profile = nvtx_profile
        self.stages = stages
        self.state_dict = {
            "clip": "cond_stage_model",
            "control_net": "control_model",
            "unet": "diffusion_model",
            "vae": "first_stage_model"
        }
        self.use_cuda_graph = use_cuda_graph
        self.stream = None
        self.tokenizer = None
        self.apply_canny = None
        self.ddim_sampler = None
        self.model = None
        self.engine = {}
        self.shared_device_memory = None
        self.embedding_dim = get_embedding_dim(self.version)

    def initialize(self):
        self.apply_canny = CannyDetector()
        self.model = create_model('./models/cldm_v15.yaml').cpu()
        self.model.load_state_dict(
            load_state_dict('./models/control_sd15_canny.pth', location='cuda')
        )
        self.model = self.model.cuda()
        for k, v in self.state_dict.items():
            temp_model = getattr(self.model, v)
            if k == "clip":
                self.tokenizer = temp_model.tokenizer
                new_model = CLIP(
                    model=temp_model,
                    device=self.device,
                    verbose=self.verbose,
                    max_batch_size=self.max_batch_size,
                    embedding_dim=self.embedding_dim
                )
                delattr(self.model, v)
                setattr(self.model, k, new_model)


            elif k == "unet":
                new_model = UNet(
                    model=temp_model,
                    device=self.device,
                    verbose=self.verbose,
                    max_batch_size=self.max_batch_size,
                    embedding_dim=self.embedding_dim
                )
                delattr(self.model, v)
                setattr(self.model, k, new_model)

            elif k == "vae":
                new_model = VAE(
                    model=temp_model,
                    device=self.device,
                    verbose=self.verbose,
                    max_batch_size=self.max_batch_size,
                    embedding_dim=self.embedding_dim
                )
                delattr(self.model, v)
                setattr(self.model, k, new_model)
            else:
                pass
                # raise Exception("Unknown stage")
        self.ddim_sampler = DDIMSampler(self.model)
    
    def load_resources(self, image_height, image_width, batch_size, seed):
        # Initialize noise generator
        # self.generator = torch.Generator(device="cuda").manual_seed(seed) if seed else None

        # Pre-compute latent input scales and linear multistep coefficients
        # self.scheduler.set_timesteps(self.denoising_steps)
        # self.scheduler.configure()

        # Create CUDA events and stream
        self.events = {}
        for stage in self.stages:
            for marker in ['start', 'stop']:
                self.events[stage + '-' + marker] = cudart.cudaEventCreate()[1]
        self.stream = cuda.Stream()

        # Allocate buffers for TensorRT engine bindings
        # for model_name, obj in self.models.items():
        for model_name in self.stages:
            obj = getattr(self.model, model_name)
            self.engine[model_name].allocate_buffers(
                shape_dict=obj.get_shape_dict(batch_size, image_height, image_width),
                device=self.device
            )

    def teardown(self):
        for e in self.events.values():
            cudart.cudaEventDestroy(e)

        for engine in self.engine.values():
            del engine

        if self.shared_device_memory:
            self.shared_device_memory.free()

        self.stream.free()
        del self.stream

    def cached_model_name(self, model_name):
        # if self.inpaint:
        #     model_name += '_inpaint'
        return model_name

    def get_onnx_path(self, model_name, onnx_dir, opt=True):
        return os.path.join(
            onnx_dir,
            self.cached_model_name(model_name) + ('.opt' if opt else '') +'.onnx'
        )

    def get_engine_path(self, model_name, engine_dir):
        return os.path.join(
            engine_dir,
            self.cached_model_name(model_name)+'.plan'
        )

    def load_engines(
        self,
        engine_dir,
        onnx_dir,
        onnx_opset,
        opt_batch_size,
        opt_image_height,
        opt_image_width,
        force_export=False,
        force_optimize=False,
        force_build=False,
        static_batch=False,
        static_shape=True,
        enable_preview=False,
        enable_all_tactics=False,
        timing_cache=None,
    ):
        """
        Build and load engines for TensorRT accelerated inference.
        Export ONNX models first, if applicable.

        Args:
            engine_dir (str):
                Directory to write the TensorRT engines.
            onnx_dir (str):
                Directory to write the ONNX models.
            onnx_opset (int):
                ONNX opset version to export the models.
            opt_batch_size (int):
                Batch size to optimize for during engine building.
            opt_image_height (int):
                Image height to optimize for during engine building. Must be a multiple of 8.
            opt_image_width (int):
                Image width to optimize for during engine building. Must be a multiple of 8.
            force_export (bool):
                Force re-exporting the ONNX models.
            force_optimize (bool):
                Force re-optimizing the ONNX models.
            force_build (bool):
                Force re-building the TensorRT engine.
            static_batch (bool):
                Build engine only for specified opt_batch_size.
            static_shape (bool):
                Build engine only for specified opt_image_height & opt_image_width. Default = True.
            enable_preview (bool):
                Enable TensorRT preview features.
            enable_all_tactics (bool):
                Enable all tactic sources during TensorRT engine builds.
            timing_cache (str):
                Path to the timing cache to accelerate build or None
        """
        # Load pipeline models
        models_args = {
            'version': self.version,
            'device': self.device,
            'verbose': self.verbose,
            'max_batch_size': self.max_batch_size
        }


        # Export models to ONNX
        for model_name in self.stages:
            obj = getattr(self.model, model_name)
            engine_path = self.get_engine_path(model_name, engine_dir)
            if force_export or force_build or not os.path.exists(engine_path):
                onnx_path = self.get_onnx_path(model_name, onnx_dir, opt=False)
                onnx_opt_path = self.get_onnx_path(model_name, onnx_dir)
                if force_export or not os.path.exists(onnx_opt_path):
                    if force_export or not os.path.exists(onnx_path):
                        print(f"Exporting model: {onnx_path}")
                        model = obj.get_model()
                        with torch.inference_mode(), torch.autocast("cuda"):
                            inputs = obj.get_sample_input(
                                opt_batch_size,
                                opt_image_height,
                                opt_image_width
                            )
                            torch.onnx.export(model,
                                    inputs,
                                    onnx_path,
                                    export_params=True,
                                    opset_version=onnx_opset,
                                    do_constant_folding=True,
                                    input_names=obj.get_input_names(),
                                    output_names=obj.get_output_names(),
                                    dynamic_axes=obj.get_dynamic_axes(),
                            )
                        del model
                        torch.cuda.empty_cache()
                        gc.collect()
                    else:
                        print(f"Found cached model: {onnx_path}")

                    # Optimize onnx
                    if force_optimize or not os.path.exists(onnx_opt_path):
                        print(f"Generating optimizing model: {onnx_opt_path}")
                        onnx_opt_graph = obj.optimize(onnx.load(onnx_path))
                        onnx.save(onnx_opt_graph, onnx_opt_path)
                    else:
                        print(f"Found cached optimized model: {onnx_opt_path} ")

        # Build TensorRT engines
        for model_name in self.stages:
            obj = getattr(self.model, model_name)
            engine_path = self.get_engine_path(model_name, engine_dir)
            engine = Engine(engine_path)
            onnx_path = self.get_onnx_path(
                model_name,
                onnx_dir,
                opt=False
            )
            onnx_opt_path = self.get_onnx_path(
                model_name, onnx_dir
            )

            if force_build or not os.path.exists(engine.engine_path):
                engine.build(
                    onnx_opt_path,
                    fp16=True,
                    input_profile=obj.get_input_profile(
                        opt_batch_size, opt_image_height, opt_image_width,
                        static_batch=static_batch, static_shape=static_shape
                    ),
                    enable_refit=enable_refit,
                    enable_preview=enable_preview,
                    enable_all_tactics=enable_all_tactics,
                    timing_cache=timing_cache,
                    workspace_size=self.max_workspace_size
                )
            self.engine[model_name] = engine

        # Load and activate TensorRT engines
        max_device_memory = 0
        for model_name, obj in self.models.items():
            engine = self.engine[model_name]
            engine.load()
            max_device_memory = max(max_device_memory, engine.engine.device_memory_size)
            if onnx_refit_dir:
                onnx_refit_path = self.getOnnxPath(model_name, onnx_refit_dir)
                if os.path.exists(onnx_refit_path):
                    engine.refit(onnx_opt_path, onnx_refit_path)
        print("max device memory: ", max_device_memory)
        self.shared_device_memory = cuda.DeviceArray.raw(
            (int(max_device_memory * 0.90),)
        )
        for engine in self.engine.values():
            engine.activate(reuse_device_memory=self.shared_device_memory.ptr)


    def process(
            self,
            input_image: np.array,
            prompt: str,
            a_prompt: str,
            n_prompt: str,
            num_samples: int,
            image_resolution: int,
            ddim_steps: int,
            guess_mode: bool,
            strength: int,
            scale: int,
            seed: int,
            eta: float,
            low_threshold: float,
            high_threshold: float
        ):
        with torch.no_grad():
            img = resize_image(HWC3(input_image), image_resolution)
            H, W, C = img.shape

            detected_map = self.apply_canny(img, low_threshold, high_threshold)
            detected_map = HWC3(detected_map)

            control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
            control = torch.stack([control for _ in range(num_samples)], dim=0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()

            if seed == -1:
                seed = random.randint(0, 65535)
            seed_everything(seed)

            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=False)

            cond = {
                "c_concat": [control],
                "c_crossattn": [
                    self.model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)
                ]
            }
            un_cond = {
                "c_concat": None if guess_mode else [control],
                "c_crossattn": [
                    self.model.get_learned_conditioning([n_prompt] * num_samples)
                ]
            }
            shape = (4, H // 8, W // 8)

            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=True)

            # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
            self.model.control_scales = [
                strength * (0.825 ** float(12 - i)) for i in range(13)
            ] if guess_mode else ([strength] * 13) 
            samples, intermediates = self.ddim_sampler.sample(
                ddim_steps,
                num_samples,
                shape,
                cond,
                verbose=False,
                eta=eta,
                unconditional_guidance_scale=scale,
                unconditional_conditioning=un_cond
            )

            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=False)

            x_samples = self.model.decode_first_stage(samples)
            x_samples = (
                einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5
            ).cpu().numpy().clip(0, 255).astype(np.uint8)

            results = [x_samples[i] for i in range(num_samples)]
        return results


if __name__ == "__main__":
    