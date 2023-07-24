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
from models import SamplerModel


class hackathon():
    def __init__(
        self,
        onnx_device: str = "cuda",
        device: str = "cuda",

    ):
        self.device = torch.device(device)
        self.onnx_device = torch.device(onnx_device)

    def initialize(self):
        self.apply_canny = CannyDetector()
        model = create_model('./models/cldm_v15.yaml').cpu()
        model.load_state_dict(
            load_state_dict(
                '/home/player/ControlNet/models/control_sd15_canny.pth',
                location='cuda',
            )
        )
        model = model.cuda()
        self.state_dict = {
            "clip": "cond_stage_model",
            "control_net": "control_model",
            "unet": "diffusion_model",
            "vae": "first_stage_model"
        }
        raw_clip = getattr(model, self.state_dict["clip"])
        self.tokenizer = raw_clip.tokenizer
        clip = raw_clip.transformer
        control_net = getattr(model, self.state_dict["control_net"])
        unet = getattr(model.model, self.state_dict["unet"])
        vae = getattr(model, self.state_dict["vae"])
        vae.forward = vae.decode
        # copy some params from model to ddim_sampler
        num_timesteps = model.num_timesteps
        scale_factor = model.scale_factor
        alphas_cumprod = model.alphas_cumprod.to(self.device)
        self.model = SamplerModel(
           clip_model=clip,
           control_model=control_net,
           unet_model=unet,
           vae_model=vae,
           num_timesteps=num_timesteps,
           scale_factor=scale_factor,
           alphas_cumprod=alphas_cumprod
        ).to(self.device)
        

    def text_tokenizer(self, text_list: list):
        input_ids = self.tokenizer.batch_encode_plus(
            text_list,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.type(torch.int32).to(self.device)
        return input_ids

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
            if seed == -1:
                seed = random.randint(0, 65535)
            seed_everything(seed)
            # because the genrate_batch pin, so do like this
            text_list = [prompt + ', ' + a_prompt] + [n_prompt]
            input_ids = self.text_tokenizer(text_list)
            samples = self.model.sample(
                control=control,
                input_ids=input_ids,
                uncond_scale=torch.tensor([scale], dtype=torch.float32, device=self.device),
                ddim_num_steps=ddim_steps,
                eta=eta,
                batch_size=num_samples,
            )
            x_samples = samples.to(torch.uint8).cpu().numpy()
            results = [x_samples[0]]
        return results