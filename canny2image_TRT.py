from share import *
import config
import cv2
import einops
# import gradio as gr
import numpy as np
import torch
import random
import gc
from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
# from cldm.ddim_hacked import DDIMSampler
# ------------ new add ---------------
import os
import onnx
import shutil
from cuda import cudart
from models import (get_embedding_dim, CLIP, UNet_V2, UnionModel, VAE)
from polygraphy import cuda
import tensorrt as trt
from utilities import Engine, TRT_LOGGER
from trt_ddim_sampler import TRT_DDIMSampler
#-----------add for compare func -----------
from polygraphy.backend.common import BytesFromPath
from polygraphy.backend.onnxrt import OnnxrtRunner, SessionFromOnnx
from polygraphy.backend.trt import EngineFromBytes, TrtRunner
from polygraphy.common import TensorMetadata
from polygraphy.comparator import Comparator, CompareFunc, DataLoader

now_dir = os.path.dirname(os.path.abspath(__file__))


class hackathon():
    def __init__(
        self,
        version="1.5",
        stages=["clip", "union_model", "unet_v2", "vae"],
        de_noising_steps=20,
        guidance_scale=9.0,
        onnx_device="cpu",
        device='cuda',
        output_dir = os.path.join(now_dir, "output"),
        verbose=False,
        nvtx_profile=False,
        use_cuda_graph=True,
        do_summarize = False,
        do_compare: bool = False,
        builder_optimization_level=5,
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
        self.guidance_scale = guidance_scale
        self.do_summarize = do_summarize
        self.do_compare = do_compare
        self.builder_optimization_level = builder_optimization_level
        # Register TensorRT plugins
        trt.init_libnvinfer_plugins(TRT_LOGGER, '')
        _, free_mem, _ = cudart.cudaMemGetInfo()
        one_gb = 2 ** 30
        if free_mem > 6 * one_gb:
            activate_memory = 4 * one_gb
            self.max_workspace_size = free_mem - activate_memory
        else:
            self.max_workspace_size = 0
        self.version = version
        self.output_dir = output_dir 
        if not os.path.exists(output_dir):
            os.mkdir(self.output_dir)
        self.device = torch.device(device)
        self.onnx_device = torch.device(onnx_device)
        self.onnx_device_raw = onnx_device
        self.verbose = verbose
        self.nvtx_profile = nvtx_profile
        self.stages = stages
        self.state_dict = {
            "clip": "cond_stage_model",
            "control_net": "control_model",
            "unet": "diffusion_model",
            "unet_v2": "diffusion_model",
            "vae": "first_stage_model"
        }
        self.stage_batch_dict = {
            "clip": {
                "min": 2,
                "opt": 2,
                "max": 2,
            },
            # union model拼接了control_net和Unet，如果是guess mode, 则batch=1,否则batch=1
            "union_model": {
                "min": 1,
                "opt": 2,
                "max": 2,
            },
            # 这个unet v2仅用于guess model，此时输入的control=None
            "unet_v2": {
                "min": 1,
                "opt": 1,
                "max": 1,
            },
            "vae": {
                "min": 1,
                "opt": 1,
                "max": 1,
            }
        }
        self.use_cuda_graph = use_cuda_graph
        self.stream = None
        self.cuda_stream = None
        self.tokenizer = None
        self.max_length = 0
        self.apply_canny = None
        self.ddim_sampler = None
        self.engine = {}
        self.events = {}
        self.shared_device_memory = None
        self.embedding_dim = get_embedding_dim(self.version)
        # 储存一些旧模型信息
        self.model_dict = {}
        # 记录上一次计算是不是guess_mode
        self.last_guess_mode = False

    def initialize(self):
        self.apply_canny = CannyDetector()
        model = create_model('./models/cldm_v15.yaml').cpu()
        model.load_state_dict(
            load_state_dict(
                '/home/player/ControlNet/models/control_sd15_canny.pth',
                location=self.onnx_device_raw
            )
        )
        model = model.to(self.onnx_device)
        for k, v in self.state_dict.items():
            if k not in ["unet", "unet_v2"]:
                temp_model = getattr(model, v)
            else:
                temp_model = getattr(model.model, v)
            if k in ["clip", "vae", "unet_v2"]:
                min_batch_size = self.stage_batch_dict[k]["min"]
                max_batch_size = self.stage_batch_dict[k]["max"]
            if k == "clip":
                self.tokenizer = temp_model.tokenizer
                self.max_length = temp_model.max_length
                new_model = CLIP(
                    model=temp_model,
                    device=self.onnx_device,
                    verbose=self.verbose,
                    min_batch_size=min_batch_size,
                    max_batch_size=max_batch_size,
                    embedding_dim=self.embedding_dim
                )
                delattr(model, v)
                self.model_dict[k] = new_model
            elif k == "unet_v2":
                new_model = UNet_V2(
                    model=temp_model,
                    device=self.onnx_device,
                    verbose=self.verbose, 
                    min_batch_size=min_batch_size,
                    max_batch_size=max_batch_size,
                )
                # 这里不用删除model下面的属性，待会union还能用
                self.model_dict[k] = new_model
            elif k == "vae":
                new_model = VAE(
                    model=temp_model,
                    device=self.onnx_device,
                    verbose=self.verbose,
                    min_batch_size=min_batch_size,
                    max_batch_size=max_batch_size,
                    embedding_dim=self.embedding_dim
                )
                delattr(model, v)
                self.model_dict[k] = new_model
            else:
                pass
        # merge control_net and unet
        control_net_model = getattr(model, self.state_dict["control_net"])
        unet_model = getattr(model.model, self.state_dict["unet"])
        min_batch_size = self.stage_batch_dict["union_model"]["min"]
        max_batch_size = self.stage_batch_dict["union_model"]["max"]
        self.model_dict["union_model"] = UnionModel(
            control_model=control_net_model,
            unet_model=unet_model,
            device=self.onnx_device,
            verbose=self.verbose,
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
            embedding_dim=self.embedding_dim
        )
        delattr(model, self.state_dict["control_net"])
        delattr(model.model, self.state_dict["unet"])
        # copy some params from model to ddim_sampler
        num_timesteps = model.num_timesteps
        scale_factor = model.scale_factor
        alphas_cumprod = model.alphas_cumprod.to(self.device)
        # clear model
        del model
        gc.collect()
        torch.cuda.empty_cache()
        # --- build or load engine --- #
        output_dir = os.path.join(now_dir, "output")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        engine_dir = os.path.join(output_dir, "engine")
        if not os.path.exists(engine_dir):
            os.makedirs(engine_dir)
        onnx_dir = os.path.join(output_dir, "onnx")
        if not os.path.exists(onnx_dir):
            os.makedirs(onnx_dir)
        # time_cache_path = os.path.join(output_dir, "time_cache_fp16.cache")
        # image size is pin
        image_height = 256
        image_width = 384
        self.load_engines(
            engine_dir=engine_dir,
            onnx_dir=onnx_dir,
            onnx_opset=17,
            opt_image_height=image_height,
            opt_image_width=image_width,
        )
        # --- load resource --- #
        self.load_resources(
            image_height=image_height,
            image_width=image_width,
        )
        # clear model_dict(貌似不用清理了，下面union_model会用到)
        # for k, v in self.model_dict.items():
        #     del v
        #     gc.collect()
        #     torch.cuda.empty_cache()
        self.ddim_sampler = TRT_DDIMSampler(
            device=self.device,
            engine=self.engine,
            events=self.events,
            stream=self.stream,
            cuda_stream=self.cuda_stream,
            num_timesteps=num_timesteps,
            scale_factor=scale_factor,
            alphas_cumprod=alphas_cumprod,
            do_summarize=self.do_summarize,
            use_cuda_graph=self.use_cuda_graph
        )

        # --- first pre predict to speed CUDA graph and other --- #
        for guess_mode in [True, False]:
            for i in range(4):
                first_image_path = os.path.join(
                    now_dir, "test_imgs", "bird", f"{i}.jpg"
                )
                first_image = cv2.imread(first_image_path)
                self.process(
                    first_image,
                    "a bird", 
                    "best quality, extremely detailed", 
                    "longbody, lowres, bad anatomy, bad hands, missing fingers", 
                    1, 
                    256, 
                    20,
                    guess_mode,
                    1.0, 
                    9, 
                    2946901, 
                    0.0, 
                    100, 
                    200
                )
    
    def text_embedding(self, text_list: list):
        # batch_encoding = self.tokenizer(
        #     text_list,
        #     truncation=True,
        #     max_length=self.max_length,
        #     return_length=True,
        #     return_overflowing_tokens=False,
        #     padding="max_length",
        #     return_tensors="pt"
        # )
        # input_ids = batch_encoding["input_ids"].int().to(self.device)
        text_input_ids = self.tokenizer.batch_encode_plus(
            text_list,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.type(torch.int32).to(self.device)
        text_input_ids_inp = text_input_ids
        text_embeddings = self.ddim_sampler.run_engine(
            "clip",
            feed_dict={
                "input_ids": text_input_ids_inp
            }
        )["text_embeddings"]
        return text_embeddings
    
    def load_resources(self, image_height, image_width):
        # Initialize noise generator
        # self.generator = torch.Generator(device="cuda").manual_seed(seed) if seed else None

        # Pre-compute latent input scales and linear multistep coefficients
        # self.scheduler.set_timesteps(self.denoising_steps)
        # self.scheduler.configure()

        # Create CUDA events and stream
        for stage in self.stages:
            if stage != "union_model":
                for marker in ['start', 'stop']:
                    self.events[stage + '-' + marker] = cudart.cudaEventCreate()[1]
            else:
                for i in range(self.de_noising_steps):
                    for marker in ['start', 'stop']:
                        self.events[stage + "_{}".format(i) + '-' + marker] = cudart.cudaEventCreate()[1]
        self.stream = cuda.Stream()
        self.cuda_stream = cuda.Stream()

        # Allocate buffers for TensorRT engine bindings
        for model_name in self.stages:
            obj = self.model_dict[model_name]
            opt_batch_size = self.stage_batch_dict[model_name]["opt"]
            self.engine[model_name].allocate_buffers(
                shape_dict=obj.get_shape_dict(
                    opt_batch_size, image_height, image_width
                ),
                device=self.device
            )

    def __del__(self):
        for e in self.ddim_sampler.events.values():
            del e

        for engine in self.ddim_sampler.engine.values():
            del engine

        if self.shared_device_memory:
            self.shared_device_memory.free()
        print("engine free")

        self.ddim_sampler.stream.free()
        del self.stream
        del self.cuda_stream
        print("cuda stream free")

    def cached_model_name(self, model_name):
        # if self.inpaint:
        #     model_name += '_inpaint'
        return model_name

    def get_onnx_path(self, model_name, onnx_dir, opt=True):
        save_dir = os.path.join(onnx_dir, model_name) + ('_opt' if opt else '')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        return os.path.join(
            save_dir,
            self.cached_model_name(model_name) +'.onnx'
        )

    def get_engine_path(self, model_name, engine_dir):
        return os.path.join(
            engine_dir,
            self.cached_model_name(model_name) + '.plan'
        )

    def load_engines(
        self,
        # model_dict,
        engine_dir,
        onnx_dir,
        onnx_opset,
        opt_image_height,
        opt_image_width,
        force_export=False,
        force_optimize=False,
        force_build=False,
        static_batch=False,
        static_shape=False,
        enable_preview=True,
        enable_all_tactics=False,
        timing_cache=None,
    ):
        """
        Build and load engines for TensorRT accelerated inference.
        Export ONNX models first, if applicable.

        Args:
            engine_dir (str)max score is:  6.631080450426229:
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
        }
        print("model args: ", models_args)
        # Export models to ONNX
        for model_name in self.stages:
            obj = self.model_dict[model_name]
            engine_path = self.get_engine_path(model_name, engine_dir)
            if force_export or force_build or not os.path.exists(engine_path):
                onnx_path = self.get_onnx_path(model_name, onnx_dir, opt=False)
                onnx_opt_path = self.get_onnx_path(model_name, onnx_dir)
                if force_export or not os.path.exists(onnx_opt_path):
                    if force_export or not os.path.exists(onnx_path):
                        print(f"Exporting model: {onnx_path}")
                        model = obj.get_model()
                        opt_batch_size = self.stage_batch_dict[model_name]["opt"]
                        # with torch.inference_mode(), torch.autocast("cuda"):
                        with torch.no_grad():
                            inputs = obj.get_sample_input(
                                opt_batch_size,
                                opt_image_height,
                                opt_image_width
                            )
                            torch.onnx.export(
                                model,
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
                        # if self.onnx_device_raw == "cpu":
                        onnx_opt_graph = obj.optimize(onnx.load(
                            onnx_path,
                            load_external_data=False
                        ))
                        onnx.save(onnx_opt_graph, onnx_opt_path)
                        onnx_model_dir = os.path.dirname(onnx_path)
                        onnx_opt_model_dir = os.path.dirname(onnx_opt_path)
                        for file in os.listdir(onnx_model_dir):
                            file_path = os.path.join(onnx_model_dir, file)
                            if file_path == onnx_path:
                                continue
                            new_file_path = os.path.join(onnx_opt_model_dir, file)
                            shutil.copy(file_path, new_file_path)
                        # else:
                        #     onnx_opt_graph = obj.optimize(onnx.load(onnx_path))
                        #     onnx.save(onnx_opt_graph, onnx_opt_path)
                    else:
                        print(f"Found cached optimized model: {onnx_opt_path} ")
        
        # clear old pytorch model
        print("=" * 20)
        for model_name in self.stages:
            obj = self.model_dict[model_name]
            obj.model = obj.model.to("cpu")
            print(f"clear old {model_name} pytorch model")
            delattr(obj, "model")
            gc.collect()
            torch.cuda.empty_cache()
        print("=" * 20)

        # Build TensorRT engines
        for model_name in self.stages:
            obj = self.model_dict[model_name]
            engine_path = self.get_engine_path(model_name, engine_dir)
            engine = Engine(engine_path)
            onnx_path = self.get_onnx_path(
                model_name,
                onnx_dir,
                opt=False
            )
            # export TensorRT engine
            onnx_opt_path = self.get_onnx_path(
                model_name, onnx_dir
            )
            opt_batch_size = self.stage_batch_dict[model_name]["opt"]
            if force_build or not os.path.exists(engine.engine_path):
                if model_name == "clip":
                    use_fp16 = False
                else:
                    use_fp16 = True
                # if model_name == 'union_model':
                #     use_sparse_weights=True
                # else:
                #     use_sparse_weights=False
                engine.build(
                    onnx_opt_path,
                    # fp16=True,
                    fp16=use_fp16,
                    sparse_weights=False,
                    input_profile=obj.get_input_profile(
                        opt_batch_size, opt_image_height, opt_image_width,
                        static_batch=static_batch,
                        static_shape=static_shape
                    ),
                    # enable_refit=enable_refit,
                    enable_refit=False,
                    enable_preview=enable_preview,
                    enable_all_tactics=enable_all_tactics,
                    timing_cache=timing_cache,
                    workspace_size=self.max_workspace_size,
                    builder_optimization_level=self.builder_optimization_level
                )
            self.engine[model_name] = engine
            if self.do_compare:
                # Compare the accuracy difference between onnx and engine
                print("")
                print("------ start compare acc of {} -----".format(model_name))
                self.compare_acc(
                    model_name,
                    onnx_path,
                    engine_path,
                    obj,
                    opt_batch_size,
                    opt_image_height,
                    opt_image_width,
                    static_batch,
                    static_shape
                )
                print("------ end compare acc of {} ------".format(model_name))
                print("")


        # Load and activate TensorRT engines
        max_device_memory = 0
        for model_name in self.stages:
            engine = self.engine[model_name]
            engine.load()
            max_device_memory = max(
                max_device_memory,
                engine.engine.device_memory_size
            )
            # if onnx_refit_dir:
            #     onnx_refit_path = self.getOnnxPath(model_name, onnx_refit_dir)
            #     if os.path.exists(onnx_refit_path):
            #         engine.refit(onnx_opt_path, onnx_refit_path)

        print("max device memory: ", max_device_memory)
        print("max device memory (GB): ", round(max_device_memory / (1 << 30), 2))
        self.shared_device_memory = cuda.DeviceArray.raw(
            (int(max_device_memory),)
        )
        for engine in self.engine.values():
            engine.activate(reuse_device_memory=self.shared_device_memory.ptr)

    def compare_acc(
            self,
            model_name, #["clip", "control_net", "unet", "vae"]
            onnx_path,
            engine_path,
            obj,
            opt_batch_size,
            opt_image_height,
            opt_image_width,
            static_batch,
            static_shape,
            use_fp16=False,
        ):
        input_profile=obj.get_input_profile(
            opt_batch_size, opt_image_height, opt_image_width,
            static_batch=static_batch, static_shape=static_shape
        )

        input_metadata=TensorMetadata()
        for key in input_profile.keys():
            if key == "timestep":
                dtype = np.int32
            elif use_fp16:
                dtype = np.float16
            else:
                dtype = np.float32
            input_metadata.add(
                key,
                dtype=dtype,
                shape=input_profile[key][1],
                min_shape=input_profile[key][0],
                max_shape=input_profile[key][2],
            )

        data_loader = DataLoader(input_metadata=input_metadata)

        #Loaders
        build_onnxrt_session = SessionFromOnnx(onnx_path, providers=["CPUExecutionProvider"])
        engine_bytes = BytesFromPath(engine_path)
        deserialize_engine = EngineFromBytes(engine_bytes)

        # Runners
        runners = [
            OnnxrtRunner(build_onnxrt_session),
            TrtRunner(deserialize_engine),
        ]

        compare_func = CompareFunc.simple(rtol={'': 5e-2}, atol={'': 5e-2})
        #Comparator
        run_results = Comparator.run(runners, data_loader=data_loader)
        Comparator.compare_accuracy(run_results, compare_func=compare_func)
    
    def print_sumary(self, ddim_num_steps: int):
        print('|------------|--------------|')
        print('| {:^25} | {:^12} |'.format('Module', 'Latency'))
        print('|------------|--------------|')
        print('| {:^25} | {:>9.2f} ms |'.format(
            'CLIP',
            cudart.cudaEventElapsedTime(
                self.events['clip-start'],
                self.events['clip-stop']
            )[1]
        ))
        for index in range(ddim_num_steps):
            print('| {:^25} | {:>9.2f} ms |'.format(
                'ControlNet_{} + Unet_{}'.format(index, index),
                cudart.cudaEventElapsedTime(
                    self.events[f'union_model_{index}-start'],
                    self.events[f'union_model_{index}-stop']
                )[1]
            ))
        print('| {:^25} | {:>9.2f} ms |'.format(
            'VAE',
            cudart.cudaEventElapsedTime(
                self.events['vae-start'],
                self.events['vae-stop']
            )[1]
        ))
    
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
        high_threshold: float,
    ):
        # 由于guess mode计算和非guess mode不一样，所以这里加入判断
        if self.last_guess_mode != guess_mode:
            # 如果这次和上次guess mode不一样, UnionModel需要重新分配shape
            model_name = "union_model"
            obj = self.model_dict[model_name]
            image_height = 256
            image_width = 384
            if guess_mode:
                min_batch_size = self.stage_batch_dict[model_name]["min"]
                self.engine[model_name].allocate_buffers(
                    shape_dict=obj.get_shape_dict(
                        min_batch_size, image_height, image_width
                    ),
                    device=self.device
                )
            else:
                opt_batch_size = self.stage_batch_dict[model_name]["opt"]
                self.engine[model_name].allocate_buffers(
                    shape_dict=obj.get_shape_dict(
                        opt_batch_size, image_height, image_width
                    ),
                    device=self.device
                )
            # shape变化，如果有cuda graph，则需要重新录制
            shape_change = True
        else:
            shape_change = False
        if guess_mode:
            # guess mode很难校准，还是用原版步数
            ddim_num_steps = ddim_steps
        else:
            ddim_num_steps = 12
        self.last_guess_mode = guess_mode
        with torch.no_grad():
            img = resize_image(HWC3(input_image), image_resolution)
            H, W, C = img.shape

            detected_map = self.apply_canny(img, low_threshold, high_threshold)
            detected_map = HWC3(detected_map)

            control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
            if seed == -1:
                seed = random.randint(0, 65535)
            seed_everything(seed)
            if self.do_summarize:
                cudart.cudaEventRecord(self.events['clip-start'], 0)
            text_list = [prompt + ', ' + a_prompt] * num_samples + \
                [n_prompt] * num_samples
            batch_crossattn = self.text_embedding(text_list)
            if self.do_summarize:
                cudart.cudaEventRecord(self.events['clip-stop'], 0)
            samples = self.ddim_sampler.sample(
                control=control,
                batch_crossattn=batch_crossattn,
                ddim_num_steps=ddim_num_steps,
                guess_mode=guess_mode,
                strength=strength,
                eta=eta,
                uncond_scale=scale,
                batch_size=num_samples,
                shape_change=shape_change,
            )
            if self.do_summarize:
                cudart.cudaEventRecord(self.events['vae-start'], 0)
            x_samples = self.ddim_sampler.decode_first_stage(samples)
            if self.do_summarize:
                cudart.cudaEventRecord(self.events['vae-stop'], 0)
            x_samples = (
                einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5
            ).clip(0, 255).to(torch.uint8).cpu().numpy()

            results = [x_samples[i] for i in range(num_samples)]
            if self.do_summarize:
                self.print_sumary(ddim_num_steps=ddim_num_steps)
        return results
    

if __name__ == "__main__":
    hk = hackathon(do_compare=True, onnx_device="cpu") 
    hk.initialize()


