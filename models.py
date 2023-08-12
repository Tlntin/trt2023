#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import einops
import numpy as np
from onnx import shape_inference
import onnx_graphsurgeon as gs
from polygraphy.backend.onnx.loader import fold_constants
import torch
import onnx
import os

class Optimizer():
    def __init__(
        self,
        onnx_graph,
        verbose=False
    ):
        self.graph = gs.import_onnx(onnx_graph)
        self.verbose = verbose

    def info(self, prefix):
        if self.verbose:
            print(f"{prefix} .. {len(self.graph.nodes)} nodes, {len(self.graph.tensors().keys())} tensors, {len(self.graph.inputs)} inputs, {len(self.graph.outputs)} outputs")

    def cleanup(self, return_onnx=False):
        self.graph.cleanup().toposort()
        if return_onnx:
            return gs.export_onnx(self.graph)

    def select_outputs(self, keep, names=None):
        self.graph.outputs = [self.graph.outputs[o] for o in keep]
        if names:
            for i, name in enumerate(names):
                self.graph.outputs[i].name = name

    def fold_constants(self, return_onnx=False):
        onnx_graph = fold_constants(gs.export_onnx(self.graph), allow_onnxruntime_shape_inference=True)
        self.graph = gs.import_onnx(onnx_graph)
        if return_onnx:
            return onnx_graph

    def infer_shapes(self, return_onnx=False):
        onnx_graph = gs.export_onnx(self.graph)
        if onnx_graph.ByteSize() > 2147483648:
            raise TypeError("ERROR: model size exceeds supported 2GB limit")
        else:
            onnx_graph = shape_inference.infer_shapes(onnx_graph)

        self.graph = gs.import_onnx(onnx_graph)
        if return_onnx:
            return onnx_graph

def get_embedding_dim(version):
    if version in ("1.4", "1.5"):
        return 768
    elif version in ("2.0", "2.0-base", "2.1", "2.1-base"):
        return 1024
    else:
        raise ValueError(f"Incorrect version {version}")

class BaseModel():
    def __init__(
        self,
        fp16=False,
        device='cuda',
        verbose=True,
        min_batch_size=1,
        max_batch_size=16,
        embedding_dim=768,
        text_maxlen=77,
    ):
        self.name = "SD Model"
        self.fp16 = fp16
        self.device = device
        self.verbose = verbose

        self.min_batch = min_batch_size
        self.max_batch = max_batch_size
        self.min_image_shape = 256   # min image resolution: 256x256
        self.max_image_shape = 512  # max image resolution: 512x512
        self.min_latent_shape = self.min_image_shape // 8
        self.max_latent_shape = self.max_image_shape // 8

        self.embedding_dim = embedding_dim
        self.text_maxlen = text_maxlen

    def get_model(self):
        pass

    def get_input_names(self):
        pass

    def get_output_names(self):
        pass

    def get_dynamic_axes(self):
        return None

    def get_sample_input(self, batch_size, image_height, image_width):
        pass

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        return None

    def get_shape_dict(self, batch_size, image_height, image_width):
        return None

    def optimize(self, onnx_graph):
        opt = Optimizer(onnx_graph, verbose=self.verbose)
        opt.info(self.name + ': original')
        opt.cleanup()
        opt.info(self.name + ': cleanup')
        opt.fold_constants()
        opt.info(self.name + ': fold constants')
        opt.infer_shapes()
        opt.info(self.name + ': shape inference')
        onnx_opt_graph = opt.cleanup(return_onnx=True)
        opt.info(self.name + ': finished')
        return onnx_opt_graph

    def check_dims(self, batch_size, image_height, image_width):
        assert batch_size >= self.min_batch and batch_size <= self.max_batch
        assert image_height % 8 == 0 or image_width % 8 == 0
        latent_height = image_height // 8
        latent_width = image_width // 8
        assert latent_height >= self.min_latent_shape and latent_height <= self.max_latent_shape
        assert latent_width >= self.min_latent_shape and latent_width <= self.max_latent_shape
        return (latent_height, latent_width)

    def get_minmax_dims(self, batch_size, image_height, image_width, static_batch, static_shape):
        min_batch = batch_size if static_batch else self.min_batch
        max_batch = batch_size if static_batch else self.max_batch
        latent_height = image_height // 8
        latent_width = image_width // 8
        min_image_height = image_height if static_shape else self.min_image_shape
        max_image_height = image_height if static_shape else self.max_image_shape
        min_image_width = image_width if static_shape else self.min_image_shape
        max_image_width = image_width if static_shape else self.max_image_shape
        min_latent_height = latent_height if static_shape else self.min_latent_shape
        max_latent_height = latent_height if static_shape else self.max_latent_shape
        min_latent_width = latent_width if static_shape else self.min_latent_shape
        max_latent_width = latent_width if static_shape else self.max_latent_shape
        return (min_batch, max_batch, min_image_height, max_image_height, min_image_width, max_image_width, min_latent_height, max_latent_height, min_latent_width, max_latent_width)

class CLIP(BaseModel):
    def __init__(self,
        model,
        device,
        verbose,
        min_batch_size,
        max_batch_size,
        embedding_dim
    ):
        super(CLIP, self).__init__(
            device=device,
            verbose=verbose,
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
            embedding_dim=embedding_dim
        )
        self.name = "CLIP"
        self.model = model
    
    def get_model(self):
        return self.model.transformer

    def get_input_names(self):
        return ['input_ids']

    def get_output_names(self):
       return ["text_embeddings", "pooler_output"]

    def get_dynamic_axes(self):
        return {
            'input_ids': {0: '2B'},
            'text_embeddings': {0: '2B'}
        }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        assert batch_size % 2 == 0
        self.check_dims(batch_size, image_height, image_width)
        min_batch, max_batch, _, _, _, _, _, _, _, _ = self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        return {
            'input_ids': [(min_batch, self.text_maxlen), (batch_size, self.text_maxlen), (max_batch, self.text_maxlen)]
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        assert batch_size % 2 == 0
        self.check_dims(batch_size, image_height, image_width)
        return {
            'input_ids': (batch_size, self.text_maxlen),
            'text_embeddings': (batch_size, self.text_maxlen, self.embedding_dim)
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        assert batch_size % 2 == 0
        self.check_dims(batch_size, image_height, image_width)
        return torch.zeros(
            batch_size, self.text_maxlen, dtype=torch.int32, device=self.device
        )
        
    def optimize(self, onnx_graph):
        # change onnx -inf to -1e6 to support fp16
        for node in onnx_graph.graph.node:
            # if node.name == "/text_model/ConstantOfShape_1":
            if node.op_type == "ConstantOfShape":
                attr = node.attribute[0]
                if attr.name == "value" and attr.t.data_type == onnx.TensorProto.FLOAT:
                    np_array = np.frombuffer(attr.t.raw_data, dtype=np.float32).copy()
                    print("raw array", np_array)
                    np_array[np_array == -np.inf] = -100000
                    attr.t.raw_data = np_array.tobytes() 
                    print("new array", np_array)
        opt = Optimizer(onnx_graph, verbose=self.verbose)
        opt.info(self.name + ': original')
        opt.select_outputs([0]) # delete graph output#1
        opt.cleanup()
        opt.info(self.name + ': remove output[1]')
        opt.fold_constants()
        opt.info(self.name + ': fold constants')
        opt.infer_shapes()
        opt.info(self.name + ': shape inference')
        opt.select_outputs([0], names=['text_embeddings']) # rename network output
        opt.info(self.name + ': remove output[0]')
        opt_onnx_graph = opt.cleanup(return_onnx=True)
        opt.info(self.name + ': finished')
        return opt_onnx_graph


class ControlNet(BaseModel):
    def __init__(self,
        model,
        fp16=False,
        device='cuda',
        verbose=True,
        min_batch_size=1,
        max_batch_size=16,
        embedding_dim=768,
        text_maxlen=77,
        unet_dim=4
    ):
        super(ControlNet, self).__init__(
            fp16=fp16,
            device=device,
            verbose=verbose,
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
            embedding_dim=embedding_dim,
            text_maxlen=text_maxlen
        )
        self.unet_dim = unet_dim
        self.name = "ControlNet"
        self.model = model

    def get_model(self):
        # model_opts = {'revision': 'fp16', 'torch_dtype': torch.float16} if self.fp16 else {}
        if self.fp16:
            self.model = self.model.to(dtype=torch.float16, revision="fp16")
        return self.model

    def get_input_names(self):
        return [
            "sample",
            "hint",
            "timestep",
            "context"
        ]

    def get_output_names(self):
        return [
            "control_0",
            "control_1",
            "control_2",
            "control_3",
            "control_4",
            "control_5",
            "control_6",
            "control_7",
            "control_8",
            "control_9",
            "control_10",
            "control_11",
            "control_12",
        ]

    def get_dynamic_axes(self):
        return {
            'sample': {0: 'B', 2: '8H', 3: '8W'},
            "hint": {0: 'B', 2: 'height', 3: 'width'},
            "timestep": {0: 'B'},
            "context": {0: 'B'},
            "control_0": {0: "B", 2: "8H", 3: "8W"},
            "control_1": {0: "B", 2: "8H", 3: "8W"},
            "control_2": {0: "B", 2: "8H", 3: "8W"},
            "control_3": {0: "B", 2: "4H", 3: "4W"},
            "control_4": {0: "B", 2: "4H", 3: "4W"},
            "control_5": {0: "B", 2: "4H", 3: "4W"},
            "control_6": {0: "B", 2: "2H", 3: "2W"},
            "control_7": {0: "B", 2: "2H", 3: "2W"},
            "control_8": {0: "B", 2: "2H", 3: "2W"},
            "control_9": {0: "B", 2: "H", 3: "W"},
            "control_10": {0: "B", 2: "H", 3: "W"},
            "control_11": {0: "B", 2: "H", 3: "W"},
            "control_12": {0: "B", 2: "H", 3: "W"},
        }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        (
            min_batch, max_batch,
            min_image_height, max_image_height,
            min_image_width, max_image_width,
            min_latent_height, max_latent_height,
            min_latent_width, max_latent_width 
        ) = \
            self.get_minmax_dims(
                batch_size,
                image_height,
                image_width,
                static_batch,
                static_shape
            )
        return {
            'sample': [
                (min_batch, self.unet_dim, min_latent_height, min_latent_width), # min
                (batch_size, self.unet_dim, latent_height, latent_width), # opt
                (max_batch, self.unet_dim, max_latent_height, max_latent_width) # max
            ],
            "hint": [
                (min_batch, 3, min_image_height, min_image_width), # min
                (batch_size, 3, image_height, image_width), # opt
                (max_batch, 3, max_image_height, max_image_width) # max
            ],
            "timestep": [
                (min_batch,),     
                (batch_size,),
                (max_batch,)
            ],
            "context": [
                (min_batch, self.text_maxlen, self.embedding_dim),
                (batch_size, self.text_maxlen, self.embedding_dim),
                (max_batch, self.text_maxlen, self.embedding_dim)
            ],
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            'sample': (batch_size, self.unet_dim, latent_height, latent_width),
            'hint': (batch_size, 3, image_height, image_width),
            "timestep": (batch_size,),
            "context": (batch_size, self.text_maxlen, self.embedding_dim),
            "control_0": (batch_size, 320, latent_height, latent_width),
            "control_1": (batch_size, 320, latent_height, latent_width),
            "control_2": (batch_size, 320, latent_height, latent_width),
            "control_3": (batch_size, 320, latent_height // 2, latent_width // 2),
            "control_4": (batch_size, 640, latent_height // 2, latent_width // 2),
            "control_5": (batch_size, 640, latent_height // 2, latent_width // 2),
            "control_6": (batch_size, 640, latent_height // 4, latent_width // 4),
            "control_7": (batch_size, 1280, latent_height // 4, latent_width // 4),
            "control_8": (batch_size, 1280, latent_height // 4, latent_width // 4),
            "control_9": (batch_size, 1280, latent_height // 8, latent_width // 8),
            "control_10": (batch_size, 1280, latent_height // 8, latent_width // 8),
            "control_11": (batch_size, 1280, latent_height // 8, latent_width // 8),
            "control_12": (batch_size, 1280, latent_height // 8, latent_width // 8),
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        dtype = torch.float16 if self.fp16 else torch.float32
        return (
            # 'sample': ['B', 4, 'latent_height', 'latent_width']
            torch.randn(
                batch_size,
                self.unet_dim,
                latent_height,
                latent_width,
                dtype=dtype,
                device=self.device
            ),
            # 'hint': ['B', 3, 'image_height', 'image_width']
            torch.randn(
                batch_size,
                3,
                image_height,
                image_width,
                dtype=dtype,
                device=self.device
            ),
            # "timestep": ['B'],
            torch.tensor([951.] * batch_size, dtype=dtype, device=self.device),
            # "context": ['B', 'T', 'E']
            torch.randn(
                batch_size,
                self.text_maxlen,
                self.embedding_dim,
                dtype=dtype,
                device=self.device
            )
        )


class UNet(BaseModel):
    def __init__(self,
        model,
        fp16=False,
        device='cuda',
        verbose=True,
        min_batch_size=1,
        max_batch_size=16,
        embedding_dim=768,
        text_maxlen=77,
        unet_dim=4
    ):
        super(UNet, self).__init__(
            fp16=fp16,
            device=device,
            verbose=verbose,
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
            embedding_dim=embedding_dim,
            text_maxlen=text_maxlen
        )
        self.unet_dim = unet_dim
        self.name = "UNet"
        self.model = model

    def get_model(self):
        # model_opts = {'revision': 'fp16', 'torch_dtype': torch.float16} if self.fp16 else {}
        if self.fp16:
            self.model = self.model.to(dtype=torch.float16, revision='fp16')
        return self.model

    def get_input_names(self):
        return [
            "sample",
            "timestep",
            "context",
            "control_0",
            "control_1",
            "control_2",
            "control_3",
            "control_4",
            "control_5",
            "control_6",
            "control_7",
            "control_8",
            "control_9",
            "control_10",
            "control_11",
            "control_12",
        ]

    def get_output_names(self):
       return ['latent']

    def get_dynamic_axes(self):
        return {
            'sample': {0: 'B', 2: '8H', 3: '8W'},
            "timestep": {0: 'B'},
            "context": {0: 'B'},
            "control_0": {0: "B", 2: "8H", 3: "8W"},
            "control_1": {0: "B", 2: "8H", 3: "8W"},
            "control_2": {0: "B", 2: "8H", 3: "8W"},
            "control_3": {0: "B", 2: "4H", 3: "4W"},
            "control_4": {0: "B", 2: "4H", 3: "4W"},
            "control_5": {0: "B", 2: "4H", 3: "4W"},
            "control_6": {0: "B", 2: "2H", 3: "2W"},
            "control_7": {0: "B", 2: "2H", 3: "2W"},
            "control_8": {0: "B", 2: "2H", 3: "2W"},
            "control_9": {0: "B", 2: "H", 3: "W"},
            "control_10": {0: "B", 2: "H", 3: "W"},
            "control_11": {0: "B", 2: "H", 3: "W"},
            "control_12": {0: "B", 2: "H", 3: "W"},
            'latent': {0: 'B', 2: '8H', 3: '8W'}
        }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        min_batch, max_batch, _, _, _, _, min_latent_height, max_latent_height, min_latent_width, max_latent_width = \
            self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        return {
            'sample': [
                (min_batch, self.unet_dim, min_latent_height, min_latent_width), # min
                (batch_size, self.unet_dim, latent_height, latent_width), # opt
                (max_batch, self.unet_dim, max_latent_height, max_latent_width) # max
            ],
            "timestep": [
                (min_batch,),     
                (batch_size,),
                (max_batch,)
            ],
            "context": [
                (min_batch, self.text_maxlen, self.embedding_dim),
                (batch_size, self.text_maxlen, self.embedding_dim),
                (max_batch, self.text_maxlen, self.embedding_dim)
            ],
            "control_0": [
                (min_batch, 320, min_latent_height, min_latent_width), # min
                (batch_size, 320, latent_height, latent_width), # opt
                (max_batch, 320, max_latent_height, max_latent_width) # max
            ],
            "control_1": [
                (min_batch, 320, min_latent_height, min_latent_width), # min
                (batch_size, 320, latent_height, latent_width), # opt
                (max_batch, 320, max_latent_height, max_latent_width) # max
            ],
            "control_2": [
                (min_batch, 320, min_latent_height, min_latent_width), # min
                (batch_size, 320, latent_height, latent_width), # opt
                (max_batch, 320, max_latent_height, max_latent_width) # max
            ],
            "control_3": [
                (min_batch, 320, min_latent_height // 2, min_latent_width // 2), # min
                (batch_size, 320, latent_height // 2, latent_width // 2), # opt
                (max_batch, 320, max_latent_height // 2, max_latent_width // 2) # max
            ],
            "control_4": [
                (min_batch, 640, min_latent_height // 2, min_latent_width // 2), # min
                (batch_size, 640, latent_height // 2, latent_width // 2), # opt
                (max_batch, 640, max_latent_height // 2, max_latent_width // 2) # max
            ],
            "control_5": [
                (min_batch, 640, min_latent_height // 2, min_latent_width // 2), # min
                (batch_size, 640, latent_height // 2, latent_width // 2), # opt
                (max_batch, 640, max_latent_height // 2, max_latent_width // 2) # max
            ],
            "control_6": [
                (min_batch, 640, min_latent_height // 4, min_latent_width // 4), # min
                (batch_size, 640, latent_height // 4, latent_width // 4), # opt
                (max_batch, 640, max_latent_height // 4, max_latent_width // 4) # max
            ],
            "control_7": [
                (min_batch, 1280, min_latent_height // 4, min_latent_width // 4), # min
                (batch_size, 1280, latent_height // 4, latent_width // 4), # opt
                (max_batch, 1280, max_latent_height // 4, max_latent_width // 4) # max
            ],
            "control_8": [
                (min_batch, 1280, min_latent_height // 4, min_latent_width // 4), # min
                (batch_size, 1280, latent_height // 4, latent_width // 4), # opt
                (max_batch, 1280, max_latent_height // 4, max_latent_width // 4) # max
            ],
            "control_9": [
                (min_batch, 1280, min_latent_height // 8, min_latent_width // 8), # min
                (batch_size, 1280, latent_height // 8, latent_width // 8), # opt
                (max_batch, 1280, max_latent_height // 8, max_latent_width // 8) # max
            ],
            "control_10": [
                (min_batch, 1280, min_latent_height // 8, min_latent_width // 8), # min
                (batch_size, 1280, latent_height // 8, latent_width // 8), # opt
                (max_batch, 1280, max_latent_height // 8, max_latent_width // 8) # max
            ],
            "control_11": [
                (min_batch, 1280, min_latent_height // 8, min_latent_width // 8), # min
                (batch_size, 1280, latent_height // 8, latent_width // 8), # opt
                (max_batch, 1280, max_latent_height // 8, max_latent_width // 8) # max
            ],
            "control_12": [
                (min_batch, 1280, min_latent_height // 8, min_latent_width // 8), # min
                (batch_size, 1280, latent_height // 8, latent_width // 8), # opt
                (max_batch, 1280, max_latent_height // 8, max_latent_width // 8) # max
            ],
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            'sample': (batch_size, self.unet_dim, latent_height, latent_width),
            "timestep": (batch_size,),
            "context": (batch_size, self.text_maxlen, self.embedding_dim),
            "control_0": (batch_size, 320, latent_height, latent_width),
            "control_1": (batch_size, 320, latent_height, latent_width),
            "control_2": (batch_size, 320, latent_height, latent_width),
            "control_3": (batch_size, 320, latent_height // 2, latent_width // 2),
            "control_4": (batch_size, 640, latent_height // 2, latent_width // 2),
            "control_5": (batch_size, 640, latent_height // 2, latent_width // 2),
            "control_6": (batch_size, 640, latent_height // 4, latent_width // 4),
            "control_7": (batch_size, 1280, latent_height // 4, latent_width // 4),
            "control_8": (batch_size, 1280, latent_height // 4, latent_width // 4),
            "control_9": (batch_size, 1280, latent_height // 8, latent_width // 8),
            "control_10": (batch_size, 1280, latent_height // 8, latent_width // 8),
            "control_11": (batch_size, 1280, latent_height // 8, latent_width // 8),
            "control_12": (batch_size, 1280, latent_height // 8, latent_width // 8),
            'latent': (batch_size, self.unet_dim, latent_height, latent_width)
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        dtype = torch.float16 if self.fp16 else torch.float32
        return (
            # 'sample': ['B', 4, '8H', '8W']
            torch.randn(
                batch_size,
                self.unet_dim,
                latent_height,
                latent_width,
                dtype=dtype,
                device=self.device
            ),
            # "timestep": ['B'],
            torch.tensor([951] * batch_size, dtype=torch.int32, device=self.device),
            # "context": ['B', 'T', 'E']
            torch.randn(
                batch_size,
                self.text_maxlen,
                self.embedding_dim,
                dtype=dtype,
                device=self.device
            ),
            [
                # "control_0": ["B", 320, "8H", "8W"],
                torch.randn(
                    batch_size,
                    320,
                    latent_height,
                    latent_width,
                    dtype=dtype,
                    device=self.device
                ),
                # "control_1": ["B", 320, "8H", "8W"],
                torch.randn(
                    batch_size,
                    320,
                    latent_height,
                    latent_width,
                    dtype=dtype,
                    device=self.device
                ),
                # "control_2": ["B", 320, "8H", "8W"],
                torch.randn(
                    batch_size,
                    320,
                    latent_height,
                    latent_width,
                    dtype=dtype,
                    device=self.device
                ),
                # "control_3": ["B", 320, "4H", "4W"],
                torch.randn(
                    batch_size,
                    320,
                    latent_height // 2,
                    latent_width // 2,
                    dtype=dtype,
                    device=self.device
                ),
                # "control_4": ["B", 640, "4H", "4W"],
                torch.randn(
                    batch_size,
                    640,
                    latent_height // 2,
                    latent_width // 2,
                    dtype=dtype,
                    device=self.device
                ),
                # "control_5": ["B", 640, "4H", "4W"],
                torch.randn(
                    batch_size,
                    640,
                    latent_height // 2,
                    latent_width // 2,
                    dtype=dtype,
                    device=self.device
                ),
                # "control_6": ["B", 640, "2H", "2W"],
                torch.randn(
                    batch_size,
                    640,
                    latent_height // 4,
                    latent_width // 4,
                    dtype=dtype,
                    device=self.device
                ),
                # "control_7": ["B", 1280, "2H", "2W"],
                torch.randn(
                    batch_size,
                    1280,
                    latent_height // 4,
                    latent_width // 4,
                    dtype=dtype,
                    device=self.device
                ),
                # "control_8": ["B", 1280, "2H", "2W"],
                torch.randn(
                    batch_size,
                    1280,
                    latent_height // 4,
                    latent_width // 4,
                    dtype=dtype,
                    device=self.device
                ),
                # "control_9": ["B", 1280, "H", "W"],
                torch.randn(
                    batch_size,
                    1280,
                    latent_height // 8,
                    latent_width // 8,
                    dtype=dtype,
                    device=self.device
                ),
                # "control_10": ["B", 1280, "H", "W"],
                torch.randn(
                    batch_size,
                    1280,
                    latent_height // 8,
                    latent_width // 8,
                    dtype=dtype,
                    device=self.device
                ),
                # "control_11": ["B", 1280, "H", "W"],
                torch.randn(
                    batch_size,
                    1280,
                    latent_height // 8,
                    latent_width // 8,
                    dtype=dtype,
                    device=self.device
                ),
                # "control_12": ["B", 1280, "H", "W"],
                torch.randn(
                    batch_size,
                    1280,
                    latent_height // 8,
                    latent_width // 8,
                    dtype=dtype,
                    device=self.device
                ),
            ]
        )


class UNet_V2(BaseModel):
    """
    this Unet is build with no control input
    """
    def __init__(self,
        model,
        fp16=False,
        device='cuda',
        verbose=True,
        min_batch_size=1,
        max_batch_size=16,
        embedding_dim=768,
        text_maxlen=77,
        unet_dim=4
    ):
        super(UNet_V2, self).__init__(
            fp16=fp16,
            device=device,
            verbose=verbose,
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
            embedding_dim=embedding_dim,
            text_maxlen=text_maxlen
        )
        self.unet_dim = unet_dim
        self.name = "UNet_V2"
        self.model = model

    def get_model(self):
        # model_opts = {'revision': 'fp16', 'torch_dtype': torch.float16} if self.fp16 else {}
        if self.fp16:
            self.model = self.model.to(dtype=torch.float16, revision='fp16')
        return self.model

    def get_input_names(self):
        return [
            "sample",
            "timestep",
            "context",
        ]

    def get_output_names(self):
       return ['latent']

    def get_dynamic_axes(self):
        return {
            "sample": {0: 'B', 2: '8H', 3: '8W'},
            "timestep": {0: 'B'},
            "context": {0: 'B'},
            'latent': {0: 'B', 2: '8H', 3: '8W'}
        }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        min_batch, max_batch, _, _, _, _, min_latent_height, max_latent_height, min_latent_width, max_latent_width = \
            self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        return {
            'sample': [
                (min_batch, self.unet_dim, min_latent_height, min_latent_width), # min
                (batch_size, self.unet_dim, latent_height, latent_width), # opt
                (max_batch, self.unet_dim, max_latent_height, max_latent_width) # max
            ],
            "timestep": [
                (min_batch,),     
                (batch_size,),
                (max_batch,)
            ],
            "context": [
                (min_batch, self.text_maxlen, self.embedding_dim),
                (batch_size, self.text_maxlen, self.embedding_dim),
                (max_batch, self.text_maxlen, self.embedding_dim)
            ]
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            'sample': (batch_size, self.unet_dim, latent_height, latent_width),
            "timestep": (batch_size,),
            "context": (batch_size, self.text_maxlen, self.embedding_dim),
            'latent': (batch_size, self.unet_dim, latent_height, latent_width)
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        dtype = torch.float16 if self.fp16 else torch.float32
        return (
            # 'sample': ['B', 4, '8H', '8W']
            torch.randn(
                batch_size,
                self.unet_dim,
                latent_height,
                latent_width,
                dtype=dtype,
                device=self.device
            ),
            # "timestep": ['B'],
            torch.tensor([951] * batch_size, dtype=torch.int32, device=self.device),
            # "context": ['B', 'T', 'E']
            torch.randn(
                batch_size,
                self.text_maxlen,
                self.embedding_dim,
                dtype=dtype,
                device=self.device
            ),
        )


class MyTempModel(torch.nn.Module):
    def __init__(self, control_model, unet_model):
        super(MyTempModel, self).__init__()
        self.control_model = control_model
        self.unet_model = unet_model

    def forward(self, sample, hint, timestep, context, control_scales):
        control = self.control_model(sample, hint, timestep, context)
        # dtype = control[0].dtype
        # control = [
        #     c * scale.to(dtype=dtype)
        #     for c, scale in zip(control, control_scales)
        # ]
        # dtype = control[0].dtype
        control = [
            c * scale
            for c, scale in zip(control, control_scales)
        ]
        latent = self.unet_model(sample, timestep, context, control)
        return latent



class UnionModel(BaseModel):
    def __init__(self,
        control_model,
        unet_model,
        fp16=False,
        device='cuda',
        verbose=True,
        min_batch_size=1,
        max_batch_size=16,
        embedding_dim=768,
        text_maxlen=77,
        unet_dim=4
    ):
        super(UnionModel, self).__init__(
            fp16=fp16,
            device=device,
            verbose=verbose,
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
            embedding_dim=embedding_dim,
            text_maxlen=text_maxlen
        )
        self.unet_dim = unet_dim
        self.name = "UnionModel"
        self.model = MyTempModel(control_model, unet_model)

    def get_model(self):
        # model_opts = {'revision': 'fp16', 'torch_dtype': torch.float16} if self.fp16 else {}
        if self.fp16:
            self.model = self.model.to(device=self.device, dtype=torch.float16)
        return self.model

    def get_input_names(self):
        return [
            "sample",
            "hint",
            "timestep",
            "context",
            "control_scales"
        ]

    def get_output_names(self):
        return ['latent']

    def get_dynamic_axes(self):
        return {
            'sample': {0: '2B', 2: 'H', 3: 'W'},
            "hint": {0: '2B', 2: 'height', 3: 'width'},
            "timestep": {0: '2B'},
            "context": {0: '2B'},
            'latent': {0: '2B', 2: 'H', 3: 'W'}
        }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        assert batch_size % 2 == 0
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        (
            min_batch, max_batch,
            min_image_height, max_image_height,
            min_image_width, max_image_width,
            min_latent_height, max_latent_height,
            min_latent_width, max_latent_width 
        ) = \
            self.get_minmax_dims(
                batch_size,
                image_height,
                image_width,
                static_batch,
                static_shape
            )
        return {
            'sample': [
                (min_batch, self.unet_dim, min_latent_height, min_latent_width), # min
                (batch_size, self.unet_dim, latent_height, latent_width), # opt
                (max_batch, self.unet_dim, max_latent_height, max_latent_width) # max
            ],
            "hint": [
                (min_batch, 3, min_image_height, min_image_width), # min
                (batch_size, 3, image_height, image_width), # opt
                (max_batch, 3, max_image_height, max_image_width) # max
            ],
            "timestep": [
                (min_batch,),     
                (batch_size,),
                (max_batch,)
            ],
            "context": [
                (min_batch, self.text_maxlen, self.embedding_dim),
                (batch_size, self.text_maxlen, self.embedding_dim),
                (max_batch, self.text_maxlen, self.embedding_dim)
            ],
            "control_scales": [
                (13,),
                (13,),
                (13,),
            ]
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            'sample': (batch_size, self.unet_dim, latent_height, latent_width),
            'hint': (batch_size, 3, image_height, image_width),
            "timestep": (batch_size,),
            "context": (batch_size, self.text_maxlen, self.embedding_dim),
            "control_scales": [13],
            'latent': (batch_size, self.unet_dim, latent_height, latent_width)

        }

    def get_sample_input(self, batch_size, image_height, image_width):
        assert batch_size % 2 == 0
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        dtype = torch.float16 if self.fp16 else torch.float32
        return (
            # 'sample': ['B', 4, 'latent_height', 'latent_width']
            torch.randn(
                batch_size,
                self.unet_dim,
                latent_height,
                latent_width,
                dtype=dtype,
                device=self.device
            ),
            # 'hint': ['B', 3, 'image_height', 'image_width']
            torch.randn(
                batch_size,
                3,
                image_height,
                image_width,
                dtype=dtype,
                device=self.device
            ),
            # "timestep": ['B'],
            torch.tensor(
                [951] * batch_size, dtype=torch.int32, device=self.device
            ),
            # "context": ['B', 'T', 'E']
            torch.randn(
                batch_size,
                self.text_maxlen,
                self.embedding_dim,
                dtype=dtype,
                device=self.device
            ),
            # "control_scales": [13]
            torch.ones([13], dtype=dtype, device=self.device)
        )


class UnionBlock(torch.nn.Module):
    def __init__(self, control_model, unet_model):
        super(UnionBlock, self).__init__()
        self.control_model = control_model
        self.unet_model = unet_model

    def forward(
        self,
        x,
        hint,
        t,
        context,
        alphas,
        alphas_prev,
        sqrt_one_minus_alphas,
        noise,
        temp_di,
        uncond_scale: torch.Tensor,
    ):
        b = x.shape[0] // 2
        # do like this like self.apply_model
        control = self.control_model(x, hint, t, context)
        b_latent = self.unet_model(x, t, context, control)
        e_t = b_latent[b:] + uncond_scale * (b_latent[:b] - b_latent[b:])
        pred_x0 = (x - sqrt_one_minus_alphas * e_t) / alphas
        # direction pointing to x_t
        dir_xt = temp_di * e_t
        x = alphas_prev * pred_x0 + dir_xt + noise
        return x


class UnionBlockPT(torch.nn.Module):
    def __init__(self, control_model, unet_model):
        super(UnionBlockPT, self).__init__()
        self.control_model = control_model
        self.unet_model = unet_model
        now_dir = os.path.dirname(os.path.abspath(__file__))
        carlibre_dir = os.path.join(now_dir, "output", "calibre")
        self.union_model_v2_dir = os.path.join(carlibre_dir, "union_model_v2")
        self.unet_dir = os.path.join(carlibre_dir, "unet")
        self.control_dir = os.path.join(carlibre_dir, "control_net")
        for dir1 in [
            carlibre_dir, self.union_model_v2_dir, self.unet_dir, self.control_dir
        ]:
            if not os.path.exists(dir1):
                os.mkdir(dir1)

    def forward(
        self,
        x,
        hint,
        t,
        context,
        control_scales,
        save_sample = False,
        model_name = "union_model_v2",
    ):

        if save_sample and model_name == "union_model_v2":
            input_dict_1 = {
                "sample": x.detach().cpu().data.numpy(),
                "hint": hint.detach().cpu().data.numpy(),
                "timestep": t.detach().cpu().data.numpy(),
                "context": context.detach().cpu().data.numpy(),
            }
            file_list = os.listdir(self.union_model_v2_dir)
            file_list = [file for file in file_list if file.endswith(".npz")]
            f_index = len(file_list)
            input_path = os.path.join(self.union_model_v2_dir, f"{f_index}.npz")
            np.savez(input_path, **input_dict_1)
        if save_sample and model_name != "union_model_v2":
            input_dict_2 = {
                "sample": x.detach().cpu().data.numpy(),
                "hint": hint.detach().cpu().data.numpy(),
                "timestep": t.detach().cpu().data.numpy(),
                "context": context.detach().cpu().data.numpy(),
            }
            file_list = os.listdir(self.control_dir)
            file_list = [file for file in file_list if file.endswith(".npz")]
            f_index = len(file_list)
            input_path = os.path.join(self.control_dir, f"{f_index}.npz")
            np.savez(input_path, **input_dict_2)
        # do like this like self.apply_model
        control = self.control_model(x, hint, t, context)
        # sample for unet model
        if save_sample and model_name != "union_model_v2":
            input_dict_3 = {
                "sample": x.detach().cpu().data.numpy(),
                "timestep": t.detach().cpu().data.numpy(),
                "context": context.detach().cpu().data.numpy(),
            }
            for i, c in enumerate(control):
                input_dict_3[f"control_{i}"] = c.detach().cpu().data.numpy()
            file_list = os.listdir(self.unet_dir)
            file_list = [file for file in file_list if file.endswith(".npz")]
            f_index = len(file_list)
            input_path = os.path.join(self.unet_dir, f"{f_index}.npz")
            np.savez(input_path, **input_dict_3)
        control = [c * scale for c, scale in zip(control, control_scales)]
        b_latent = self.unet_model(x, t, context, control)
        return b_latent



class UnionModelV2(BaseModel):
    def __init__(self,
        control_model,
        unet_model,
        fp16=False,
        device='cuda',
        verbose=True,
        min_batch_size=1,
        max_batch_size=16,
        embedding_dim=768,
        text_maxlen=77,
        unet_dim=4
    ):
        super(UnionModelV2, self).__init__(
            fp16=fp16,
            device=device,
            verbose=verbose,
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
            embedding_dim=embedding_dim,
            text_maxlen=text_maxlen
        )
        self.unet_dim = unet_dim
        self.name = "UnionModel"
        self.model = UnionBlock(control_model, unet_model).to(device)

    def get_model(self):
        # model_opts = {'revision': 'fp16', 'torch_dtype': torch.float16} if self.fp16 else {}
        if self.fp16:
            self.model = self.model.to(
                device=self.device,
                dtype=torch.float16,
                revision=torch.float16
            )
        return self.model

    def get_input_names(self):
        return [
            "sample",
            "hint",
            "timestep",
            "context",
            "alphas",
            "alphas_prev",
            "sqrt_one_minus_alphas",
            "noise",
            "temp_di",
            "uncond_scale",
        ]

    def get_output_names(self):
        return ['latent']

    def get_dynamic_axes(self):
        return {
            'sample': {0: '2B', 2: 'H', 3: 'W'},
            "hint": {0: '2B', 2: 'height', 3: 'width'},
            "timestep": {0: "2B"},
            "context": {0: '2B'},
            "noise": {2: "H", 3: "W"},
            'latent': {0: '2B', 2: 'H', 3: 'W'}
        }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        (
            min_batch, max_batch,
            min_image_height, max_image_height,
            min_image_width, max_image_width,
            min_latent_height, max_latent_height,
            min_latent_width, max_latent_width 
        ) = \
            self.get_minmax_dims(
                batch_size,
                image_height,
                image_width,
                static_batch,
                static_shape
            )
        return {
            'sample': [
                (min_batch * 2, self.unet_dim, min_latent_height, min_latent_width), # min
                (batch_size * 2, self.unet_dim, latent_height, latent_width), # opt
                (max_batch * 2, self.unet_dim, max_latent_height, max_latent_width) # max
            ],
            "hint": [
                (min_batch * 2, 3, min_image_height, min_image_width), # min
                (batch_size * 2, 3, image_height, image_width), # opt
                (max_batch * 2, 3, max_image_height, max_image_width) # max
            ],
            "timestep": [
                (batch_size * 2, ),     
                (batch_size * 2, ),
                (batch_size * 2, )
            ],
            "context": [
                (min_batch * 2, self.text_maxlen, self.embedding_dim),
                (batch_size * 2, self.text_maxlen, self.embedding_dim),
                (max_batch * 2, self.text_maxlen, self.embedding_dim)
            ],
            "alphas": [
                (1, ),     
                (1, ),
                (1, )
            ],
            "alphas_prev":[
                (1, ),     
                (1, ),
                (1, )
            ],
            "sqrt_one_minus_alphas": [
                (1, ),     
                (1, ),
                (1, )
            ],
            "noise": [
                (1, self.unet_dim, min_latent_height, min_latent_width),
                (1, self.unet_dim, latent_height, latent_width),
                (1, self.unet_dim, max_latent_height, max_latent_width)
            ],
            "temp_di": [
                (1, ),
                (1, ),
                (1, ),
            ],
            "uncond_scale": [
                (1, ),
                (1, ),
                (1, )
            ],
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            'sample': (batch_size * 2, self.unet_dim, latent_height, latent_width),
            'hint': (batch_size * 2, 3, image_height, image_width),
            "timestep": (2 * batch_size, ),
            "context": (batch_size * 2, self.text_maxlen, self.embedding_dim),
            "alphas": (1, ),
            "alphas_prev": (1, ),
            "sqrt_one_minus_alphas": (1, ),
            "noise": (1, self.unet_dim, latent_height, latent_width),
            "temp_di": (1, ),
            "uncond_scale": (1,),
            'latent': (batch_size * 2, self.unet_dim, latent_height, latent_width)
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        dtype = torch.float16 if self.fp16 else torch.float32
        return (
            # 'sample': ['2B', 4, 'latent_height', 'latent_width']
            torch.randn(
                batch_size * 2,
                self.unet_dim,
                latent_height,
                latent_width,
                dtype=dtype,
                device=self.device
            ),
            # 'hint': ['2B', 3, 'image_height', 'image_width']
            torch.randn(
                batch_size * 2,
                3,
                image_height,
                image_width,
                dtype=dtype,
                device=self.device
            ),
            # "timestep": ["2B"],
            torch.tensor([951, 951], dtype=dtype, device=self.device),
            # "context": ['2B', 'T', 'E']
            torch.randn(
                batch_size * 2,
                self.text_maxlen,
                self.embedding_dim,
                dtype=dtype,
                device=self.device
            ),
            # alphas: [4]
            torch.randn(1, dtype=dtype, device=self.device),
            # alphas_prev: [4]
            torch.randn(1, dtype=dtype, device=self.device),
            # sqrt_one_minus_alphas: [4]
            torch.randn(1, dtype=dtype, device=self.device),
            # noise: [1, 4, H, W]
            torch.randn(
                (1, self.unet_dim, latent_height, latent_width),
                dtype=dtype,
                device=self.device,
            ),
            # temp_di
            torch.randn(1, dtype=dtype, device=self.device),
            # "uncond_scale":
            torch.tensor([9.0], dtype=dtype, device=self.device),
        )



class VAE(BaseModel):
    def __init__(self,
        model,
        device,
        verbose,
        min_batch_size,
        max_batch_size,
        embedding_dim
    ):
        super(VAE, self).__init__(
            device=device,
            verbose=verbose,
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
            embedding_dim=embedding_dim
        )
        self.name = "VAE decoder"
        self.model = model

    def get_model(self):
        self.model.forward = self.model.decode
        return self.model

    def get_input_names(self):
        return ['latent']

    def get_output_names(self):
       return ['images']

    def get_dynamic_axes(self):
        return {
            'latent': {0: 'B', 2: 'H', 3: 'W'},
            'images': {0: 'B', 2: '8H', 3: '8W'}
        }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        min_batch, max_batch, _, _, _, _, min_latent_height, max_latent_height, min_latent_width, max_latent_width = \
            self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        return {
            'latent': [
                (min_batch, 4, min_latent_height, min_latent_width),
                (batch_size, 4, latent_height, latent_width),
                (max_batch, 4, max_latent_height, max_latent_width)
            ]
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            'latent': (batch_size, 4, latent_height, latent_width),
            'images': (batch_size, 3, image_height, image_width)
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return (
            torch.randn(
                batch_size,
                4,
                latent_height,
                latent_width,
                dtype=torch.float32,
                device=self.device
            ),
        )



class SamplerModel(torch.nn.Module):
    def __init__(
            self,
            clip_model,
            control_model,
            unet_model,
            vae_model,
            num_timesteps,
            scale_factor,
            alphas_cumprod,
            schedule="linear",
        ):
        super().__init__()
        self.clip_model = clip_model
        self.union_model = UnionBlockPT(control_model, unet_model)
        self.vae_model = vae_model
        self.scale_factor = scale_factor
        self.alphas_cumprod = alphas_cumprod
        self.ddpm_num_timesteps = num_timesteps
        self.schedule = schedule
        device = clip_model.device
        # for guess mode
        self.guess_control_scales = torch.from_numpy(
            np.array(
                [
                    (0.825 ** float(12 - i)) for i in range(13)
                ],
                dtype=np.float32,
            ),
        ).to(dtype=torch.float32, device=device)
        # for un guess mode
        self.unguess_control_scales = torch.ones(
            [13], dtype=torch.float32, device=device)

    def sample(
        self,
        control: torch.Tensor,
        input_ids: torch.Tensor,
        uncond_scale: torch.Tensor,
        guess_mode,
        strength,
        eta = 0.0,
        ddim_num_steps = 20,
        temperature: float = 1.,
        batch_size: int = 1,
        save_sample=False,
        model_name="union_model_v2",
    ):
        h, w, c = control.shape
        device = control.device
        shape = (batch_size, 4, h // 8, w // 8)
        control = control.unsqueeze(0)
        control = einops.rearrange(control, 'b h w c -> b c h w')
        clip_outputs = self.clip_model(input_ids)
        batch_crossattn = clip_outputs.last_hidden_state
        # --- copy from make schedule ---
        c = self.ddpm_num_timesteps // ddim_num_steps
        ddim_timesteps = torch.arange(
            1, self.ddpm_num_timesteps + 1, c,
            dtype=torch.int32,
            device=device
        )
        ddim_sampling_tensor = ddim_timesteps.unsqueeze(1).repeat(1, 2 * batch_size).float()
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
        # becasuse seed, rand is pin, use unsqueeze(0) to auto boradcast
        noise = sigmas.unsqueeze(1).unsqueeze(2).unsqueeze(3) * rand_noise
        # --optimizer code end -- #
        if guess_mode:
            control_scales = self.guess_control_scales * strength
            for i in range(0, ddim_num_steps):
                index = ddim_num_steps - i - 1
                cond_output = self.union_model(
                    img,
                    hint=control,
                    t=ddim_sampling_tensor[index][:batch_size],
                    context=batch_crossattn[:batch_size],
                    control_scales=control_scales,
                    save_sample=save_sample,
                    model_name=model_name,
                )
                uncond_output = self.union_model.unet_model(
                    img,
                    ddim_sampling_tensor[index][batch_size:],
                    batch_crossattn[batch_size:]
                )
                e_t = uncond_output + uncond_scale * (cond_output - uncond_output)
                pred_x0 = (img - sqrt_one_minus_alphas[index] * e_t) / alphas[index]
                # direction pointing to x_t
                dir_xt = temp_di[index] * e_t
                img = alphas_prev[index] * pred_x0 + dir_xt + noise[index]
        else:
            img = img.repeat(2 * batch_size, 1, 1, 1)
            control = control.repeat(2 * batch_size, 1, 1, 1)
            control_scales = self.unguess_control_scales * strength
            for i in range(0, ddim_num_steps):
                index = ddim_num_steps - i - 1
                b_latent = self.union_model(
                    img,
                    hint=control,
                    t=ddim_sampling_tensor[index],
                    context=batch_crossattn,
                    control_scales=control_scales,
                    save_sample=save_sample,
                    model_name=model_name,
                )
                e_t = b_latent[batch_size:] + uncond_scale * (b_latent[:batch_size] - b_latent[batch_size:])
                pred_x0 = (img - sqrt_one_minus_alphas[index] * e_t) / alphas[index]
                # direction pointing to x_t
                dir_xt = temp_di[index] * e_t
                img = alphas_prev[index] * pred_x0 + dir_xt + noise[index]
            img = img[:batch_size]
        img = 1. / self.scale_factor * img
        img = self.vae_model(img)
        images = (
            einops.rearrange(img, 'b c h w -> b h w c') * 127.5 + 127.5
        ).clip(0, 255)
        return images


class Sampler(BaseModel):
    def __init__(self,
        device,
        verbose,
        min_batch_size,
        max_batch_size,
        embedding_dim
    ):
        super(Sampler, self).__init__(
            device=device,
            verbose=verbose,
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
            embedding_dim=embedding_dim
        )
        self.name = "Sample"

    def get_input_names(self):
        return [
            "control",
            "input_ids",
            "eta",
            "uncond_scale"
        ]

    def get_output_names(self):
       return ["images"]

    def get_dynamic_axes(self):
        return {
            "control": {0: 'H', 1: 'W'},
            "input_ids": {0: '2B'},
            "images": {0: 'B', 1: 'H', 2: 'W'}
        }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        self.check_dims(batch_size, image_height, image_width)
        min_batch, max_batch, _, _, _, _, _, _, _, _ = self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        return {
            "control": [
                (image_height, image_width, 3),
                (image_height, image_width, 3),
                (image_height, image_width, 3),
            ],
            'input_ids': [
                (min_batch * 2, self.text_maxlen),
                (batch_size * 2, self.text_maxlen),
                (max_batch * 2, self.text_maxlen)
            ],
            "eta": [(1,), (1,), (1,)],
            "uncond_scale": [(1,), (1,), (1,)],
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        return {
            "control": (image_height, image_width, 3),
            "input_ids": (batch_size * 2, self.text_maxlen),
            "eta": (1,),
            "uncond_scale": (1,),
            'images': (batch_size, image_height, image_width, 3)
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        return (
            # control
            torch.rand(
                (image_height, image_width, 3),
                dtype=torch.float32,
                device=self.device
            ),
            # input_ids
            torch.zeros(
                (batch_size * 2, self.text_maxlen),
                dtype=torch.int32,
                device=self.device
            ),
            # eta
            torch.tensor([0.0], dtype=torch.float32, device=self.device),
            # uncond_scale
            torch.tensor([9.0], dtype=torch.float32, device=self.device)
        )
        
    def optimize(self, onnx_graph):
        # change onnx -inf to -1e6 to support fp16
        for node in onnx_graph.graph.node:
            # if node.name == "/text_model/ConstantOfShape_1":
            if node.op_type == "ConstantOfShape":
                attr = node.attribute[0]
                if attr.name == "value" and attr.t.data_type == onnx.TensorProto.FLOAT:
                    np_array = np.frombuffer(attr.t.raw_data, dtype=np.float32).copy()
                    print("name", node.name, "raw array", np_array)
                    np_array[np_array == -np.inf] = -100000
                    attr.t.raw_data = np_array.tobytes() 
                    print("name", node.name,"new array", np_array)
        opt = Optimizer(onnx_graph, verbose=self.verbose)
        opt.info(self.name + ': original')
        opt.select_outputs([0]) # delete graph output#1
        opt.cleanup()
        opt.info(self.name + ': remove output[1]')
        opt.fold_constants()
        opt.info(self.name + ': fold constants')
        opt.infer_shapes()
        opt.info(self.name + ': shape inference')
        opt.select_outputs([0], names=['text_embeddings']) # rename network output
        opt.info(self.name + ': remove output[0]')
        opt_onnx_graph = opt.cleanup(return_onnx=True)
        opt.info(self.name + ': finished')
        return opt_onnx_graph
