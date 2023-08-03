import os
import numpy as np
from onnxruntime.quantization import (
    quantize_static, QuantFormat, CalibrationMethod,
    QuantType, QuantizationMode, CalibrationDataReader
)

now_dir = os.path.dirname(os.path.abspath(__file__))
onnx_dir = os.path.join(now_dir, "output", "onnx")
raw_calibre_dir = os.path.join(now_dir, "output", "calibre")


mode = QuantizationMode.QLinearOps
op_types_to_quantize = ['Conv']


class DataReader(CalibrationDataReader):
	
    def __init__(self, calibre_dir):
        """
        parameter data_feeds: list of input feed, each input feed is diction of {input_name: np_array}
        """
        self.data_feeds = self.calib_data(calibre_dir)
        self.iter_next = iter(self.data_feeds)
	
    @staticmethod
    def calib_data(calibre_dir):
        file_list = os.listdir(calibre_dir)
        file_list.sort(key=lambda x: int(x.split(".")[0]))
        for file in file_list:
            file_path = os.path.join(calibre_dir, file)
            raw_data = np.load(file_path)
            data = {key: raw_data[key] for key in raw_data.files}
            yield data

    def get_next(self):
        return next(self.iter_next, None)

    def rewind(self):
        self.iter_next = iter(self.data_feeds)




def quantize_static_with_int8(input_onnx_path, carlibre_dir):
	# breakpoint()
    file_name = os.path.splitext(input_onnx_path)[0]
    quant_model_path = file_name + "_int8.onnx"
    print("quant_model_path: ", quant_model_path)
    data_reader = DataReader(carlibre_dir)
    quantize_static(
        input_onnx_path,
        quant_model_path,
        data_reader,
        per_channel=True,
        reduce_range=False,
        #quant_format=QuantFormat.QOperator,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=['Conv'],
        nodes_to_exclude=['conv2'],
        calibrate_method=CalibrationMethod.MinMax,
        extra_options={
            'ActivationSymmetric': True,
            'DedicatedQDQPair': True,
            'QuantizeBias': False,
            'AddQDQPairToWeight': True,
            "OpTypesToExcludeOutputQuantization": op_types_to_quantize
        },
	)

if __name__ == "__main__":
    control_model_input = os.path.join(onnx_dir, "control_net_opt", "control_net.onnx")
    unet_model_input = os.path.join(onnx_dir, "unet_opt", "unet.onnx")
    control_calibre_dir = os.path.join(raw_calibre_dir, "control") 
    unet_calibre_dir = os.path.join(raw_calibre_dir, "unet") 
    # --- test data_reader for control model ---
    # data_reader = DataReader(control_calibre_dir)
    # data1 = data_reader.get_next()
    # print(data1)
    # print(data1.keys())
    # for control model
    quantize_static_with_int8(control_model_input, control_calibre_dir)
    # for unet
    quantize_static_with_int8(unet_model_input, unet_calibre_dir)
