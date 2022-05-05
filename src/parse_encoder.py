
# trtexec \
# 	--onnx=encoder_sed.onnx \
# 	--explicitBatch \
# 	--minShapes=speech:1x16x80,speech_lengths:1 \
# 	--optShapes=speech:16x64x80,speech_lengths:16 \
# 	--maxShapes=speech:64x256x80,speech_lengths:64 \
# 	--saveEngine=encoder.plan \
# 	--plugins=/target/LayerNormPlugin.so \
# 	--workspace=40960 \
# 	--buildOnly \
# 	--fp16 \
# 	--verbose \

import os
import ctypes

import numpy as np
from cuda import cudart
import tensorrt as trt


onnxFile = "./encoder_sed.onnx"
trtFile = "./encoder.plan"
soFile = "./LayerNormPlugin.so"


np.set_printoptions(precision=4, linewidth=200, suppress=True)
cudart.cudaDeviceSynchronize()

logger = trt.Logger(trt.Logger.ERROR)
trt.init_libnvinfer_plugins(logger, '')
ctypes.cdll.LoadLibrary(soFile)
if os.path.isfile(trtFile):
    with open(trtFile, 'rb') as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    if engine == None:
        print("Failed loading engine!")
        exit()
    print("Succeeded loading engine!")
else:
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.STRICT_TYPES)
    config.max_workspace_size = 3 << 30
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    parser = trt.OnnxParser(network, logger)
    if not os.path.exists(onnxFile):
        print("Failed finding onnx file!")
        exit()
    print("Succeeded finding onnx file!")
    with open(onnxFile, 'rb') as model:
        if not parser.parse(model.read()):
            print("Failed parsing .onnx file!")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            exit()
        print("Succeeded parsing .onnx file!")
        
    inputSpeech = network.get_input(0)
    inputSpeechLengths = network.get_input(1)
    profile.set_shape(inputSpeech.name, (1, 16, 80), (16, 64, 80), (64, 256, 80))
    profile.set_shape(inputSpeechLengths.name, (1,), (16,), (64,))
    config.add_optimization_profile(profile)
    
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        if "MyLN" in layer.name:
            layer.precision = trt.DataType.FLOAT
            layer.get_output(0).dtype = trt.DataType.FLOAT
        # else:
        #     layer.precision = trt.DataType.HALF
        #     no = layer.num_outputs
        #     for j in range(no):
        #         layer.get_output(j).dtype = trt.DataType.HALF

    
    engineString = builder.build_serialized_network(network, config)
    if engineString == None:
        print("Failed building engine!")
        exit()
    print("Succeeded building engine!")
    with open(trtFile, 'wb') as f:
        f.write(engineString)
        
