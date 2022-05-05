# trtexec \                                                                                                              │
#         --onnx=decoder_sed.onnx \                                                                                      │
#         --explicitBatch \                                                                                              │
#         --minShapes=encoder_out:1x16x256,encoder_out_lens:1,hyps_pad_sos_eos:1x10x64,hyps_lens_sos:1x10,ctc_score:1x10 │
# \                                                                                                                      │
#         --optShapes=encoder_out:16x64x256,encoder_out_lens:16,hyps_pad_sos_eos:16x10x64,hyps_lens_sos:16x10,ctc_score:1│
# 6x10 \                                                                                                                 │
#         --maxShapes=encoder_out:64x256x256,encoder_out_lens:64,hyps_pad_sos_eos:64x10x64,hyps_lens_sos:64x10,ctc_score:│
# 64x10 \                                                                                                                │
#         --saveEngine=decoder.plan \                                                                                    │
#         --plugins=/target/LayerNormPlugin.so \                                                                         │
#         --workspace=40960 \                                                                                            │
#         --buildOnly \                                                                                                  │
#         --noTF32 \                                                                                                     │
#         --verbose \

import os
import ctypes

import numpy as np
from cuda import cudart
import tensorrt as trt


onnxFile = "./decoder_sed.onnx"
trtFile = "./decoder.plan"
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
        
    encoderOut = network.get_input(0)
    encoderOutLens = network.get_input(1)
    hypsPadSosEos = network.get_input(2)
    hypsLensSos = network.get_input(3)
    ctcScore = network.get_input(4)
    profile.set_shape(encoderOut.name, (1, 16, 256), (16, 64, 256), (64, 256, 256))
    profile.set_shape(encoderOutLens.name, (1,), (16,), (64,))
    profile.set_shape(hypsPadSosEos.name, (1, 10, 64), (16, 10, 64), (64, 10, 64))
    profile.set_shape(hypsLensSos.name, (1, 10), (16, 10), (64, 10))
    profile.set_shape(ctcScore.name, (1, 10), (16, 10), (64, 10))
    config.add_optimization_profile(profile)
    
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        if "MyLN" in layer.name:
            layer.precision = trt.DataType.FLOAT
            layer.get_output(0).dtype = trt.DataType.FLOAT
            # layer.set_output_type(i, trt.DataType.FLOAT)
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
        
