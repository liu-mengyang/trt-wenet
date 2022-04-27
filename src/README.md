# 源代码目录

- build.sh

  调用所有需要的步骤，编译生成Plugin的.so文件；生成encoder和decoder的TensorRT文件

- Makefile

  描述Plugin的编译过程

- LayerNormPlugin.cu

  LayerNorm的Plugin算子描述文件

- LayerNormPlugin.h

  LayerNorm的Plugin算子头文件

- surgeon_encoder.py

  修改encoder计算图

- surgeon_decoder.py

  修改decoder计算图

- parse_encoder.sh

  调用trtexec生成encoder的plan文件

- parse_decoder.sh

  调用trtexec生成decoder的plan文件