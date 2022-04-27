# trt-wenet
A TensorRT8 implementation of WeNet

## 任务

* [x] 不适类型转换
* [ ] FP16加速

## 补充资料

* `onnx_graphsurgeon.Node`支持哪些`op`？👉[看这](https://github.com/NVIDIA/TensorRT/blob/052281f0ab795b6c1a19047dc8a449cd397995a9/tools/onnx-graphsurgeon/onnx_graphsurgeon/ir/graph.py#L521)👉[还有比较特殊的这](https://github.com/NVIDIA/TensorRT/blob/f4a8635399adbfc9264707e9af4535d55829d956/tools/onnx-graphsurgeon/onnx_graphsurgeon/ir/graph.py#L632)👉[还有也有点特殊的这](https://github.com/NVIDIA/TensorRT/blob/f4a8635399adbfc9264707e9af4535d55829d956/tools/onnx-graphsurgeon/onnx_graphsurgeon/ir/graph.py#L570)

## 如何使用？

需要首先通过DOCKERFILE安装好容器镜像并进入容器，再进行后面的操作。

```bash
cd /target
```

- 生成encoder和decoder的plan文件

  ```bash
  bash build.sh
  ```

- 单独生成encoder的plan文件

  ```bash
  make
  bash parse_encoder.sh
  ```

- 单独生成decoder的plan文件

  ```bash
  make
  bash parse_decoder.sh
  ```

- 测试

  ```bash
  python /workspace/testEncoderandDecoder.py
  ```

  查看性能表现。
