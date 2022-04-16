import onnx_graphsurgeon as gs
import onnx
import numpy as np

graph = gs.import_onnx(onnx.load("/workspace/decoder.onnx"))

graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), "decoder_sed.onnx")
