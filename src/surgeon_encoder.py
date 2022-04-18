import onnx_graphsurgeon as gs
import onnx
import numpy as np

graph = gs.import_onnx(onnx.load("/workspace/encoder.onnx"))

# goe_27 = [node for node in graph.nodes if node.name=="GreaterOrEqual_27"][0]
# goe_27.op = "LessOrEqual"

not_30 = [node for node in graph.nodes if node.name=="Not_30"][0]
# inp_node = not_30.i()
# inp_node.outputs = not_30.outputs
# not_30.outputs.clear()

# unsq_29 = [node for node in graph.nodes if node.name=="Unsqueeze_29"][0]

cast_out1 = gs.Variable("cast_out1", dtype=np.int64)
cast1 = gs.Node(name="my_cast_1", op="Cast", inputs=not_30.outputs, outputs=[cast_out1], attrs={"to":getattr(onnx.TensorProto, 'INT64')})
graph.nodes.append(cast1)

slice_79 = [node for node in graph.nodes if node.name=="Slice_79"][0]
slice_79.inputs[0] = cast_out1

slice_84 = [node for node in graph.nodes if node.name=="Slice_84"][0]

cast_out2 = gs.Variable("cast_out2", dtype=np.bool)
cast2 = gs.Node(name="my_cast_2", op="Cast", inputs=slice_84.outputs, outputs=[cast_out2], attrs={"to":getattr(onnx.TensorProto, 'BOOL')})
graph.nodes.append(cast2)

not_193 = [node for node in graph.nodes if node.name=="Not_193"][0]
not_193.inputs[0] = cast_out2
castaf_not_193 = not_193.o()
not_193.outputs = castaf_not_193.outputs
castaf_not_193.outputs.clear()

not_204 = [node for node in graph.nodes if node.name=="Not_204"][0]
not_204.inputs[0] = cast_out2
castaf_not_204 = not_204.o()
not_204.outputs = castaf_not_204.outputs
castaf_not_204.outputs.clear()

not_350 = [node for node in graph.nodes if node.name=="Not_350"][0]
not_350.inputs[0] = cast_out2
castaf_not_350 = not_350.o()
not_350.outputs = castaf_not_350.outputs
castaf_not_350.outputs.clear()

not_361 = [node for node in graph.nodes if node.name=="Not_361"][0]
not_361.inputs[0] = cast_out2
castaf_not_361 = not_361.o()
not_361.outputs = castaf_not_361.outputs
castaf_not_361.outputs.clear()

not_507 = [node for node in graph.nodes if node.name=="Not_507"][0]
not_507.inputs[0] = cast_out2
castaf_not_507 = not_507.o()
not_507.outputs = castaf_not_507.outputs
castaf_not_507.outputs.clear()

not_518 = [node for node in graph.nodes if node.name=="Not_518"][0]
not_518.inputs[0] = cast_out2
castaf_not_518 = not_518.o()
not_518.outputs = castaf_not_518.outputs
castaf_not_518.outputs.clear()

not_664 = [node for node in graph.nodes if node.name=="Not_664"][0]
not_664.inputs[0] = cast_out2
castaf_not_664 = not_664.o()
not_664.outputs = castaf_not_664.outputs
castaf_not_664.outputs.clear()

not_675 = [node for node in graph.nodes if node.name=="Not_675"][0]
not_675.inputs[0] = cast_out2
castaf_not_675 = not_675.o()
not_675.outputs = castaf_not_675.outputs
castaf_not_675.outputs.clear()

not_821 = [node for node in graph.nodes if node.name=="Not_821"][0]
not_821.inputs[0] = cast_out2
castaf_not_821 = not_821.o()
not_821.outputs = castaf_not_821.outputs
castaf_not_821.outputs.clear()

not_832 = [node for node in graph.nodes if node.name=="Not_832"][0]
not_832.inputs[0] = cast_out2
castaf_not_832 = not_832.o()
not_832.outputs = castaf_not_832.outputs
castaf_not_832.outputs.clear()

not_978 = [node for node in graph.nodes if node.name=="Not_978"][0]
not_978.inputs[0] = cast_out2
castaf_not_978 = not_978.o()
not_978.outputs = castaf_not_978.outputs
castaf_not_978.outputs.clear()

not_989 = [node for node in graph.nodes if node.name=="Not_989"][0]
not_989.inputs[0] = cast_out2
castaf_not_989 = not_989.o()
not_989.outputs = castaf_not_989.outputs
castaf_not_989.outputs.clear()

not_1135 = [node for node in graph.nodes if node.name=="Not_1135"][0]
not_1135.inputs[0] = cast_out2
castaf_not_1135 = not_1135.o()
not_1135.outputs = castaf_not_1135.outputs
castaf_not_1135.outputs.clear()

not_1146 = [node for node in graph.nodes if node.name=="Not_1146"][0]
not_1146.inputs[0] = cast_out2
castaf_not_1146 = not_1146.o()
not_1146.outputs = castaf_not_1146.outputs
castaf_not_1146.outputs.clear()

not_1292 = [node for node in graph.nodes if node.name=="Not_1292"][0]
not_1292.inputs[0] = cast_out2
castaf_not_1292 = not_1292.o()
not_1292.outputs = castaf_not_1292.outputs
castaf_not_1292.outputs.clear()

not_1303 = [node for node in graph.nodes if node.name=="Not_1303"][0]
not_1303.inputs[0] = cast_out2
castaf_not_1303 = not_1303.o()
not_1303.outputs = castaf_not_1303.outputs
castaf_not_1303.outputs.clear()

not_1449 = [node for node in graph.nodes if node.name=="Not_1449"][0]
not_1449.inputs[0] = cast_out2
castaf_not_1449 = not_1449.o()
not_1449.outputs = castaf_not_1449.outputs
castaf_not_1449.outputs.clear()

not_1460 = [node for node in graph.nodes if node.name=="Not_1460"][0]
not_1460.inputs[0] = cast_out2
castaf_not_1460 = not_1460.o()
not_1460.outputs = castaf_not_1460.outputs
castaf_not_1460.outputs.clear()

not_1606 = [node for node in graph.nodes if node.name=="Not_1606"][0]
not_1606.inputs[0] = cast_out2
castaf_not_1606 = not_1606.o()
not_1606.outputs = castaf_not_1606.outputs
castaf_not_1606.outputs.clear()

not_1617 = [node for node in graph.nodes if node.name=="Not_1617"][0]
not_1617.inputs[0] = cast_out2
castaf_not_1617 = not_1617.o()
not_1617.outputs = castaf_not_1617.outputs
castaf_not_1617.outputs.clear()

not_1763 = [node for node in graph.nodes if node.name=="Not_1763"][0]
not_1763.inputs[0] = cast_out2
castaf_not_1763 = not_1763.o()
not_1763.outputs = castaf_not_1763.outputs
castaf_not_1763.outputs.clear()

not_1774 = [node for node in graph.nodes if node.name=="Not_1774"][0]
not_1774.inputs[0] = cast_out2
castaf_not_1774 = not_1774.o()
not_1774.outputs = castaf_not_1774.outputs
castaf_not_1774.outputs.clear()

not_1920 = [node for node in graph.nodes if node.name=="Not_1920"][0]
not_1920.inputs[0] = cast_out2
castaf_not_1920 = not_1920.o()
not_1920.outputs = castaf_not_1920.outputs
castaf_not_1920.outputs.clear()

not_1931 = [node for node in graph.nodes if node.name=="Not_1931"][0]
not_1931.inputs[0] = cast_out2
castaf_not_1931 = not_1931.o()
not_1931.outputs = castaf_not_1931.outputs
castaf_not_1931.outputs.clear()

graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), "encoder_sed.onnx")
