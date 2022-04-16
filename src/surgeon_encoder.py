import onnx_graphsurgeon as gs
import onnx
import numpy as np

graph = gs.import_onnx(onnx.load("/workspace/encoder.onnx"))

not_30 = [node for node in graph.nodes if node.name=="Not_30"][0]
cast_out1 = gs.Variable("cast_out1", dtype=np.int64)
cast1 = gs.Node(name="my_cast_1", op="Cast", inputs=not_30.outputs, outputs=[cast_out1], attrs={"to":getattr(onnx.TensorProto, 'INT32')})
graph.nodes.append(cast1)

slice_79 = [node for node in graph.nodes if node.name=="Slice_79"][0]
slice_79.inputs[0] = cast_out1

cons_0 = [node for node in graph.nodes if node.name=="Constant_163"][0] 
for node in graph.nodes:
    if "Not" in node.name and node.name != "Not_30":
        node.op = "Equal"
        node.inputs = [node.inputs[0], cons_0.outputs[0]]
        fake_node = node.o()
        node.outputs = fake_node.outputs
        fake_node.outputs.clear()

#slice_84 = [node for node in graph.nodes if node.name=="Slice_84"][0]
#cast_out2 = gs.Variable("cast_out2", dtype=np.bool)
#cast2 = gs.Node(name="my_cast_2", op="Cast", inputs=slice_84.outputs, outputs=[cast_out2], attrs={"to":getattr(onnx.TensorProto, 'BOOL')})
#graph.nodes.append(cast2)
#
#not_193 = [node for node in graph.nodes if node.name=="Not_193"][0]
#not_193.inputs[0].dtype = np.bool_
#
#cast_out3 = gs.Variable("cast_out3", dtype=np.bool)
#cast3 = gs.Node(name="my_cast_3", op="Cast", inputs=not_193.outputs, outputs=[cast_out3], attrs={"to":getattr(onnx.TensorProto, 'BOOL')})
#graph.nodes.append(cast3)

#not_204 = [node for node in graph.nodes if node.name=="Not_204"][0]
#not_204.inputs[0] = cast_out3

#cast_out4 = gs.Variable("cast_out4", dtype=np.bool)
#cast4 = gs.Node(name="my_cast_4", op="Cast", inputs=slice_84.outputs, outputs=[cast_out4], attrs={"to":getattr(onnx.TensorProto, 'BOOL')})
#graph.nodes.append(cast4)
#
#not_350 = [node for node in graph.nodes if node.name=="Not_350"][0]
#not_350.inputs[0] = cast_out4
#
# cast_out5 = gs.Variable("cast_out5", dtype=np.bool)
# cast5 = gs.Node(name="my_cast_5", op="Cast", inputs=slice_84.outputs, outputs=[cast_out5], attrs={"to":getattr(onnx.TensorProto, 'BOOL')})
# graph.nodes.append(cast5)

# not_361 = [node for node in graph.nodes if node.name=="Not_361"][0]
# not_361.inputs[0] = cast_out5

# cast_out6 = gs.Variable("cast_out6", dtype=np.bool)
# cast6 = gs.Node(name="my_cast_6", op="Cast", inputs=slice_84.outputs, outputs=[cast_out6], attrs={"to":getattr(onnx.TensorProto, 'BOOL')})
# graph.nodes.append(cast6)

# not_507 = [node for node in graph.nodes if node.name=="Not_507"][0]
# not_507.inputs[0] = cast_out6

# cast_out7 = gs.Variable("cast_out7", dtype=np.bool)
# cast7 = gs.Node(name="my_cast_7", op="Cast", inputs=slice_84.outputs, outputs=[cast_out7], attrs={"to":getattr(onnx.TensorProto, 'BOOL')})
# graph.nodes.append(cast7)

# not_518 = [node for node in graph.nodes if node.name=="Not_518"][0]
# not_518.inputs[0] = cast_out7

# cast_out8 = gs.Variable("cast_out8", dtype=np.bool)
# cast8 = gs.Node(name="my_cast_8", op="Cast", inputs=slice_84.outputs, outputs=[cast_out8], attrs={"to":getattr(onnx.TensorProto, 'BOOL')})
# graph.nodes.append(cast8)

# not_664 = [node for node in graph.nodes if node.name=="Not_664"][0]
# not_664.inputs[0] = cast_out8

# cast_out9 = gs.Variable("cast_out9", dtype=np.bool)
# cast9 = gs.Node(name="my_cast_9", op="Cast", inputs=slice_84.outputs, outputs=[cast_out9], attrs={"to":getattr(onnx.TensorProto, 'BOOL')})
# graph.nodes.append(cast9)

# not_675 = [node for node in graph.nodes if node.name=="Not_675"][0]
# not_675.inputs[0] = cast_out9

# cast_out10 = gs.Variable("cast_out10", dtype=np.bool)
# cast10 = gs.Node(name="my_cast_10", op="Cast", inputs=slice_84.outputs, outputs=[cast_out10], attrs={"to":getattr(onnx.TensorProto, 'BOOL')})
# graph.nodes.append(cast10)

# not_821 = [node for node in graph.nodes if node.name=="Not_821"][0]
# not_821.inputs[0] = cast_out10

# cast_out11 = gs.Variable("cast_out11", dtype=np.bool)
# cast11 = gs.Node(name="my_cast_11", op="Cast", inputs=slice_84.outputs, outputs=[cast_out11], attrs={"to":getattr(onnx.TensorProto, 'BOOL')})
# graph.nodes.append(cast11)

# not_832 = [node for node in graph.nodes if node.name=="Not_832"][0]
# not_832.inputs[0] = cast_out11

# cast_out12 = gs.Variable("cast_out12", dtype=np.bool)
# cast12 = gs.Node(name="my_cast_12", op="Cast", inputs=slice_84.outputs, outputs=[cast_out12], attrs={"to":getattr(onnx.TensorProto, 'BOOL')})
# graph.nodes.append(cast12)

# not_978 = [node for node in graph.nodes if node.name=="Not_978"][0]
# not_978.inputs[0] = cast_out12

# cast_out13 = gs.Variable("cast_out13", dtype=np.bool)
# cast13 = gs.Node(name="my_cast_13", op="Cast", inputs=slice_84.outputs, outputs=[cast_out13], attrs={"to":getattr(onnx.TensorProto, 'BOOL')})
# graph.nodes.append(cast13)

# not_989 = [node for node in graph.nodes if node.name=="Not_989"][0]
# not_989.inputs[0] = cast_out13

# cast_out14 = gs.Variable("cast_out14", dtype=np.bool)
# cast14 = gs.Node(name="my_cast_14", op="Cast", inputs=slice_84.outputs, outputs=[cast_out14], attrs={"to":getattr(onnx.TensorProto, 'BOOL')})
# graph.nodes.append(cast14)

# not_1135 = [node for node in graph.nodes if node.name=="Not_1135"][0]
# not_1135.inputs[0] = cast_out14

# cast_out15 = gs.Variable("cast_out15", dtype=np.bool)
# cast15 = gs.Node(name="my_cast_15", op="Cast", inputs=slice_84.outputs, outputs=[cast_out15], attrs={"to":getattr(onnx.TensorProto, 'BOOL')})
# graph.nodes.append(cast15)

# not_1146 = [node for node in graph.nodes if node.name=="Not_1146"][0]
# not_1146.inputs[0] = cast_out15

# cast_out16 = gs.Variable("cast_out16", dtype=np.bool)
# cast16 = gs.Node(name="my_cast_16", op="Cast", inputs=slice_84.outputs, outputs=[cast_out16], attrs={"to":getattr(onnx.TensorProto, 'BOOL')})
# graph.nodes.append(cast16)

# not_1292 = [node for node in graph.nodes if node.name=="Not_1292"][0]
# not_1292.inputs[0] = cast_out16

# cast_out17 = gs.Variable("cast_out17", dtype=np.bool)
# cast17 = gs.Node(name="my_cast_17", op="Cast", inputs=slice_84.outputs, outputs=[cast_out17], attrs={"to":getattr(onnx.TensorProto, 'BOOL')})
# graph.nodes.append(cast17)

# not_1303 = [node for node in graph.nodes if node.name=="Not_1303"][0]
# not_1303.inputs[0] = cast_out17

# cast_out18 = gs.Variable("cast_out18", dtype=np.bool)
# cast18 = gs.Node(name="my_cast_18", op="Cast", inputs=slice_84.outputs, outputs=[cast_out18], attrs={"to":getattr(onnx.TensorProto, 'BOOL')})
# graph.nodes.append(cast18)

# not_1449 = [node for node in graph.nodes if node.name=="Not_1449"][0]
# not_1449.inputs[0] = cast_out18

# cast_out19 = gs.Variable("cast_out19", dtype=np.bool)
# cast19 = gs.Node(name="my_cast_19", op="Cast", inputs=slice_84.outputs, outputs=[cast_out19], attrs={"to":getattr(onnx.TensorProto, 'BOOL')})
# graph.nodes.append(cast19)

# not_1460 = [node for node in graph.nodes if node.name=="Not_1460"][0]
# not_1460.inputs[0] = cast_out19

# cast_out20 = gs.Variable("cast_out20", dtype=np.bool)
# cast20 = gs.Node(name="my_cast_20", op="Cast", inputs=slice_84.outputs, outputs=[cast_out20], attrs={"to":getattr(onnx.TensorProto, 'BOOL')})
# graph.nodes.append(cast20)

# not_1606 = [node for node in graph.nodes if node.name=="Not_1606"][0]
# not_1606.inputs[0] = cast_out20

# cast_out21 = gs.Variable("cast_out21", dtype=np.bool)
# cast21 = gs.Node(name="my_cast_21", op="Cast", inputs=slice_84.outputs, outputs=[cast_out21], attrs={"to":getattr(onnx.TensorProto, 'BOOL')})
# graph.nodes.append(cast21)

# not_1617 = [node for node in graph.nodes if node.name=="Not_1617"][0]
# not_1617.inputs[0] = cast_out21

# cast_out22 = gs.Variable("cast_out22", dtype=np.bool)
# cast22 = gs.Node(name="my_cast_22", op="Cast", inputs=slice_84.outputs, outputs=[cast_out22], attrs={"to":getattr(onnx.TensorProto, 'BOOL')})
# graph.nodes.append(cast22)

# not_1763 = [node for node in graph.nodes if node.name=="Not_1763"][0]
# not_1763.inputs[0] = cast_out22

# cast_out23 = gs.Variable("cast_out23", dtype=np.bool)
# cast23 = gs.Node(name="my_cast_23", op="Cast", inputs=slice_84.outputs, outputs=[cast_out23], attrs={"to":getattr(onnx.TensorProto, 'BOOL')})
# graph.nodes.append(cast23)

# not_1774 = [node for node in graph.nodes if node.name=="Not_1774"][0]
# not_1774.inputs[0] = cast_out23

# cast_out24 = gs.Variable("cast_out24", dtype=np.bool)
# cast24 = gs.Node(name="my_cast_24", op="Cast", inputs=slice_84.outputs, outputs=[cast_out24], attrs={"to":getattr(onnx.TensorProto, 'BOOL')})
# graph.nodes.append(cast24)

# not_1920 = [node for node in graph.nodes if node.name=="Not_1920"][0]
# not_1920.inputs[0] = cast_out24

# cast_out25 = gs.Variable("cast_out25", dtype=np.bool)
# cast25 = gs.Node(name="my_cast_25", op="Cast", inputs=slice_84.outputs, outputs=[cast_out25], attrs={"to":getattr(onnx.TensorProto, 'BOOL')})
# graph.nodes.append(cast25)

# not_1931 = [node for node in graph.nodes if node.name=="Not_1931"][0]
# not_1931.inputs[0] = cast_out25



graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), "encoder_sed.onnx")
