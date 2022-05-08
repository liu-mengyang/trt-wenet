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

slice_84 = [node for node in graph.nodes if node.name=="Slice_84"][0] # 看来这个节点就是造成问题的核心

# ↓↓↓↓↓待修改的节点名叫Not_193，需要在其输入部分加一层类型转换，把INT转BOOL↓↓↓↓↓
cast_out2 = gs.Variable("cast_out2", dtype=np.bool) # ←定义一个作为输入的变量cast_out2
# ↓定义一个节点（计算过程），以前面定义的slice_84的输出作为输入，将其进行类型转换，输出到变量cast_out2中
cast2 = gs.Node(name="my_cast_2", op="Cast", inputs=slice_84.outputs, outputs=[cast_out2], attrs={"to":getattr(onnx.TensorProto, 'BOOL')})
graph.nodes.append(cast2) # ←节点加入图中

not_193 = [node for node in graph.nodes if node.name=="Not_193"][0] # ←找到待修改的节点Not_193
not_193.inputs[0] = cast_out2 # 删除其原有的输入，把进行类型转换后输出的变量cast_out2作为它的输入，从而完成类型转换层的添加
castaf_not_193 = not_193.o() # 定位其原本的下一层，也是一个Cast算子，但是由于在输入进行了转换，不必再次转换
not_193.outputs = castaf_not_193.outputs # 将其下一层的输出作为该层的输出
castaf_not_193.outputs.clear() # 删除原本存在的多余的cast层
# ↑↑↑↑待修改的节点名叫Not_193，需要在其输入部分加一层类型转换，把INT转BOOL↑↑↑↑

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

add_1968 = [node for node in graph.nodes if node.name=="Add_1968"][0]
print(add_1968.o(0))
print(add_1968.o(1))

pos_ln = ['Div_93', 'Div_113', 'Div_189', 'Div_219', 'Div_239', 'Div_250',
          'Div_270', 'Div_346', 'Div_376', 'Div_396', 'Div_407', 'Div_427',
          'Div_503', 'Div_533', 'Div_553', 'Div_564', 'Div_584', 'Div_660',
          'Div_690', 'Div_710', 'Div_721', 'Div_741', 'Div_817', 'Div_847',
          'Div_867', 'Div_878', 'Div_898', 'Div_974', 'Div_1004', 'Div_1024',
          'Div_1035', 'Div_1055', 'Div_1131', 'Div_1161', 'Div_1181',
          'Div_1192', 'Div_1212', 'Div_1288', 'Div_1318', 'Div_1338',
          'Div_1349', 'Div_1369', 'Div_1445', 'Div_1475', 'Div_1495',
          'Div_1506', 'Div_1526', 'Div_1602', 'Div_1632', 'Div_1652',
          'Div_1663', 'Div_1683', 'Div_1759', 'Div_1789', 'Div_1809',
          'Div_1820', 'Div_1840', 'Div_1916', 'Div_1946', 'Div_1966',
          'Div_1977']

# replace layernorm with plugin
pos_ln_1 = ['Div_93', 'Div_189', 'Div_219', 'Div_250',
          'Div_346', 'Div_376', 'Div_407',
          'Div_503', 'Div_533', 'Div_564', 'Div_660',
          'Div_690', 'Div_721', 'Div_817', 'Div_847',
          'Div_878', 'Div_974', 'Div_1004',
          'Div_1035', 'Div_1131', 'Div_1161',
          'Div_1192', 'Div_1288', 'Div_1318',
          'Div_1349', 'Div_1445', 'Div_1475',
          'Div_1506', 'Div_1602', 'Div_1632',
          'Div_1663', 'Div_1759', 'Div_1789',
          'Div_1820', 'Div_1916', 'Div_1946']
pos_ln_2 = ['Div_1966', 'Div_1977']
pos_ln_3 = ['Div_239', 'Div_396', 'Div_553', 'Div_710', 'Div_867', 'Div_1024',
            'Div_1181', 'Div_1338', 'Div_1495', 'Div_1652', 'Div_1809']
pos_ln_4 = ['Div_113', 'Div_270', 'Div_427', 'Div_584', 'Div_741', 'Div_898',
            'Div_1055', 'Div_1212', 'Div_1369', 'Div_1526', 'Div_1683',
            'Div_1840']
pos_ln_dic = {1:pos_ln_1, 2:pos_ln_2, 3:pos_ln_3, 4:pos_ln_4}
ln_id = 0
for div_node_name in pos_ln:
    print(div_node_name)
    ln_id += 1
    div_node = [node for node in graph.nodes if node.name==div_node_name][0]
    pluginVariable = gs.Variable("MyLN"+str(ln_id), np.dtype(np.float32), None)
    pluginNode = gs.Node("LayerNorm",
                        "MyLN"+str(ln_id),
                        inputs=[div_node.i(0).i(0).outputs[0]],
                        outputs=[pluginVariable],
                        attrs={"epsilon:": div_node.i(1).i().i(1).attrs['value'].values.reshape(1),
                            "gamma:": div_node.o().inputs[1].values.reshape(256),
                            "beta:": div_node.o().o().inputs[1].values.reshape(256)})
    graph.nodes.append(pluginNode)
    # print(add_1968.o(0))
    # print(add_1968.o(1))
    node_lst = []
    if div_node_name in pos_ln_1:
        key = 1
    elif div_node_name in pos_ln_2:
        key = 2
    elif div_node_name in pos_ln_3:
        key = 3
    elif div_node_name in pos_ln_4:
        key = 4
    for i in range(key):
        if div_node_name == "Div_1977":
            pluginNode.outputs[0] = div_node.o().o().outputs[0]
            break
        node_lst.append(div_node.o().o().o(i))
    for node in node_lst:
        node.inputs[0] = pluginVariable

    div_node.o().o().outputs.clear()
        

graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), "encoder_sed.onnx")
