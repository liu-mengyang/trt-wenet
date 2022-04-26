import onnx_graphsurgeon as gs
import onnx
import numpy as np

graph = gs.import_onnx(onnx.load("/workspace/decoder.onnx"))

# replace layernorm with plugin
pos_ln = ['Div_176', 'Div_238', 'Div_300', 'Div_317', 'Div_379', 'Div_441',
          'Div_458', 'Div_520', 'Div_582', 'Div_599', 'Div_661', 'Div_723',
          'Div_740', 'Div_802', 'Div_864', 'Div_881', 'Div_943', 'Div_1005',
          'Div_1022']
ln_id = 0
for div_node_name in pos_ln:
    ln_id += 1
    div_node = [node for node in graph.nodes if node.name==div_node_name][0]
    pluginVariable = gs.Variable("MyLN"+str(ln_id), np.dtype(np.float32), None)
    pluginNode = gs.Node("LayerNorm",
                        "MyLN"+str(ln_id),
                        inputs=[div_node.i(0).i(0).outputs[0]],
                        outputs=[pluginVariable],
                        attrs={"epsilon:": div_node.i(1).i().i(1).attrs['value'].values.reshape(1)})
    graph.nodes.append(pluginNode)
    div_node.o().inputs[0] = pluginVariable
    div_node.outputs.clear()

graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), "decoder_sed.onnx")
