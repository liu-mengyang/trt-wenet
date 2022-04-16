import onnx_graphsurgeon as gs
import onnx
import numpy as np


graph = gs.import_onnx(onnx.load("encoder_sed.onnx"))

graph.inputs[0].shape = ['B', 1, 28, 28]
graph.outputs[0].shape = ['B', 10]

print("# Traverse the node: -----------------------------------------------------")  # 遍历节点，打印：节点信息，输入张量，输出张量，父节点名，子节点名
for index, node in enumerate(graph.nodes):
    print("Node%4d: op=%s, name=%s, attrs=%s"%(index, node.op,node.name, "".join(["{"] + [str(key)+":"+str(value)+", " for key, value in node.attrs.items()] + ["}"])))
    for jndex, inputTensor in enumerate(node.inputs):
        print("\tInTensor  %d: %s"%(jndex, inputTensor))
    for jndex, outputTensor in enumerate(node.outputs):
        print("\tOutTensor %d: %s"%(jndex, outputTensor))

    fatherNodeList = []
    for newNode in graph.nodes:
        for newOutputTensor in newNode.outputs:
            if newOutputTensor in node.inputs:
                fatherNodeList.append(newNode)
    for jndex, newNode in enumerate(fatherNodeList):
        print("\tFatherNode%d: %s"%(jndex,newNode.name))

    sonNodeList = []
    for newNode in graph.nodes:
        for newInputTensor in newNode.inputs:
            if newInputTensor in node.outputs:
                sonNodeList.append(newNode)
    for jndex, newNode in enumerate(sonNodeList):
        print("\tSonNode   %d: %s"%(jndex,newNode.name))

print("# Traverse the tensor: ---------------------------------------------------") # 遍历张量，打印：张量信息，以本张量作为输入张量的节点名，以本张量作为输出张量的节点名，父张量信息，子张量信息
for index,(name,tensor) in enumerate(graph.tensors().items()):
    print("Tensor%4d: name=%s, desc=%s"%(index, name, tensor))
    for jndex, inputNode in enumerate(tensor.inputs):
        print("\tInNode      %d: %s"%(jndex, inputNode.name))
    for jndex, outputNode in enumerate(tensor.outputs):
        print("\tOutNode     %d: %s"%(jndex, outputNode.name))

    fatherTensorList = []
    for newTensor in list(graph.tensors().values()):
        for newOutputNode in newTensor.outputs:
            if newOutputNode in tensor.inputs:
                fatherTensorList.append(newTensor)
    for jndex, newTensor in enumerate(fatherTensorList):
        print("\tFatherTensor%d: %s"%(jndex,newTensor))

    sonTensorList = []
    for newTensor in list(graph.tensors().values()):
        for newInputNode in newTensor.inputs:
            if newInputNode in tensor.outputs:
                sonTensorList.append(newTensor)
    for jndex, newTensor in enumerate(sonTensorList):
        print("\tSonTensor   %d: %s"%(jndex,newTensor))