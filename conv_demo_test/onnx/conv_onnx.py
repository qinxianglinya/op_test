import onnx
from onnx import helper
from onnx import TensorProto
import numpy as np

weight = np.random.randn(36)
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 2, 4, 4])
W = helper.make_tensor('W', TensorProto.FLOAT, [2, 2, 3, 3], weight)
B = helper.make_tensor('B', TensorProto.FLOAT, [2], [1.0, 2.0])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 2, 2, 2])
node_def = helper.make_node(
'Conv', # node name
['X', 'W', 'B'],
['Y'], # outputs
# attributes
strides=[2,2],
)
graph_def = helper.make_graph(
[node_def],
'test_conv_mode',
[X], # graph inputs
[Y], # graph outputs
initializer=[W, B],
)
mode_def = helper.make_model(graph_def, producer_name='onnx-example')
onnx.checker.check_model(mode_def)
onnx.save(mode_def, "./Conv.onnx")
print("conv model save.")