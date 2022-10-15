import onnx
from onnx import helper
from onnx import TensorProto
import numpy as np


seq_len = 3
batch_size = 2
input_size = 2
num_direction = 2
hidden_size = 2

# X [seq_len, batch_size, input_size]
# W [num_direction, 3*hidden_size, input_size]
# R [num_direction, 3*hidden_size, hidden_size]

# Y [seq_len, num_direction, batch_size, hidden_size]
# Y_h [num_direction, batch_size, hidden_size]

ww = np.random.randn(num_direction* 3 * hidden_size * input_size)
wr = np.random.randn(num_direction* 3 * hidden_size * hidden_size)
bias_value = np.random.randn(num_direction * 6 * hidden_size)
# seq_lens_value dtype : uint8
seq_lens_value = np.ones([batch_size], dtype=np.uint32)
h0_value = np.random.randn(num_direction * batch_size * hidden_size)

# input
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, shape=(seq_len, batch_size, input_size))
W = helper.make_tensor('W', TensorProto.FLOAT, [num_direction, 3*hidden_size, input_size], ww)
R = helper.make_tensor('R', TensorProto.FLOAT, [num_direction, 3*hidden_size, hidden_size], wr)
B = helper.make_tensor('B', TensorProto.FLOAT, [num_direction, 6*hidden_size], bias_value)
sequence_lens = helper.make_tensor_value_info('sequence_lens', TensorProto.INT32, [batch_size])
initial_h = helper.make_tensor_value_info('initial_h', TensorProto.FLOAT, [num_direction, batch_size, hidden_size])

# attributes
# direction : forward / bidirectional

# output
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, shape=(seq_len, num_direction, batch_size, hidden_size))
Y_h = helper.make_tensor_value_info('Y_h', TensorProto.FLOAT, shape=(num_direction, batch_size, hidden_size))

node_def = helper.make_node(
    'GRU',
    inputs=['X', 'W', 'R', 'B', 'sequence_lens', 'initial_h'],
    outputs=['Y', 'Y_h'],
    hidden_size=hidden_size,
    direction='bidirectional',
    activations=['Sigmoid', 'Tanh', 'Sigmoid', 'Relu']
)

graph_def = helper.make_graph(
    nodes=[node_def],
    name='test_gru_mode',
    inputs=[X, sequence_lens, initial_h], # graph inputs
    outputs=[Y, Y_h], # graph outputs
    initializer=[W, R, B]
)

mode_def = helper.make_model(graph_def, producer_name='onnx-gru')
onnx.checker.check_model(mode_def)
onnx.save(mode_def, "./GRU.onnx")
print("gru model save.")