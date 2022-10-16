import onnx
from onnx import helper
from onnx import TensorProto
import numpy as np
from test_tools import *
import onnxruntime as ort

seq_len = 128
batch_size = 4
input_size = 512
num_direction = 2
hidden_size = 256
direction = ''


# seq_len = 1
# batch_size = 3
# input_size = 2
# num_direction = 1
# hidden_size = 3
# direction = ''

if 1 == num_direction:
    direction = 'forward'
elif 2 == num_direction:
    direction = 'bidirectional'

# X [seq_len, batch_size, input_size]
# W [num_direction, 3*hidden_size, input_size]
# R [num_direction, 3*hidden_size, hidden_size]

# Y [seq_len, num_direction, batch_size, hidden_size]
# Y_h [num_direction, batch_size, hidden_size]

# prepare test data
ww_path = './test_data/ww.txt'
wr_path = './test_data/wr.txt'
bias_path = './test_data/bias.txt'

x_path = './test_data/x.txt'
seq_lens_path = './test_data/seq_lens.txt'
h_0_path = './test_data/h0.txt'

tag = True  # generate_onnx_model
# tag = False # runtime

def generate_onnx_mode():
    ww = np.random.randn(num_direction* 3 * hidden_size * input_size)
    ww = save_and_read_data(ww, ww_path, 'float32')

    wr = np.random.randn(num_direction* 3 * hidden_size * hidden_size)
    wr = save_and_read_data(wr, wr_path, 'float32')

    bias_value = np.random.randn(num_direction * 6 * hidden_size)
    bias_value = save_and_read_data(bias_value, bias_path, 'float32')
    # ww = read_data_from_txt(ww_path, 'float32')
    # wr = read_data_from_txt(wr_path, 'float32')  
    # bias_value = read_data_from_txt(bias_path, 'float32')

    x_value = np.random.randn(seq_len * batch_size * input_size)
    save_data_to_local(x_value, x_path, 'float32')
    seq_lens_value = np.random.randint(low=1, high=(seq_len+1), size=[batch_size])
    save_data_to_local(seq_lens_value, seq_lens_path, 'int32')
    h0_value = np.random.randn(num_direction * batch_size * hidden_size)
    save_data_to_local(h0_value, h_0_path, 'float32')

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
        direction=direction,
        linear_before_reset=1,
        activations=['Sigmoid', 'Relu', 'Sigmoid', 'Relu']
    )
    
    graph_def = helper.make_graph(
        nodes=[node_def],
        name='test_gru_mode',
        inputs=[X, sequence_lens, initial_h], # graph inputs
        outputs=[Y, Y_h], # graph outputs
        initializer=[W, R, B]
    )
    
    # node_def = helper.make_node(
    #     'GRU',
    #     inputs=['X', 'W', 'R', 'B', 'sequence_lens', 'initial_h'],
    #     outputs=['Y', 'Y_h'],
    #     hidden_size=hidden_size,
    #     direction=direction,
    #     linear_before_reset=1,
    #     activations=['Sigmoid', 'Sigmoid']
    # )

    # graph_def = helper.make_graph(
    #     nodes=[node_def],
    #     name='test_gru_mode',
    #     inputs=[X], # graph inputs
    #     outputs=[Y, Y_h], # graph outputs
    #     initializer=[W, R, B]
    # )

    mode_def = helper.make_model(graph_def, producer_name='onnx-gru')
    onnx.checker.check_model(mode_def)
    onnx.save(mode_def, "./GRU.onnx")
    print("gru model save.")

def runtime():
    Y_path = './test_data/y.txt'
    Y_h_path = './test_data/yh.txt'

    x = read_data_from_txt(x_path, 'float32').reshape([seq_len, batch_size, input_size])
    h0 = read_data_from_txt(h_0_path, 'float32').reshape([num_direction, batch_size, hidden_size])
    seq_lens = read_data_from_txt(seq_lens_path, 'int32').reshape([batch_size])
    # x = np.array([-0.6983, 2.0323, 0.9536, -1.9587, 0.4631, 0.3348], dtype=np.float32).reshape([seq_len, batch_size, input_size])
    
    sess = ort.InferenceSession("GRU.onnx")
    input_name = sess.get_inputs()[0].name
    seq_lens_name = sess.get_inputs()[1].name
    h0_name = sess.get_inputs()[2].name
    Y_name = sess.get_outputs()[0].name
    Y_h_name = sess.get_outputs()[1].name
    Y, Y_h = sess.run([Y_name, Y_h_name], 
                    {input_name : x, seq_lens_name : seq_lens, h0_name : h0})
    # Y, Y_h = sess.run([Y_name, Y_h_name], {input_name : x})
    print("onnx Y'shape is : ", Y.shape)
    print("onnx Y_h'shape is : ", Y_h.shape)

    # <onnx output shape> ---> <mm output shape>
    Y_transpose = Y.transpose(0, 2, 1, 3)
    Y_reshape = Y_transpose.reshape([seq_len, batch_size, num_direction * hidden_size])
    print("mm Y'shape is : ", Y_reshape.shape)
    print("mm Y_h'shape is : ", Y_h.shape)
    print('-----------> Y\'value is : \n', Y)
    print('-----------> Y_h\'value is : \n', Y_h)

    save_data_to_local(Y_reshape.reshape(-1), Y_path, 'float32')
    save_data_to_local(Y_h.reshape(-1), Y_h_path, 'float32')
    print("infer finish...")
    
if __name__ == '__main__':
    if tag:
        generate_onnx_mode()
        runtime()  
    elif tag == False:
        runtime()      
    # a = np.random.randn(30).reshape([3,2,5])
    # print(a)
    # print(np.split(a, 3, 0))
           

