import numpy as np
import onnx
import onnxruntime as ort

seq_len = 3
batch_size = 2
input_size = 2
num_direction = 2
hidden_size = 2

x = np.ones([seq_len, batch_size, input_size], dtype=np.float32)
seq_lens = np.ones([batch_size], dtype=np.int32)
h0 = np.ones([num_direction, batch_size, hidden_size], dtype=np.float32)
print(x)

sess = ort.InferenceSession("GRU.onnx")
input_name = sess.get_inputs()[0].name
seq_lens_name = sess.get_inputs()[1].name
h0_name = sess.get_inputs()[2].name
output_name = sess.get_outputs()[0].name
output = sess.run([output_name], {input_name : x, seq_lens_name : seq_lens, h0_name : h0})
print(output)
print("infer finish...")
