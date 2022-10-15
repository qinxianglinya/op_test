import numpy as np
import onnx
import onnxruntime as ort

x = np.ones([1,2,4,4], dtype = np.float32)
print(x)

sess = ort.InferenceSession("Conv.onnx")
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name
output = sess.run([output_name], {input_name : x})
print(output)
print("infer finish...")
