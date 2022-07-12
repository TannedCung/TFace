import onnx


model = onnx.load('/workspace/checkpoints/MS1M_Quality_Regression/S4/TFace_latest.onnx')
# # # for fixed batchsize
# # model.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 32
# # model.graph.output[0].type.tensor_type.shape.dim[0].dim_value = 32
# # model.graph.output[1].type.tensor_type.shape.dim[0].dim_value = 32
# # model.graph.output[2].type.tensor_type.shape.dim[0].dim_value = 32
# # onnx.save(model, 'faceDetector_640_b32.onnx')


model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = 'batch'  # for dynamic batchsize
model.graph.output[0].type.tensor_type.shape.dim[0].dim_param = 'batch'
onnx.save(model, '/workspace/checkpoints/MS1M_Quality_Regression/S4/TFace_batch_latest.onnx')
print(f"")


####################################################
# SHow model onnx

import onnxruntime as rt
ort_session = rt.InferenceSession("/workspace/checkpoints/MS1M_Quality_Regression/S4/TFace_batch_latest.onnx", providers=["CUDAExecutionProvider"])
print("====INPUT====")
for i in ort_session.get_inputs():
    print("Name: {}, Shape: {}, Dtype: {}".format(i.name, i.shape, i.type))
print("====OUTPUT====")
for i in ort_session.get_outputs():
    print("Name: {}, Shape: {}, Dtype: {}".format(i.name, i.shape, i.type))

# import numpy as np
# input_name = ort_session.get_inputs()[0].name
# img = np.random.randn(4, 3, 180, 320).astype(np.float32)
# data = ort_session.run(None, {input_name: img})
# print("Done")