import numpy as np
import io
import torch
import sys
import os
pwd = os.path.realpath(os.path.dirname(__file__))
sys.path.insert(1, os.path.join(pwd, ".."))

from generate_pseudo_labels.extract_embedding.model import model_mobilefaceNet, model

def network(eval_model, device):
    # net = model.R50([112, 112], use_type="Qua").to(device)
    net = model_mobilefaceNet.MobileFaceNet([112, 112], 512, \
                    output_name = 'GDC', use_type = "Qua").to(device)
    net_dict = net.state_dict()     
    data_dict = {
        key.replace('module.', ''): value for key, value in torch.load(eval_model, map_location=device).items()}
    net_dict.update(data_dict)
    net.load_state_dict(net_dict)
    net.eval()
    return net

# output_onnx = "MFN_net_3epoch.onnx"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
eval_model = '/workspace/checkpoints/MS1M_Quality_Regression/S4/MFN_net_6epoch.pth'
output_onnx = eval_model[:-4] + ".onnx"
model = network(eval_model, device)
model = model.to(device)

print("==> Exporting model to ONNX format at '{}'".format(output_onnx))
input_names = ['input_1']
output_names = ['output_1']

onnx_bytes = io.BytesIO()
zero_input = torch.zeros([1, 512, 7, 7])
zero_input = zero_input.to(device)
dynamic_axes = {input_names[0]: {0:'batch'}}
for _, name in enumerate(output_names):
    dynamic_axes[name] = dynamic_axes[input_names[0]]
extra_args = {'opset_version': 10, 'verbose': False,
                'input_names': input_names, 'output_names': output_names,
                'dynamic_axes': dynamic_axes}
torch.onnx.export(model, zero_input, onnx_bytes, **extra_args)
with open(output_onnx, 'wb') as out:
    out.write(onnx_bytes.getvalue())