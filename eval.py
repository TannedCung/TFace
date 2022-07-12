import os
import os.path as osp
import torch
import torch.nn as nn
import torch.optim as opti
from tqdm import tqdm
import torchvision.transforms as T
from generate_pseudo_labels.extract_embedding.model import model_mobilefaceNet, model
import numpy as np
from scipy import stats
import pdb
from PIL import Image
from pathlib import Path
import timeit
import torch.onnx

import sys
sys.path.insert(1, "/workspace/data")
from onnx_infer import OnnxInfer

IN_PATH = "/workspace/real_test/1"
OUT_PATH = "/workspace/real_out"

def read_img(imgPath):     # read image & data pre-process
    data = torch.randn(1, 3, 112, 112)
    transform = T.Compose([
        T.Resize((112, 112)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    img = Image.open(imgPath).convert("RGB")
    data[0, :, :, :] = transform(img)
    return data, img


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

if __name__ == "__main__":
    export = True
    os.makedirs(OUT_PATH, exist_ok=True)
    imgdir = Path(IN_PATH)                         # [1,2,3.jpg]
    device = 'cpu'                                        # 'cpu' or 'cuda:x'
    # eval_model = '/research/classification/TFace/generate_pseudo_labels/extract_embedding/model/SDD_FIQA_checkpoints_r50.pth'   # checkpoint
    eval_model = '/workspace/checkpoints/MS1M_Quality_Regression/S4/MFN_net_6epoch.pth'
    net = network(eval_model, device)
    model_type = "webface"
    onnx_net = OnnxInfer(weight_paths="/workspace/weights/modified.onnx", type=model_type)

    for imgpath in imgdir.glob("*.jpg"):
        imgpath = str(imgpath)
        input_data, image = read_img(imgpath)

        input_data = onnx_net(input_data.numpy())[1]
        input_data = torch.from_numpy(input_data)

        if export:
            torch.onnx.export(net,               # model being run
                input_data,                         # model input (or a tuple for multiple inputs)
                "/workspace/checkpoints/MS1M_Quality_Regression/S4/TFace_latest.onnx",   # where to save the model (can be a file or file-like object)
                export_params=True,        # store the trained parameter weights inside the model file
                opset_version=10,          # the ONNX version to export the model to
                do_constant_folding=True,  # whether to execute constant folding for optimization
                input_names = ['input_1'],   # the model's input names
                output_names = ['output_1']) # the model's output names
                # dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                #             'output' : {0 : 'batch_size'}})
            print(f"[INFO] Exported Onnx")
            export = False

        t0 = timeit.default_timer()
        input_data = input_data.to(device)
        pred_score = net(input_data).data.cpu().numpy().squeeze()
        t1 = timeit.default_timer()
        # print(t1-t0, imgpath)
        print(f"{imgpath} =====> Quality score = {pred_score}")
        save_name = f"{pred_score*100:3.5f}.jpg"
        _i = 0
        while save_name in os.listdir(OUT_PATH):
            save_name = f"{pred_score*100:3.5f}_{_i}.jpg"
            _i += 1
        image.save(os.path.join(OUT_PATH, f"{pred_score*100:3.5f}.jpg"))


