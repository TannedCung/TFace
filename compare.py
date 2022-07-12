import os
import os.path as osp
from re import L
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
# import cv2[INFO]:[demo_imgs/f33f6cc5-ed10-461f-b124-c9288efe45cd.jpg] - sum diff: -0.0003461742599029094 ====>  old score: [[0.2721825]] -- new score: [[0.10057434]]

import sys
sys.path.insert(1, "/workspace/data")
from onnx_infer import OnnxInfer

NET1 = "/workspace/webface_r50_tface.onnx"
NET2 = "/workspace/Web_mod.onnx"

def read_img(imgPath):     # read image & data pre-process
    data = torch.randn(1, 3, 112, 112)
    transform = T.Compose([
        T.Resize((112, 112)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    img = Image.open(imgPath).convert("RGB")
    data[0, :, :, :] = transform(img)
    return data

def compare():
    imgdir = Path('./real_test/1/')  
    model_type = "webface"
    old_net = OnnxInfer(weight_paths=NET1, type=model_type)
    new_net = OnnxInfer(weight_paths=NET2, type=model_type)

    for imgpath in imgdir.glob("*.jpg"):
        imgpath = str(imgpath)
        input_data = read_img(imgpath)

        out1 = old_net(input_data.numpy())
        out2 = new_net(input_data.numpy())

        # compare emb
        emb_diff = np.abs(np.array(out1[0])) - np.abs(np.array(out2[0]))
        # print(f"[INFO]: sum diff: {emb_diff.sum()}")        
        # print(f"[INFO]: max diff: {emb_diff.max()}")        
        # print(f"[INFO]: min diff: {emb_diff.min()}")  

        # compare score
        print(f"[INFO]:[{imgpath}] - sum diff: {emb_diff.sum()} ====>  old score: {out1[1]} -- new score: {out2[1]}")


if __name__=="__main__":
    compare()