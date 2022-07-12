import torch
import torchvision.transforms as T

import os
pwd = os.path.realpath(os.path.dirname(__file__))


class Config:
    # dataset
    data_root = ''
    img_list = os.path.join(pwd, '../../DATA.labelpath')
    feat_list = os.path.join(pwd, '../../DATA.labelfeature')
    eval_model = '/mnt/sda2/ExternalHardrive/research/classification/TFace/generate_pseudo_labels/extract_embedding/model/MobileFaceNet_MS1M.pth'
    outfolder = os.path.join(pwd, '../../Embedding_Features.npy')
    # data preprocess
    transform = T.Compose([
        T.Resize((112, 112)),
        T.ToTensor(),
        T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
    ])
    # network settings
    backbone = 'MFN'               # [MFN, R_50]
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    multi_GPUs = [0]
    embedding_size = 512
    batch_size = 128
    pin_memory = True
    num_workers = 8
config = Config()
