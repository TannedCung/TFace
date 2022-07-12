from __future__ import print_function
import os
import argparse
import torch
import numpy as np
import cv2
import time
from torchvision import transforms as T
from PIL import Image
import math
from tqdm import tqdm
import glob

LABEL = "/mnt/sdb1/data/widerface/train/unmask_label.txt"
NEW_LABEL = "/mnt/sdb1/data/widerface/train/unmasked.txt"

# COVERED_LABEL = "/mnt/sdb1/data/widerface/train/covered_label.txt"
# PRETRAIN = "/mnt/sdb3/Git_clone/Face-Detector-1MB-with-landmark/data/SeeTheFullFace/checkpoints/maskMobileULite.pth"
# AREA = 2500

if __name__ == '__main__':
    # Load labels
    f = open(LABEL,'r')
    lines = f.readlines()
    isFirst = True
    labels = []
    words = []
    imgs_path = []
    imgs_covered_path = []
    imgs_save_path = []
    links = []
    for line in lines:
        line = line.rstrip()
        if line.startswith('#'):
            links.append(line)
            if isFirst is True:
                isFirst = False
            else:
                labels_copy = labels.copy()
                words.append(labels_copy)
                labels.clear()
        else:
            line = line.split(' ')
            label = [float(x) for x in line]
            labels.append(label)
    words.append(labels)
    for index in tqdm(range(len(links))):
        labels = words[index]
        if len(labels) == 0:
            continue


        n_f = n_f = open(NEW_LABEL, "a")
        n_f.write(links[index] + "\n")
        for l in labels:
            new_anno = ""
            for a in label:
                new_anno += str(a) + " "
            n_f.write(new_anno[0:-1] + " 1" + "\n")
        n_f.close()
        
            
