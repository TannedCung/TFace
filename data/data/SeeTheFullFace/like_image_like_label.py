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

UNMASK_LABEL = "/mnt/sdb1/data/widerface/train/unmask_label.txt"

class Artist:
    def __init__(self, txt_path):
        self.imgs_path = []
        self.words = []
        if os.path.exists(UNMASK_LABEL):
            os.remove(UNMASK_LABEL)
        f = open(txt_path,'r')
        lines = f.readlines()
        isFirst = True
        labels = []
        self.links = []
        for line in lines:
            line = line.rstrip()
            if line.startswith('#'):
                self.links.append(line)
                if isFirst is True:
                    isFirst = False
                else:
                    labels_copy = labels.copy()
                    self.words.append(labels_copy)
                    labels.clear()
                path = line[2:]
                path = txt_path.replace('label.txt','unmask/') + path
                self.imgs_path.append(path)
            else:
                line = line.split(' ')
                label = [float(x) for x in line]
                labels.append(label)

        self.words.append(labels)

    def __len__(self):
        return len(self.imgs_path)

    def run(self):
        save_count = 0
        for index in tqdm(range(len(self.imgs_path))):
            if os.path.exists(self.imgs_path[index]):
                n_f = n_f = open(UNMASK_LABEL, "a")
                n_f.write(self.links[index] + "\n")

                # folder = ""
                # path_split = self.imgs_draw_path[index].split("/")[:-1]
                # for p in path_split:
                #     folder += p + "/" if p != path_split[-1] else p
                # os.makedirs(folder, exist_ok=True)

                labels = self.words[index]
                if len(labels) == 0:
                    continue
                for idx, label in enumerate(labels):
                    new_anno = ""
                    for a in label:
                        new_anno += str(a) + " "
                    n_f.write(new_anno[0:-1] + "\n")
                n_f.close()
                save_count+=1
        print(f"Remain {save_count} images over {len(self.imgs_path)} total")
                



if __name__ == '__main__':
    a = Artist(txt_path="/mnt/sdb1/data/widerface/train/label.txt")
    a.run()
            
