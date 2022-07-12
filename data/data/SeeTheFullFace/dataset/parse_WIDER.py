import numpy as np
import cv2
from PIL import Image
from bs4 import BeautifulSoup
import os
import glob
import time
from refine_name import *
from tqdm import tqdm

class Extractor():
    def __init__(self):
        self.sherrif = spellingPolice()
        self.SAVE_PATH = "./data/train"
        self.area_filter = 20000
        self.infor = {}
        self.start = time.time()
        self.count = 0

    def ratio_filter(self, W, H, ratio=2):
        if W*H == 0:
            return False
        elif 1/ratio <= W/H <= ratio:
            return True
        else:
            return False

    def blur_filter(self, img, S, threshold=60):
        # print(S)
        if img.shape[0]*img.shape[1]>self.area_filter:
            blurness_max = threshold+(988397742/(S*S))
            if cv2.Laplacian(img, cv2.CV_64F).var() >= blurness_max:
                return True
            else:
                return False
        else:
            return False

    def get_info_from_txt(self, txtfile):
        file = open(txtfile, 'r')
        info = file.readlines()
        file.close()
        return(info)

    def parse_txt(self, source_dir, save_dir, txtfile):
        info = self.get_info_from_txt(txtfile)
        for i, line in enumerate(tqdm(info)):
            if ".jpg" in line:
                img = cv2.imread(os.path.join(source_dir, line.replace('\n', '')))
            elif img is not None and len(line) > 8:
                in4 = line.split(' ')
                x, y, w, h = int(in4[0]), int(in4[1]), int(in4[2]), int(in4[3])
                save = img[y:y+h, x:x+w]
                if self.ratio_filter(w, h) and self.blur_filter(img=save, S=w*h):
                    save_name = os.path.join(save_dir, "{}.jpg".format(self.count))
                    cv2.imwrite(save_name, save)
                    self.count += 1
        

a = Extractor()
a.parse_txt(source_dir='data/images', save_dir='data/train_WIDER', txtfile='data/wider_face_split/wider_face_train_bbx_gt.txt')