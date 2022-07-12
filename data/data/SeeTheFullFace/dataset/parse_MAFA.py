import scipy.io as reader
import numpy as np
import cv2
import os
import glob
from refine_name import *
import time
from tqdm import tqdm

class Extractor():
    def __init__(self):
        self.sherrif = spellingPolice()
        self.SAVE_PATH = "./data/train"
        self.area_filter = 2200
        self.infor = {}
        self.start = time.time()
        self.OCC = {1 : 'mask_only',
                    2 : 'mask_plus',
                    3 : 'human_body'}
    
    def ratio_filter(self, W, H, ratio=3):
        if 1/ratio <= W/H <= ratio:
            return True
        else:
            return False

    def blur_filter(self, img, S, threshold=90):
        # print(S)
        if img.shape[0]*img.shape[1]>self.area_filter:
            blurness_max = threshold+(988397742/(S*S))
            if cv2.Laplacian(img, cv2.CV_64F).var() >= blurness_max:
                return True
            else:
                return False
        else:
            return False
        
    def parse_mat_file(self, source_dir='data/MAFA/train-images/images', save_dir='data/train_MAFA', matfile='data/MAFA/LabelTrainAll.mat'):
        # maybe in one image there could be more than just one image
        # mat["label_train"][0] -> 25876 images
        # mat["label_train"][0][i] -> each of 25876 images
        # mat["label_train"][0][i][1] -> name of the iamge
        # mat["label_train"][0][i][2][0] -> info
        # mat["label_train"][0][1][2][0][i] -> each info

        mat = reader.loadmat(matfile)
        mat =  mat["label_train"][0]
        file_names = []
        faces = []
        occ_types = []
        for m in tqdm(mat):      
            file_names.append(str(m[1])[2:-2])
            img = cv2.imread(os.path.join(source_dir, str(m[1])[2:-2]))
            for i, info in enumerate(m[2]):           
                x, y, w, h = info[0], info[1], info[2], info[3]
                occ_type = int(info[12])
                if occ_type in self.OCC.keys():
                    save = os.path.join(save_dir, self.OCC[occ_type])
                    if self.ratio_filter(W=w, H=h):
                        self.crop_and_save(img, loca=(x,y,w,h), save_dir=save, save_name="{}_{:.8}.jpg".format(self.OCC[occ_type], time.time()-self.start))
                        faces.append([x, y, w, h])
                        occ_types.append(occ_type)
        
        return faces, occ_types

    def crop_and_save(self, img, loca, save_dir, save_name):
        (x,y,w,h) = (loca)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save = img[y:y+h, x:x+w]
        if self.blur_filter(img=save, S=w*h) :
            cv2.imwrite(os.path.join(save_dir, save_name), save)

a = Extractor()
a.parse_mat_file()