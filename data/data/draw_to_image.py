import os
import os.path
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
from tqdm import tqdm

class Artist:
    def __init__(self, txt_path, image_dir):
        self.imgs_path = []
        self.imgs_draw_path =[]
        self.words = []
        f = open(txt_path,'r')
        lines = f.readlines()
        isFirst = True
        labels = []
        for line in lines:
            line = line.rstrip()
            if line.startswith('#'):
                if isFirst is True:
                    isFirst = False
                else:
                    labels_copy = labels.copy()
                    self.words.append(labels_copy)
                    labels.clear()
                path = line[2:]
                # path = txt_path.replace('.txt', "/") + path
                path = os.path.join(image_dir, path)
                self.imgs_path.append(path)
                self.imgs_draw_path.append(path.replace("images", "images_drawed_images"))
            else:
                line = line.split(' ')
                label = [float(x) for x in line]
                labels.append(label)

        self.words.append(labels)

    def __len__(self):
        return len(self.imgs_path)

    def run(self):
        for index in tqdm(range(len(self.imgs_path))):
            img = cv2.imread(self.imgs_path[index])
            folder = ""
            path_split = self.imgs_draw_path[index].split("/")[:-1]
            for p in path_split:
                folder += p + "/" if p != path_split[-1] else p
            os.makedirs(folder, exist_ok=True)
            height, width, _ = img.shape

            labels = self.words[index]
            annotations = np.zeros((0, 15))
            if len(labels) == 0:
                return annotations
            for idx, label in enumerate(labels):
                annotation = np.zeros((1, 15)).astype(np.int32)
                # bbox
                annotation[0, 0] = int(label[0])  # x1
                annotation[0, 1] = int(label[1]) # y1
                annotation[0, 2] = int((label[0] + label[2]))  # x2
                annotation[0, 3] = int((label[1] + label[3]))   # y2
                # landmarksint()
                annotation[0, 4] = int(label[4])    # l0_x
                annotation[0, 5] = int(label[5])    # l0_y
                annotation[0, 6] = int(label[7])    # l1_x
                annotation[0, 7] = int(label[8])    # l1_y
                annotation[0, 8] = int(label[10])   # l2_x
                annotation[0, 9] = int(label[11])   # l2_y
                annotation[0, 10] = int(label[13])  # l3_x
                annotation[0, 11] = int(label[14])  # l3_y
                annotation[0, 12] = int(label[16])  # l4_x
                annotation[0, 13] = int(label[17])  # l4_y

                if (annotation[0, 4]<0):
                    annotation[0, 14] = -1 *label[-1]
                else:
                    annotation[0, 14] = label[-1]
                if abs(label[-1]) == 1 :
                    color = (0,0,255)
                elif abs(label[-1]) == 2:
                    color = (0,255,0)
                else:
                    color = (255,0,0)
                img = cv2.rectangle(img, (annotation[0, 0], annotation[0, 1]), (annotation[0, 2], annotation[0, 3]), color, 1)
                for i in range(4, 14, 2):
                    if annotation[0, i] > 0:
                        img = cv2.circle(img, (annotation[0, i], annotation[0, i+1]), radius=2, color=(0, 0, 255), thickness=-1)
                # cv2.imshow("Hi there", img)
                cv2.imwrite(self.imgs_draw_path[index], img)
                
if __name__ == "__main__":
    a = Artist(txt_path="/mnt/sda1/HiEveryOneThisIsTannedCung/Data/yoloface/Cxview/train/labels.txt", image_dir="/mnt/sda1/HiEveryOneThisIsTannedCung/Data/yoloface/Cxview/train/images")
    # a = Artist(txt_path="/mnt/sda1/HiEveryOneThisIsTannedCung/Data/yoloface/part_masked/train/labels.txt", image_dir="/mnt/sda1/HiEveryOneThisIsTannedCung/Data/yoloface/part_masked/train/images")
    a.run()