import os
import os.path
from re import L
from tqdm import tqdm
import torch
import torch.utils.data as data
import cv2
import numpy as np

class Yolo2Wider:
    def __init__(self, yolo_label_dir, image_dir, save_txt):
        self.image_dir = image_dir
        self.yolo_label_dir = yolo_label_dir
        self.yolo_labels = os.listdir(yolo_label_dir)
        self.yolo_labels.remove("classes.txt")

        self.imgs = os.listdir(image_dir)
        self.count = 0
        self.mutual_names = self.synchronize(self.imgs,self.yolo_labels)
        self.save_txt = save_txt

    def synchronize(self, imgs, labels):
        shared = []
        for img in imgs:
            if img.replace(".jpg", ".txt") in labels:
                shared.append(img.replace(".jpg", ""))
        return shared

    def convert(self):
        save_file = open(self.save_txt, "w")

        for n in tqdm(self.mutual_names):
            img = n + ".jpg"
            save_file.write(f"# {img}\n")
            image = cv2.imread(f"{os.path.join(self.image_dir, img)}")
            img_h, img_w = image.shape[:2]
            txt = n + ".txt"
            label_f = open(os.path.join(self.yolo_label_dir, txt),'r')
            label_lines = label_f.readlines()
            label_f.close()

            these_fix_dets = []
            for f_l in label_lines:
                these_fix_dets.append([float(i) for i in f_l.rstrip().replace("  ", " ").split(" ")])

            for d in these_fix_dets:
                w = d[3]*img_w
                h = d[4]*img_h
                x = max(0, img_w*(d[1]-d[3]/2))
                y = max(0, img_h*(d[2]-d[4]/2))
                for i, v in enumerate(d[5:]):
                    d[i+5] = ((v*img_w*(i%2==0) + v*img_h*(i%2==1))) if v > 0 else -1.0

                str_label = "%.6f %.6f %.6f %.6f" % (x, y , w, h)
                str_label += " %.6f %.6f 0.0" % (d[5], d[6])
                str_label += " %.6f %.6f 0.0" % (d[7], d[8])
                str_label += " %.6f %.6f 0.0" % (d[9], d[10])
                str_label += " %.6f %.6f 0.0" % (d[11], d[12])
                str_label += " %.6f %.6f 0.0" % (d[13], d[14])

                str_label += f" {abs(int(d[0] + 1))}\n"
                save_file.write(str_label)
        save_file.close()

if __name__ == "__main__":
    YOLO_LABEL_DIR = "/workspace/data/Cxview/train/labels"
    IMAGE_DIR = "/workspace/data/Cxview/train/images"
    SAVE_TXT = "/workspace/data/Cxview/train/labels.txt"
    Converter = Yolo2Wider(YOLO_LABEL_DIR, IMAGE_DIR, SAVE_TXT)
    Converter.convert()
