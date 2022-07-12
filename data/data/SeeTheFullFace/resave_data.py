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

LABEL = "/mnt/sdb1/data/widerface/train/label.txt"
NEW_LABEL = "/mnt/sdb1/data/widerface/train/new_label.txt"
COVERED_LABEL = "/mnt/sdb1/data/widerface/train/covered_label.txt"
PRETRAIN = "/mnt/sdb3/Git_clone/Face-Detector-1MB-with-landmark/data/SeeTheFullFace/checkpoints/maskMobileULite.pth"
AREA = 2500

parser = argparse.ArgumentParser(description='Retinaface')
parser.add_argument('-m', '--trained_model', default='checkpoints/mobilenet0.25_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--cpu', action="store_true", default=True, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
args = parser.parse_args()

TARGET_BRIGHTNESS = 0.43
TARGET_CONSTRAST = 0.43

def check_mask(model, transform, img):
    if img.shape[0]*img.shape[1] >0 :
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img).convert('RGB')
        img = transform(img)
        img = img.unsqueeze(0)
        res = model(img)
        r = np.argmax(res.detach().cpu())

        if r:
            return False, math.exp(res[0][r])/(math.exp(res[0][0])+math.exp(res[0][1]))
        else:
            return True, math.exp(res[0][r])/(math.exp(res[0][0])+math.exp(res[0][1]))
    
    else:
        return False, -1
    
class FPS_counter():
    def __init__(self):
        self.start = time.time()
        self.count = 0

    def step(self):
        self.now = time.time()
        self.count += 1
        fps = self.estimate()
        return fps
    
    def estimate(self):
        if self.count >=50 :
            self.count = 0
            self.start = time.time()
        fps = self.count/(self.now-self.start)
        self.count += 1
        return fps


if __name__ == '__main__':
    torch.set_grad_enabled(False)

    # net and model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    maskNet = torch.load(PRETRAIN, map_location=device)
    maskNet.eval()

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = T.Compose([
                            T.Resize((112,112)),
                            T.ToTensor(),
                            T.Normalize(mean, std)])


    print('Finished loading model!')
    # Load labels
    f = open(LABEL,'r')
    # n_f = open(NEW_LABEL, "a")
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
            path = line[2:]
            path = LABEL.replace('label.txt','images/') + path
            imgs_path.append(path)
            imgs_save_path.append(path.replace("images/", "unmask/"))
            imgs_covered_path.append(path.replace("images/", "covered/"))
        else:
            line = line.split(' ')
            label = [float(x) for x in line]
            labels.append(label)
    words.append(labels)
    skip_count = 0
    save_count = 0
    for index in tqdm(range(len(imgs_path))):
        save = True
        img = cv2.imread(imgs_path[index])
        folder = ""
        path_split = imgs_save_path[index].split("/")[:-1]
        for p in path_split:
            folder += p + "/" if p != path_split[-1] else p
        os.makedirs(folder, exist_ok=True)
        os.makedirs(folder.replace("unmask", "covered"), exist_ok=True)
        height, width, _ = img.shape

        annotations = np.zeros((0, 15))
        labels = words[index]
        dets = []
        if len(labels) == 0:
            continue
        for idx, label in enumerate(labels):
            annotation = np.zeros((15)).astype(np.int32)
            # bbox
            annotation[0] = int(label[0])  # x1
            annotation[1] = int(label[1]) # y1
            annotation[2] = int(label[2])  # w
            annotation[3] = int(label[3])  # h
            # landmarks()
            annotation[4] = int(label[4])    # l0_x
            annotation[5] = int(label[5])    # l0_y
            annotation[6] = int(label[7])    # l1_x
            annotation[7] = int(label[8])    # l1_y
            annotation[8] = int(label[10])   # l2_x
            annotation[9] = int(label[11])   # l2_y
            annotation[10] = int(label[13])  # l3_x
            annotation[11] = int(label[14])  # l3_y
            annotation[12] = int(label[16])  # l4_x
            annotation[13] = int(label[17])  # l4_y

            this_det = []
            for i in annotation:
                this_det.append(i)
            dets.append(this_det)

        for det in dets:
            if det[3]*det[2] < AREA:
                continue
            else:
                face_in_frame = img[det[1]:det[1]+det[3], det[0]:det[0] + det[2]]
                res, prob = check_mask(maskNet, transform, face_in_frame) 
                # 1 = no mask
                if res:
                    cv2.imshow("test", face_in_frame)
                    cv2.waitKey(10)
                    save = False
                    print(f"skiped {skip_count}/{index} frames")
        if save:
            n_f = n_f = open(NEW_LABEL, "a")
            n_f.write(links[index] + "\n")
            cv2.imwrite(imgs_save_path[index], img)
            for det in dets:
                new_anno = ""
                for a in det:
                    new_anno += str(a) + " " if a != det[-1] else str(a)
                n_f.write(new_anno + " 1.0 " + "\n")
            n_f.close()
            save_count+=1
        else:
            n_f = n_f = open(COVERED_LABEL, "a")
            n_f.write(links[index] + "\n")
            cv2.imwrite(imgs_covered_path[index], img)
            for det in dets:
                new_anno = ""
                for a in det:
                    new_anno += str(a) + " " if a != det[-1] else str(a)
                n_f.write(new_anno + " 1.0 " + "\n")
            n_f.close()
            skip_count += 1

    print(f"Saved {save_count} frames, skiped {skip_count} frames from {len(imgs_path)} frames")
            
