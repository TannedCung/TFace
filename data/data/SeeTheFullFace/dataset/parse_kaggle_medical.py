import numpy as np
import cv2
from PIL import Image
from bs4 import BeautifulSoup
import os
import glob
import time
from refine_name import *
from tqdm import tqdm
# from .. import ..utils.refine_name

class Extractor():
    def __init__(self):
        self.sherrif = spellingPolice()
        self.SAVE_PATH = "./data/train"
        self.area_filter = 2200
        self.infor = {}
        self.start = time.time()

    def ratio_filter(self, xmin, xmax, ymin, ymax, ratio=3):
        W = xmax - xmin
        H = ymax - ymin
        if 1/ratio <= W/H <= ratio:
            return True
        else:
            return False

    def blur_filter(self, img, S, threshold=90):
        blurness_max = threshold+(988397742/(S*S))
        if cv2.Laplacian(img, cv2.CV_64F).var() >= blurness_max:
            return True
        else:
            return False

    def get_info_from_xml(self, filepath):
        """
        Parse .xml file
        Args:
            filepath: path to .xml file
        Return:
            filename, class, position()
        """
        file = open(filepath, 'r')
        content = file.read()
        soup = BeautifulSoup(content, 'lxml')


        objs = list(soup.find_all("object"))
        file_name = soup.find("filename").text
        # obj = soup.find("object")
        boxes = []
        names = []
        # for i in range(len(objs)):
        #     if i > 0:
        #         obj = soup.find_next("object")
        #         # obj = soup.find_next("object")
        for obj in objs:
            name = obj.find("name").text
            name = self.sherrif.check_decoded(name)
            name = self.sherrif.check_special_characters(name)

            for box in obj.find_all("bndbox"):
                # a = box.find("xmin")
                Xmin = max(0, int(box.find("xmin").text))
                Ymin = max(0, int(box.find("ymin").text))
                Xmax = max(0, int(box.find("xmax").text))
                Ymax = max(0, int(box.find("ymax").text))

                names.append(name)
                boxes.append([Xmin, Ymin, Xmax, Ymax])
        return file_name, names, boxes

    def crop_and_save(self, filename, names, boxes, cls_path):
        try:
            image = cv2.imread(os.path.join(cls_path, filename))
            # Check spelling
            names = [self.sherrif.check_special_characters(name) for name in names]
            names = [self.sherrif.check_decoded(name) for name in names]
            for i in range(len(names)):
                save_path = os.path.join(self.SAVE_PATH, names[i])
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                xmin, ymin, xmax, ymax = boxes[i]
                if (xmax-xmin)*(ymax-ymin)>= self.area_filter and self.ratio_filter(xmin, xmax, ymin, ymax, ratio=4):
                    save_path = os.path.join(save_path, "{}_{:.8}.jpg".format(names[i], time.time()-self.start))
                    img = image[ymin:ymax, xmin:xmax]
                    if self.blur_filter(img, S=(xmax-xmin)*(ymax-ymin), threshold=90):
                        cv2.imwrite(save_path, img)
                        self.infor[names[i]] = self.infor[names[i]] + 1 if names[i] in self.infor.keys() else 1
                    # self.save_to_txt(save_path)
        except Exception as e:
            print (e)

    def parse(self, path):
        list_ano = list(glob.iglob(os.path.join(path, "*.xml")))
        
        for ano in tqdm(list_ano):
            filename, names, boxes = self.get_info_from_xml(ano)
            # filename = ano.replace("xml", "jpg")
            self.crop_and_save(filename, names, boxes, cls_path="data/medical-masks-dataset/images")

        for key in self.infor.keys():
            print("{}: {}".format(key, self.infor[key]))
        self.save_to_txt(self.infor, "./data/infor.txt")
        for i in self.infor:
            print ("{}:{}".format(i, self.infor[i]))

    def save_to_txt(self, what2save, where2save):
        file = open(where2save, "w")
        if isinstance(what2save,dict):
            what2save = {k: v for k, v in sorted(what2save.items(), key=lambda item: item[1])}
            for key in what2save:
                save = "{}: {}".format(key, what2save[key])
                file.write(save)
        # file.write(what2save)
        file.close()       

if __name__ == "__main__":
    a = Extractor()
    a.parse("data/medical-masks-dataset/labels")         
            