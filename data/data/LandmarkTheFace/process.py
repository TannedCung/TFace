from bs4 import BeautifulSoup as bs
import os 
import glob
import numpy as np
import cv2
import onnxruntime as rt
from tqdm import tqdm

DATA_PATH = '/mnt/sdb1/data/FaceMaskDataset/train'
LABEL_PATH = '/mnt/sdb1/data/widerface/train/real_mask.txt'
LANDMARK_PATH = '/mnt/sdb3/Git_clone/Face-Detector-1MB-with-landmark/data/yoloface-landmark106/v3.onnx'

def forward_landmark(landmark_net, face_roi, bbox):
    ih, iw, _ = face_roi.shape
    sw = float(iw) / float(112)
    sh = float(ih) / float(112)
    input_name = landmark_net.get_inputs()[0].name
    label_name = landmark_net.get_outputs()[1].name

    blob = cv2.dnn.blobFromImage(face_roi, scalefactor=1 / 255.0, size=(112, 112))
    # landmark_net.setInput(blob)
    out = landmark_net.run([label_name], {input_name: blob.astype(np.float32)})[0]
    # out = landmark_net.forward()
    points = out[0].flatten()
    for i in range(int(len(points) / 2)):
        points[i * 2] = round(((points[i * 2] * 112) * sw) + bbox[0], 3)
        points[(i * 2) + 1] = round(((points[(i * 2) + 1] * 112) * sh) + bbox[1], 3)
    return points.reshape(-1,2)

def parse_xml(xml_file):
    content = []
    # Read the XML file
    with open(xml_file, "r") as file:
        # Read each line in the file, readlines() returns a list of lines
        content = file.readlines()
        # Combine the lines in the list into a string
        content = "".join(content)
        bs_content = bs(content, "xml")
        results = bs_content.find_all("object")
        dets = []
        for result in results:
            _l = result.find("name").getText()
            if _l =='face':
                label = 1
            elif _l =='face_mask':
                label = 2
            else:
                raise "unknown label"
            xmin = int(result.bndbox.xmin.text)
            ymin = int(result.bndbox.ymin.text)
            xmax = int(result.bndbox.xmax.text)
            ymax = int(result.bndbox.ymax.text)
            dets.append((xmin, ymin, xmax-xmin, ymax-ymin, label))
    return dets

def process():
    landmark_net = rt.InferenceSession(LANDMARK_PATH)
    list_images = list(glob.iglob(os.path.join(DATA_PATH, "*.jpg")))
    for img in tqdm(list_images):
        # try:
            image = cv2.imread(img)
            label_file = open(LABEL_PATH, "a")
            label_file.write("# " + img.split("/")[-1] + "\n")
            xml_file = img.replace(".jpg", ".xml")
            dets = parse_xml(xml_file=xml_file)
            for det in dets:
                anno = f"{det[0]} {det[1]} {det[2]} {det[3]} "
                # face_roi = image[det[1]:det[1]+det[3], det[0]:det[0]+det[2]]
                # lmks = forward_landmark(landmark_net, face_roi, det)
                # anno += str(lmks[38][0]) + " " + str(lmks[38][1]) + " 0.0 " \
                #     + str(lmks[88][0]) + " " + str(lmks[88][1]) + " 0.0 " \
                #     + str(lmks[86][0]) + " " + str(lmks[86][1]) + " 0.0 " \
                #     + str(lmks[52][0]) + " " + str(lmks[52][1]) + " 0.0 " \
                #     + str(lmks[61][0]) + " " + str(lmks[61][1]) + " 0.0 "
                anno += str(-1.0) + " " + str(-1.0) + " 0.0 " \
                      + str(-1.0) + " " + str(-1.0) + " 0.0 " \
                      + str(-1.0) + " " + str(-1.0) + " 0.0 " \
                      + str(-1.0) + " " + str(-1.0) + " 0.0 " \
                      + str(-1.0) + " " + str(-1.0) + " 0.0 "
                anno += f'{det[4]}\n' 
                label_file.write(anno)
            label_file.close()

        # except Exception as e:
        #     print(e)
        #     continue

if __name__ == "__main__":
    process()