from random import random
import cv2
import os
import glob
import numpy as np
import torch
from utils.box_utils import decode, decode_landm
from models.retinaface import RetinaFace as RF_model
from utils.nms.py_cpu_nms import py_cpu_nms
from layers.functions.prior_box import PriorBox
from data import cfg_mnet, cfg_slim, cfg_rfb

DETECTOR_PATH = "/mnt/sda1/HiEveryOneThisIsTannedCung/TFace/weights/mobilenet0.25_epoch_1133.pth"
SRC_PATH = "/mnt/sda1/HiEveryOneThisIsTannedCung/Data/landscape-pictures"
SAVE_PATH = "/mnt/nvme0n1/datasets/face/outsider_save"
TOTAL_CROP = 12000

class RetinaFace:
    def __init__(self, nms_threshold, keep_top_k, vis_thres, pretrained, cpu=True):
        self.nms_threshold = nms_threshold
        self.keep_top_k = keep_top_k
        self.obj_thresh = vis_thres
        self.top_k = 50
        self.confidence_threshold = 0.02
        self.cfg = cfg_mnet
        net = RF_model(cfg=self.cfg, phase = 'test')
        self.load_model(net, pretrained, load_to_cpu=True)
        self.model.eval()
        self.device = torch.device("cpu" if cpu else "cuda")
        self.model = self.model.to(self.device)
    
    @staticmethod
    def check_keys(model, pretrained_state_dict):
        ckpt_keys = set(pretrained_state_dict.keys())
        model_keys = set(model.state_dict().keys())
        used_pretrained_keys = model_keys & ckpt_keys
        unused_pretrained_keys = ckpt_keys - model_keys
        missing_keys = model_keys - ckpt_keys
        print('Missing keys:{}'.format(len(missing_keys)))
        print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
        print('Used keys:{}'.format(len(used_pretrained_keys)))
        assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
        return True

    def remove_prefix(self, state_dict, prefix):
        ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
        print('remove prefix \'{}\''.format(prefix))
        f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
        return {f(key): value for key, value in state_dict.items()}

    def load_model(self, model, pretrained_path, load_to_cpu):
        print('Loading pretrained model from {}'.format(pretrained_path))
        if load_to_cpu:
            pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
        else:
            device = torch.cuda.current_device()
            pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = self.remove_prefix(pretrained_dict['state_dict'], 'module.')
        else:
            pretrained_dict = self.remove_prefix(pretrained_dict, 'module.')
        self.check_keys(model, pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)
        self.model = model
    
    def preprocess(self, img):
        img = np.float32(img)
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        return img
    
    def postprocess(self, loc, conf, landms, scale, img, resize=1):
        priorbox = PriorBox(self.cfg, image_size=[img.shape[2], img.shape[3]])
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        scores_mask = conf.squeeze(0).data.cpu().numpy()[:, 2]
        landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(self.device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.unique(np.concatenate((np.where(scores > self.confidence_threshold)[0], np.where(scores_mask > self.confidence_threshold)[0])))
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]
        scores_mask = scores_mask[inds]
        _mask = np.greater(scores_mask, scores)
        scores = scores_mask*_mask + scores*(~_mask)
        _cls = _mask + 1
        # keep top-K before NMS
        order = scores.argsort()[::-1][:self.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]
        _cls = _cls[order]
        # print(_cls)

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis], _cls[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:self.keep_top_k, :]
        landms = landms[:self.keep_top_k, :]
        dets = np.concatenate((dets, landms), axis=1)
        return dets
        
    def detect(self, img):
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img = self.preprocess(img)
        scale = scale.to(self.device)
        loc, conf, landms = self.model(img)
        dets = self.postprocess(loc, conf, landms, scale, img)
        out = []
        for d in dets:
            if d[4] >= self.obj_thresh:
                out.append(d)
        return out

class OutSiderCrop:
    def __init__(self, src_path, save_path, total_crop, size=[112,112], crop_ensure_center=False):
        self.src_path = src_path
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
        self.src_images = list(glob.iglob(os.path.join(self.src_path, "*.jpg")))
        self.detector = RetinaFace(nms_threshold=0.4, keep_top_k=750, vis_thres=1.1, pretrained=DETECTOR_PATH)
        self.total_crop = total_crop
        self.count = len(os.listdir(self.save_path))
        self.crop_per_image = int((self.total_crop - self.count)/len(self.src_images))
        self.crop_ensure_center = crop_ensure_center
        self.size = size
    
    def face_exist(self, image):
        dets = self.detector.detect(image)
        if len(dets) > 0:
            return True
        else:
            return False

    def crop(self):
        exclude = []
        while self.count < self.total_crop:
            self.crop_per_image = np.ceil((self.total_crop - self.count)/len(self.src_images)).astype(np.int32)
            for i in range(len(self.src_images)):
                image = self.src_images[i]
                this_image = cv2.imread(image)
                h,w,c = this_image.shape
                if h <= self.size[0] or w <= self.size[1]:
                    continue
                if image in exclude:
                    continue
                if self.face_exist(this_image):
                    print(f"[INFO]: image {image} contains faces, skipping...")
                    exclude.append(image)
                    continue
                for j in range(self.crop_per_image):
                    if self.crop_ensure_center and j == 0:
                        random_cord = [int((w-self.size[0])/2), int((h-self.size[1])/2)]
                    else:
                        random_cord = [np.random.randint(0, w-self.size[0]), np.random.randint(0, h-self.size[1])]
                    this_crop = this_image[random_cord[1]: random_cord[1]+self.size[1], random_cord[0]: random_cord[0]+self.size[0]]
                    cv2.imwrite(os.path.join(self.save_path, f"{self.count}.jpg"), this_crop)
                    self.count += 1
                    print(self.count)
                    if self.count >= self.total_crop:
                        break
                if self.count >= self.total_crop:
                        break
                
if __name__ == "__main__":
    A_Yu_Jin = OutSiderCrop(SRC_PATH, SAVE_PATH, TOTAL_CROP)
    A_Yu_Jin.crop()

            

