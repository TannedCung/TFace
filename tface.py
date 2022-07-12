import cv2
import numpy as np
import sys
sys.path.insert(1, "/research/classification/face.evolve/projects/face_mask/eval")

from onnx_infer import OnnxInfer


class TFace:
    def __init__(self, path) -> None:
        model_type = "webface"
        self.onnx_net = OnnxInfer(weight_paths=path, type=model_type, use_gpu=False)
    

    def predict(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (112, 112))
        img = self.onnx_net.preprocess(img)
        pred_score = self.onnx_net(img[np.newaxis])[1]

        return pred_score


if __name__=='__main__':
    img = cv2.imread("/research/classification/TFace/demo_imgs/24.jpg")
    model = TFace("/research/classification/face.evolve/projects/face_mask/modified1.onnx")
    print(model.predict(img))

