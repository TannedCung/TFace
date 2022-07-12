# with retinaface
import cv2
import numpy as np
import sys
import os
from imutils.video import VideoStream, FileVideoStream
import imutils
pwd = os.path.dirname(__file__)
sys.path.insert(1, os.path.join(pwd, ".."))
sys.path.insert(1, "/research/object_detection/face/Face-Detector-1MB-with-landmark/")

from tface import TFace
from retinaface import FaceDetector
from utils.infer_utils import align_face

# vs = VideoStream("rtsp://localhost:8554/stream1").start()
vs = VideoStream("rtsp://admin:meditech123@192.168.100.90:554").start()
fd = FaceDetector("/research/object_detection/face/Face-Detector-1MB-with-landmark/weights/mobilenet0.25_Final.pth")
model = TFace("/research/classification/face.evolve/projects/face_mask/modified2.onnx")

while True:
    frame = vs.read()
    # frame = imutils.rotate_bound(frame, 90)
    if frame is None:
        continue
    
    frame = frame.copy()
    
    bboxs, scores, landms = fd.detect(frame)
    for i in range(scores.shape[0]):
        color = (0, 0, 255)
        x1, y1, x2, y2 = bboxs[i].astype(np.int32)

        alighed_face = align_face(frame, landms[i].copy())

        face_score = model.predict(alighed_face)
        text_score = "score: {:.3f}".format(face_score[0, 0])
        cv2.putText(frame, text_score, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        landm = landms[i].astype(np.int32)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame, tuple(landm[0]), 1, (0, 0, 255), 4)
        cv2.circle(frame, tuple(landm[1]), 1, (0, 255, 255), 4)
        cv2.circle(frame, tuple(landm[2]), 1, (255, 0, 255), 4)
        cv2.circle(frame, tuple(landm[3]), 1, (0, 255, 0), 4)
        cv2.circle(frame, tuple(landm[4]), 1, (255, 0, 0), 4)
    
    # fps.update()
    # text_fps = "FPS: {:.3f}".format(fps.get_fps_n())
    # cv2.putText(frame, text_fps, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("frame", imutils.resize(frame, height=1000))
    key = cv2.waitKey(1) & 0xff
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()