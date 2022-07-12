import onnxruntime as rt
import cv2
import numpy as np
import h5py


def l2_norm(x, axis=1):
    """l2 norm"""
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    output = x / norm
    return output


def create_id(cls, img_ids):
    ids = []
    for c, img_id in zip(cls, img_ids):
        face_id = int((int(c)+1e6)*1e4+int(img_id))
        ids.append(face_id)

    return ids


class HDF5DatasetWriter:
    def __init__(self, length, outputPath, bufSize=10000):
        if outputPath.is_file():
            raise ValueError("The supplied ‘outputPath‘ already "
            "exists and cannot be overwritten. Manually delete "
            "the file before continuing.", outputPath)
        
        self.db = h5py.File(outputPath, "w")
        self.embs = self.db.create_dataset("embs", (length, 512), dtype=np.float32)
        self.ids = self.db.create_dataset("ids", (length,), dtype="int")

        self.bufSize = bufSize
        self.buffer = {"embs": [], "ids": []}
        self.idx = 0
    
    def add(self, embs, ids):
        self.buffer["embs"].extend(embs)
        self.buffer["ids"].extend(ids)
        if len(self.buffer["embs"]) >= self.bufSize:
            self.flush()
    
    def flush(self):
        # write the buffers to disk then reset the buffer
        i = self.idx + len(self.buffer["embs"])
        self.embs[self.idx:i] = self.buffer["embs"]
        self.ids[self.idx:i] = self.buffer["ids"]
        self.idx = i
        self.buffer = {"embs": [], "ids": []}
    
    def close(self):
        if len(self.buffer["embs"]) > 0:
            self.flush()
        self.db.close()


class OnnxInfer:
    def __init__(self, weight_paths, type="webface", use_gpu=True):
        if use_gpu:
            self.ort_session = rt.InferenceSession(weight_paths, providers=["CUDAExecutionProvider"])
        else:
            self.ort_session = rt.InferenceSession(weight_paths)
        self.input_name = self.ort_session.get_inputs()[0].name
        self.type = type
    
    
    def __call__(self, image_tensor):
        '''
        image: preprocessed image
        return: (batch, 512)
        '''
        onnx_output = self.ort_session.run(None, {self.input_name: image_tensor})
        if len(onnx_output) == 1:
            return l2_norm(onnx_output[0])
        else:
            return l2_norm(onnx_output[0]), onnx_output[1]
    
    
    def preprocess(self, img):
        '''
        image: pil data
        return: data preprocessed
        '''
        if self.type == "evolve":
            return self.preprocess_evolve(img)
        
        if self.type == "insightface" or self.type=="webface":
            return self.preprocess_insightface(img)
    
    
    def preprocess_evolve(self, img):
        img = np.array(img).astype(np.uint8)
        resized = cv2.resize(img, (128, 128))
        # center crop image
        a=int((128-112)/2) # x start
        b=int((128-112)/2+112) # x end
        c=int((128-112)/2) # y start
        d=int((128-112)/2+112) # y end
        ccropped = resized[a:b, c:d] # center crop the image
        # ccropped = ccropped[...,::-1] # BGR to RGB
        
        ccropped = ccropped.transpose(2, 0, 1)
        ccropped = ccropped.astype(np.float32)
        ccropped = (ccropped - 127.5) * 0.0078125
        
        return ccropped

    
    def preprocess_insightface(self, img):
        img = cv2.resize(img, (112,112))
        input_mean = 127.5
        input_std = 127.5
        img = np.array(img).astype(np.uint8)
        # img = img[...,::-1] # BGR to RGB
        img = img.transpose(2, 0, 1)
        img = img.astype(np.float32)
        img = (img - input_mean) / input_std
        
        return img
    

# if __name__ == "__main__":
#     model_type = "webface"
#     net = OnnxInfer(weight_paths="/research/classification/face.evolve/projects/face_mask/weights/webface_batch.onnx", type=model_type)
    
#     import timeit
#     while True:
#         img = np.random.randn(64, 3, 112, 112).astype(np.float32)
#         t0 = timeit.default_timer()
#         net(img)
#         t1 = timeit.default_timer()
#         print(t1-t0)
        
        
        
    