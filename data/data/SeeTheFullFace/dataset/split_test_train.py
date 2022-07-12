import os
import shutil
import glob


class Split():
    def __init__(self):
        self.TRAIN_PATH = "data/train_MAFA/V2/train" 
        self.TEST_PATH = "data/train_MAFA/V2/test"

    def copy_file(self, src, dst):
        shutil.copyfile(src, dst)
    
    def delete_file(self, file):
        os.remove(file)

    def split(self, path="data/train_MAFA/V2/train" , ratio=0.1):
        sub_dirs = []
        for d in os.listdir(path):
            if ".txt" not in d:
                sub_dirs.append(d)
        count = 0
        for sd in sub_dirs:
            old_dir = os.path.join(self.TRAIN_PATH, sd)
            new_dir = os.path.join(self.TEST_PATH, sd)
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
            
            imgs = os.listdir(old_dir)
            for i in range(int(ratio*len(imgs))):
                self.copy_file(os.path.join(old_dir,imgs[i]), os.path.join(new_dir,imgs[i]))
                self.delete_file(os.path.join(old_dir,imgs[i]))
                count += 1
        print("split {} files from train set".format(count))


sp = Split()
sp.split()