import os
import glob
import shutil


def run(src_folder, dis_folder):
    max = 1000
    count = 0
    dirs = os.listdir(src_folder)
    for dir in dirs:
        files = list(glob.iglob(os.path.join(os.path.join(src_folder, dir), "*.jpg")))
        for file in files:
            shutil.copyfile(file, os.path.join(dis_folder, f"{count}.jpg"))
            count += 1
            if count >= max:
                break
        if count >= max:
                break

if __name__ == "__main__":
    SRC = "/mnt/sda1/HiEveryOneThisIsTannedCung/Data/lfw_exxtracted/Extracted_Faces/Extracted_Faces"
    DIST = "/mnt/sda1/HiEveryOneThisIsTannedCung/TFace/real_test/2"
    run(SRC, DIST)