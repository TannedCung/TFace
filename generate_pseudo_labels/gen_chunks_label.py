import numpy as np
import os
from tqdm import tqdm
import random
import pdb
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance
import torch
import time
import threading
import glob
from tqdm import tqdm

def gen_distance(pos_similarity_dist, neg_similarity_dist, outfile):
    '''
    This method is to calculate quality scores via wasserstein distance
    '''
    outfile = open(outfile, 'w')
    print('='*20 + 'OUTPUT' + '='*20)
    posimglist = list(pos_similarity_dist.keys())
    negimglist = list(pos_similarity_dist.keys())
    assert len(posimglist) == len(negimglist)
    imglists = posimglist
    for img in tqdm(imglists):
        tar_pos_similaritys = pos_similarity_dist[img]
        tar_neg_similaritys = neg_similarity_dist[img]
        w_distance = wasserstein_distance(tar_pos_similaritys, tar_neg_similaritys)
        outfile.write(img + '\t' + str(w_distance) + '\n')

def cal_idscore(result_file):
    '''
    This method is to calculate quality scores of identity
    '''
    with open(result_file, 'r') as f: txtContent = f.readlines()
    peoid= []
    quality_scores = []
    idsocre_dist = {}
    for i in txtContent: idsocre_dist[i.split()[0].split('/')[1]] = [0,0]  # init
    quality_scores = []
    for i in txtContent: quality_scores.append(float(i.split()[1]))
    # normalize quality pseudo labels
    quality_scores = (quality_scores - np.min(quality_scores)) / \
                     (np.max(quality_scores) - np.min(quality_scores)) * 100
    count = 0
    peoid = set()
    for i in tqdm(txtContent):
        idname = i.split()[0].split('/')[1]
        idsocre_dist[idname][0] += quality_scores[count]
        idsocre_dist[idname][1] += 1
        count += 1
        peoid.add(idname)
    peoid = list(peoid)
    id_score = {}
    for i in tqdm(peoid): id_score[i] = (idsocre_dist[i][0] / idsocre_dist[i][1])
    return id_score

def norm_labels(data_root, outfile_wdistacne, id_score):
    '''
    This method is to normalize quality scores
    '''
    # outfile_result = open(outfile_result, 'w')
    with open(outfile_wdistacne, 'r') as f: txtContent = f.readlines()
    imgpath = []
    featpath = []
    quality_scores = []
    for i in tqdm(txtContent):
        imgname = i.split()[0]
        mean_wdistance = float(i.split()[1])
        if "outsider" in imgname:
            imgpath.append(data_root + imgname.split("/")[2] + "/" + imgname.split("/")[3])
            featpath.append(data_root + imgname.split("/")[2] + "/" + imgname.split("/")[3].replace(".jpg", ".npy"))
        else:
            imgpath.append(data_root + imgname)
            featpath.append(data_root + imgname.replace(".jpg", ".npy"))

        quality_scores.append(mean_wdistance)
    quality_scores = np.asarray(quality_scores)
    quality_scores = (quality_scores-np.min(quality_scores)) / \
                     (np.max(quality_scores) - np.min(quality_scores)) * 100
    quality_scores_select = []
    imgpath_select = []
    featpath_select = []
    for i in tqdm(range(len(quality_scores))):
        quality_scores_select.append(quality_scores[i])
        imgpath_select.append(imgpath[i])
        featpath_select.append(featpath[i])

    quality_scores_select = (quality_scores_select-np.min(quality_scores_select)) / \
                            (np.max(quality_scores_select) - np.min(quality_scores_select)) * 100
    return featpath_select, imgpath_select, quality_scores_select

def take_first(string):
    return string.split()[0]

def aggregate(folders, outfolder):
    os.makedirs(outfolder, exist_ok=True)
    # outfile_distances = os.path.join(outfolder, "w_distances.txt")
    outfile_quality = os.path.join(outfolder, "quality.txt")
    data_root = "/Data"

    total_files = {}
    for folder in folders:
        files = list(glob.iglob(os.path.join(folder, "w_distances_*.txt")))
        for file in files:
            filename = file.split("/")[-1]
            with open(file, 'r') as f: lines = f.readlines()

            if filename not in total_files.keys():
                total_files[filename] = lines
            else:
                total_files[filename] += lines

    for k, v in total_files.items():
        assert len(list(total_files.values())[0]) == len(v), "Misalign between 12 chunks length"

    quality_scores_metrix = np.zeros([len(v), 12]) #
    first_imgpath_select = []
    for i in range(12):
        outfile_distances = os.path.join(outfolder, f"w_distances_{i}.txt")
        with open(outfile_distances, "w") as f:
            total_files[list(total_files.keys())[i]].sort(key=take_first)
            for line in total_files[list(total_files.keys())[i]]:
                f.write(line)
        id_score = cal_idscore(outfile_distances)
        featpath_select, imgpath_select, quality_scores_select = norm_labels(data_root, outfile_distances, \
                                                            id_score)
        if i==0:
            first_imgpath_select = imgpath_select
        assert first_imgpath_select==imgpath_select, "Misalign imgpath"

        quality_scores_metrix[:,i] = quality_scores_select
    quality_pseudo_labels = np.mean(quality_scores_metrix, axis=1)
    outfile_result = open(outfile_quality, 'w')
    # output quality pseudo labels
    for index, value in enumerate(quality_pseudo_labels):                      
        outfile_result.write(imgpath_select[index] + '\t' + str(value) + '\n')

if __name__ == "__main__":
    FOLDERS = ["/workspace/annotations_chunk_0",
                "/workspace/annotations_chunk_568023",
                "/workspace/annotations_chunk_1266284",
                "/workspace/annotations_chunk_1918099",
                "/workspace/annotations_chunk_3170390",
                "/workspace/annotations_chunk_2555813",
                "/workspace/annotations_chunk_3670530",
                "/workspace/annotations_chunk_4080729",
                "/workspace/annotations_chunk_4577621", 
                "/workspace/annotations_chunk_5004731",
                "/workspace/annotations_chunk_5500945",
                "/workspace/annotations_chunk_5991096",
                "/workspace/annotations_chunk_6404980",
                "/workspace/annotations_chunk_6827584",
                "/workspace/annotations_chunk_7603777",
                "/workspace/annotations_chunk_7208685"]
    OUT_FOLDER = "/workspace/annotations"

    aggregate(FOLDERS, OUT_FOLDER)