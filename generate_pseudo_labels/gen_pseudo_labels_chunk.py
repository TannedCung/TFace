import numpy as np
import os
from tqdm import tqdm
import random
import operator
# from tkinter import _flatten
import pdb
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance
import torch
import time
import threading
from tqdm import tqdm
 
def buildDict_people(featlist, line_num, num_people_per_open):      # Building data dictionary
    '''
    This method is to build data dictionary 
    for collecting positive and negative pair similarities
    '''
    # line_num = chunk[0]
    num_people_listed = 0
    last_person = ""

    imgName = []
    peopleName = []
    peopleList = set()
    # with open(datalistFile, 'r') as f: datalist = f.readlines()
    # with open(featsFile, 'r') as f: featlist = f.readlines()
    # assert(len(datalist) == len(featlist)), "[ERROR]: length of image list not align with feature list"
    featsDict = {}
    # feats = np.zeros([len(featlist), 512])
    # feats = np.load("/Data/feats_2m.npy")
    for i, d in enumerate(tqdm(featlist[line_num:])): 
        tmpName = "/"+"/".join(d.split()[0].split("/")[-3:])
        # tmpeopleName = tmpName.split('/')[-2]
        tmpeopleName = d.split()[1]
        if tmpeopleName != last_person:
            num_people_listed += 1
            last_person = tmpeopleName
        if num_people_listed > num_people_per_open:
            break
        imgName.append(tmpName)
        # this_feat = np.load(d.split()[0])
        featsDict[tmpName] = [d.split()[0], tmpeopleName, d]
        # this_feat = featlist[i].split()[0]
   
        # feats[i] = this_feat
        peopleName.append(tmpeopleName)
        peopleList.add(tmpeopleName)
    feats = np.zeros([len(featsDict.keys()), 512])
    for j, (k, v) in enumerate(tqdm(featsDict.items())):
        this_feat = np.load(v[0])
        feats[j] = this_feat

    line_num += len(peopleName)
    # feats = np.load(featsFile)
    peopleList = sorted(list(peopleList))
    # for index, value in enumerate(imgName):
    #     featsDict[value] = feats[index,:]
    print('='*20 + 'LOADING DATALIST' + '='*20)
    print(f"People Number: {len(peopleList)}")
    print(f"Image Number: {len(peopleName)}")
    print(f"Features Shape: {np.shape(feats)}")
    print('='*56)
    return line_num, featsDict, peopleList, peopleName, feats

def getIndex(target_value, data_list):       # Search
    '''
    Search the index of sample by identity
    '''   
    # index = [i for i, v in enumerate(data_list) if v == target_value]
    index = np.where(data_list == target_value)[0]
    return index

def cos(feats1, feats2):
    '''
    Computing cosine distance
    For similarity
    '''   
    cos = np.dot(feats1, feats2) / (np.linalg.norm(feats1) * np.linalg.norm(feats2))
    return cos

def gen_samepeople_Similarity(featsDict, peopleList, peopleName, feats, outfile_dist_info, verbose=False, fix_num=24):
    '''
    This method is to collect positive pair similarities
    '''
    if verbose:
        print('='*20 + 'GENERATE POSITIVE PAIRS' + '='*20)
    imgsName = list(featsDict.keys())
    imgsame_list1 = []
    imgsame_list2 = []
    same_similaritys = []
    onepeople_count = 0
    id_similaritys = []
    pos_similarity_dist = {}
    people_list = tqdm(peopleList) if verbose else peopleList
    for tar_people in people_list:
        index = getIndex(tar_people, peopleName) # filter bases on people name which is the (fake) class in the .label file
        assert len(index) != 1
        id_similaritys = []
        for i in range(0,len(index)):
            compared_index = [value for value in index if index[i] != value and "_aug" not in featsDict[imgsName[value]][2] and "_mask" not in featsDict[imgsName[value]][2]]
            random.shuffle(compared_index)
            # avoid out_index
            pick_num = fix_num if len(compared_index) > fix_num else len(compared_index)            
            similaritys = []
            for j in range(pick_num):
                imgsame_list1.append(imgsName[index[i]])
                imgsame_list2.append(imgsName[compared_index[j]])
                embedding_similarity = cos(feats[index[i]], feats[compared_index[j]])
                same_similaritys.append(embedding_similarity)
                similaritys.append(embedding_similarity)
                id_similaritys.append(embedding_similarity)
            pos_similarity_dist[imgsName[index[i]]] = np.asarray(similaritys)
    same_similaritys = np.asarray(same_similaritys)
    assert len(imgsame_list1) == len(imgsame_list2) == len(same_similaritys)
    if verbose:
        print(f"Positive Pair Number: {len(imgsame_list1)}")
        print(f"Maximum similarity value: {np.max(same_similaritys)}")
        print(f"Mean similarity value: {np.mean(same_similaritys)}")
        print(f"Minute similarity value: {np.min(same_similaritys)}")
        print(f"Std of similarity value: {np.std(same_similaritys)}")
        print('='*63)
    # Output the information of similarity
    outfile_dist_info.write('='*20 + " For Positive Pairs " + '='*20 + '\n')                       
    outfile_dist_info.write(f"One People Number: {onepeople_count}"+ '\n')
    outfile_dist_info.write(f"Positive Pair Number: {len(imgsame_list1)}"+ '\n')
    outfile_dist_info.write(f"Maximum similarity value: {np.max(same_similaritys)}"+ '\n')
    outfile_dist_info.write(f"Mean similarity value: {np.mean(same_similaritys)}"+ '\n')
    outfile_dist_info.write(f"Minimum similarity value: {np.min(same_similaritys)}"+ '\n')
    outfile_dist_info.write(f"Std of similarity value: {np.std(same_similaritys)}"+ '\n')
    return pos_similarity_dist, len(same_similaritys)

def gen_diffpeople_Similarity(featsDict, peopleList, peopleName, feats, allpospair_nums, outfile_dist_info, verbose=False, fix_num=24):
    '''
    This method is to collect negative pair similarities
    '''
    if verbose:
        print('='*20 + 'GENERATE NEGATIVE PAIRS' + '='*20)
    imgsName = list(featsDict.keys())
    imgdiff_list1 = []
    imgdiff_list2 = []
    diff_similaritys = []
    id_similaritys = []
    onepeople_count = 0
    neg_similarity_dist = {}
    # pick_num = round(allpospair_nums / len(peopleName)) #mean pick num
    pick_num = fix_num
    compared_imgname = [value for value in imgsName]
    random.shuffle(compared_imgname)
    people_list = tqdm(peopleList) if verbose else peopleList
    for tar_people in people_list:
        id_similaritys = []
        same_idx_list = getIndex(tar_people, peopleName)
        for i in same_idx_list:
            similaritys = []
            repetition = []
            pick_count = 0
            rand_num = 0
            while pick_count < pick_num:   # number of same people in negative pairs
                rand_num = random.randint(0,len(compared_imgname)-1)
                if compared_imgname[rand_num] in repetition: continue # check repeated pairs 
                if featsDict[compared_imgname[rand_num]][1] == featsDict[compared_imgname[i]][1]: continue # check the same people
                repetition.append(compared_imgname[rand_num])
                imgdiff_list1.append(imgsName[i])
                imgdiff_list2.append(compared_imgname[rand_num])
                embedding_similarity = cos(feats[i], feats[rand_num])
                diff_similaritys.append(embedding_similarity)
                similaritys.append(embedding_similarity)
                rand_num += 1
                pick_count +=1
                id_similaritys.append(embedding_similarity)
            neg_similarity_dist[imgsName[i]] = np.asarray(similaritys)
    diff_similaritys = np.asarray(diff_similaritys)
    assert len(imgdiff_list1) == len(imgdiff_list2) == len(diff_similaritys)
    if verbose:
        print(f"One People Number: {onepeople_count}")
        print(f"Negative Pair Number: {len(imgdiff_list1)}")
        print(f"Maximum similarity value: {np.max(diff_similaritys)}")
        print(f"Mean similarity value: {np.mean(diff_similaritys)}")
        print(f"Minute similarity value: {np.min(diff_similaritys)}")
        print(f"Std of similarity value: {np.std(diff_similaritys)}")
        print('='*63)
    outfile_dist_info.write('='*20 + " For Negative Pairs " + '='*20 + '\n')
    outfile_dist_info.write(f"One People Number: {onepeople_count}"+ '\n')
    outfile_dist_info.write(f"Negative Pair Number: {len(imgdiff_list1)}"+ '\n')
    outfile_dist_info.write(f"Maximum similarity value: {np.max(diff_similaritys)}"+ '\n')
    outfile_dist_info.write(f"Mean similarity value: {np.mean(diff_similaritys)}"+ '\n')
    outfile_dist_info.write(f"Minute similarity value: {np.min(diff_similaritys)}"+ '\n')
    outfile_dist_info.write(f"Std of similarity value: {np.std(diff_similaritys)}"+ '\n')
    return neg_similarity_dist

def gen_distance(pos_similarity_dist, neg_similarity_dist, outfile):
    '''
    This method is to calculate quality scores via wasserstein distance
    '''
    outfile = open(outfile, 'w')
    print('='*20 + 'OUTPUT' + '='*20)
    posimglist = list(pos_similarity_dist.keys())
    negimglist = list(neg_similarity_dist.keys())
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
            imgpath.append(data_root + imgname[1:])
            featpath.append(data_root + imgname.replace(".jpg", ".npy")[1:])

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

def chunk_down(peopleList, peopleName, num_people_per_chunk):
    """
    Return: index of chunk
    """
    list_indexes = [] # [[1,2,3,...], [4,5,6,...], ...]
    one_list = []
    num_people = 0
    print(f"[INFO]: Chunking down...")
    for person in peopleList:
        this_people_ids = getIndex(person, peopleName)
        one_list.extend(this_people_ids)
        num_people += 1
        if num_people >= num_people_per_chunk:
            num_people = 0
            list_indexes.append(one_list)
            one_list = []
    if len(one_list) > 0:
        list_indexes.append(one_list)
    print(f"[INFO]: Chunked down to {len(list_indexes)} chunks")
    return list_indexes

def a_chunk_gen(pos_similarity_dist, neg_similarity_dist, featsDict, peopleList, peopleName, feats, outfile_dist_info):
    _pos_similarity_dist, pos_pairs_num = gen_samepeople_Similarity(featsDict, peopleList, \
                                                    peopleName, feats, outfile_dist_info)
    _neg_similarity_dist = gen_diffpeople_Similarity(featsDict, peopleList, peopleName, \
                                                    feats, pos_pairs_num, outfile_dist_info)
    pos_similarity_dist.update(_pos_similarity_dist)
    neg_similarity_dist.update(_neg_similarity_dist)

def a_total_gen(quality_scores_metrix, featsDict, feats, peopleList, peopleName, create_dir, thread_num, return_back):
    outfile_dist_info = f"{create_dir}/distribution_info_tmp_{thread_num}.txt"
    outfile_wdistacne = f"{create_dir}/w_distances_{thread_num}.txt"
    outfile_dist_info = open(outfile_dist_info, 'w')
    num_people_per_chunk = 500
    chunks = chunk_down(peopleList, peopleName, num_people_per_chunk=num_people_per_chunk)
    pos_similarity_dist = {}
    neg_similarity_dist = {}
    jobs = []
    for i, chunk in enumerate(chunks):
        _featsDict = {}
        # _feats = feats[chunk]
        _peopleName = peopleName[chunk]
        _peopleList = peopleList[i*num_people_per_chunk:(i+1)*num_people_per_chunk]
        _featsDict_keys = np.array(list(featsDict.keys()))[chunk]
        _featsDict_values = np.array(list(featsDict.values()))[chunk].tolist()
        _featsDict = dict(zip(_featsDict_keys,_featsDict_values))
        this_job = threading.Thread(target=a_chunk_gen, args=(pos_similarity_dist, neg_similarity_dist, _featsDict, _peopleList, _peopleName, feats[chunk], outfile_dist_info))
        this_job.start()
        print(f"[INFO]: Started job from {np.min(chunk)} to {np.max(chunk)}")
        jobs.append(this_job)
    
    for i, job in enumerate(jobs):
        job.join()
        print(f"[INFO]: job {i} th stopped")

    gen_distance(pos_similarity_dist, neg_similarity_dist, outfile_wdistacne)
    id_score = cal_idscore(outfile_wdistacne)
    featpath_select, imgpath_select, quality_scores_select = norm_labels(data_root, outfile_wdistacne, \
                                                            id_score)
    quality_scores_metrix[:,thread_num] = quality_scores_select
    return_back.append([featpath_select, imgpath_select])

if __name__ == "__main__":
    '''
    This method is to generate quality pseudo scores by similarity distribution distance (SDD)
    and save it to txt files
    '''
    # data_root    = '/mnt/nvme0n1/datasets/face/sample/nomask'
    data_root    = '/Data/'
    datalistFile = '/workspace/DATA.labelpath'
    featsFile    = '/workspace/DATA.labelfeature'
    # outfile_dist_info = f"{create_dir}/distribution_info_tmp.txt"
    # outfile_wdistacne = f"{create_dir}/w_distances.txt"
    # outfile_dist_info = open(outfile_dist_info, 'w')
    # peopleName = ["A-San", "A-San", "B_San"]
    # peopleList = ("A-San", "B-San")
    # print("[INFO]: Creating quality_scores_metrix")
    # print("[INFO]: Generating similarities")
    with open(featsFile, 'r') as f: featlist = f.readlines()
    # featlist = featlist[]
    num_people_per_open = 6000
    line_num = 0
    while 1:
        create_dir = f'/workspace/annotations_chunk_{line_num}'
        os.makedirs(create_dir, exist_ok=True)
        outfile_result    = f"{create_dir}/quality_pseudo_labels.txt"

        print(f"[INFO]: Building people dictionary from {line_num}")
        line_num, featsDict, peopleList, peopleName, feats = buildDict_people(featlist, line_num, num_people_per_open)
        if len(peopleList)<= 50:
            break
        quality_scores_metrix = np.zeros([len(peopleName), 12]) #
        peopleName = np.array(peopleName)
        
        return_back = []
        for i in range(12):
            a_total_gen(quality_scores_metrix, featsDict, feats, peopleList, peopleName, create_dir, i, return_back)

        quality_pseudo_labels = np.mean(quality_scores_metrix, axis=1)
        outfile_result = open(outfile_result, 'w')
        # output quality pseudo labels
        for index, value in enumerate(quality_pseudo_labels):                      
            outfile_result.write(return_back[0][1][index] + '\t' + str(value) + '\n')
