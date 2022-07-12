docker run -itd --net=host --privileged --shm-size=10.05gb --name TFace-gen  -v /mnt/sda1/HiEveryOneThisIsTannedCung/TFace:/workspace -v /mnt/nvme0n1/datasets/face:/Data nvcr.io/nvidia/ pytorch:21.10-py3


```
==== reproduce ==== 

cd generate_pseudo_labels
python gen_data_list.py # gen txt files, add outsider, mask, aug images
python extrace_embedding/extract_feats.py # from files above gen embeddings
python gen_pseudo_labels_chunk.py # output quality score

cd ..
python train.py [args] # train regresion
```