from glob import glob
from tqdm import tqdm
import json
import cv2
import os
import shutil

dst_folder = 'workspace/room/cvpr24/samples/true'
true_list = []
with open('workspace/room/uni_v8/data/anno_test.txt', 'r') as f:
    for idx, line in enumerate(tqdm(f)):
        line = line.strip()
        line_dic = eval(line)
        true = line_dic['target']
        dst = os.path.join(dst_folder, 'pano_' + str(idx+1).zfill(3) + '.png')
        shutil.copy(true, dst)
f.close()
