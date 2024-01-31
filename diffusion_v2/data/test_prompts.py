from glob import glob
from tqdm import tqdm
import json
import cv2

prompt_list = []
with open('workspace/room/uni_v8/data/anno_test.txt', 'r') as f:
    for line in f:
        line = line.strip()
        line_dic = eval(line)
        prompt = line_dic['prompt']
        prompt_list.append(prompt)
f.close()
