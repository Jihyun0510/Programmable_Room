import sys
sys.path.append('workspace/room/Uni-ControlNet')
print(sys.path)
from annotator.content import ContentDetector
from glob import glob
from tqdm import tqdm
import cv2
import numpy as np
import os

condet = ContentDetector()

img_path = sorted(glob('dataset/control_db/control_v7/empty/*'))
np_path = 'dataset/control_db/control_v7/numpy'

for img in tqdm(img_path):
    id = img.split('/')[-1][:-10]
    src = cv2.imread(img)
    det = condet(src)
    dst = os.path.join(np_path, id)
    np.save(dst, det)
    