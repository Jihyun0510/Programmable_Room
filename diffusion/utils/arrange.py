from tqdm import tqdm
from glob import glob
import os
import cv2
import sys
import shutil

coord_folder = '/dataset/control_db/control_v8/layout_coord'

root_path = './log_local_val'
dst_folder = os.path.join(root_path, 'image_log_arrange')

data_id_list = sorted(glob(os.path.join(root_path, 'image_log', 'train', '*data_id.txt')))
data_id_first = data_id_list[0]
f = open(data_id_first, "r")
lines = f.readlines()
batch_size = len(lines)
f.close()

src_list = sorted(glob(os.path.join(root_path, 'image_log', 'train', '*')))
src_types = ['data_id', 'local_control_0', 'local_control_1', 'prompt', 'reconstruction', 'sample']

for src in tqdm(src_list):
    src_step = src.split('/')[-1][:9]
    for src_type in src_types:
        if src_type in src.split('/')[-1][:-4]:
            if src.endswith('png'):
                img = cv2.imread(src)
                if batch_size == 3:
                    data_id_path = os.path.join(root_path, 'image_log', 'train', src.split('/')[-1][:27] + '_data_id.txt')
                    f_id = open(data_id_path, "r")
                    lines = f_id.readlines()
                    lines_new = []
                    for line in lines:
                        lines_new.append(line.strip())
                    f_id.close()
                    img0 = img[2:514, 2:514, :]
                    img1 = img[2:514, 516:1028, :]
                    img2 = img[2:514, 1030:1542, :]
                    img0 = cv2.resize(img0, (1024, 512))
                    img1 = cv2.resize(img1, (1024, 512))
                    img2 = cv2.resize(img2, (1024, 512))
                    dst = os.path.join(dst_folder, src_type)
                    os.makedirs(dst, exist_ok=True)
                    cv2.imwrite(os.path.join(dst, src_step + '_' + lines_new[0] + '_batch_0.jpg'), img0)
                    cv2.imwrite(os.path.join(dst, src_step + '_' + lines_new[1] + '_batch_1.jpg'), img1)
                    cv2.imwrite(os.path.join(dst, src_step + '_' + lines_new[2] + '_batch_2.jpg'), img2)
                elif batch_size == 2:
                    data_id_path = os.path.join(root_path, 'image_log', 'train', src.split('/')[-1][:27] + '_data_id.txt')
                    f_id = open(data_id_path, "r")
                    lines = f_id.readlines()
                    lines_new = []
                    for line in lines:
                        lines_new.append(line.strip())
                    f_id.close()
                    img0 = img[2:514, 2:514, :]
                    img1 = img[2:514, 516:1028, :]
                    img0 = cv2.resize(img0, (1024, 512))
                    img1 = cv2.resize(img1, (1024, 512))
                    dst = os.path.join(dst_folder, src_type)
                    os.makedirs(dst, exist_ok=True)
                    cv2.imwrite(os.path.join(dst, src_step + '_' + lines_new[0] + '_batch_0.jpg'), img0)
                    cv2.imwrite(os.path.join(dst, src_step + '_' + lines_new[1] + '_batch_1.jpg'), img1)
            elif 'prompt' in src:
                data_id_path = os.path.join(root_path, 'image_log', 'train', src.split('/')[-1][:27] + '_data_id.txt')
                f_id = open(data_id_path, "r")
                lines = f_id.readlines()
                lines_new = []
                for line in lines:
                    lines_new.append(line.strip())
                f_id.close()
                dst = os.path.join(dst_folder, src_type)
                os.makedirs(dst, exist_ok=True)
                f_src = open(src, "r")
                lines = f_src.readlines()
                for idx, line in enumerate(lines):
                    line = line.strip()
                    f_dst = open(os.path.join(dst, src_step + '_' + lines_new[idx] + '_batch_' + str(idx) + '.txt'), "w")
                    f_dst.write(line)
                    f_dst.close()
                f_src.close()
            elif 'data_id' in src:
                data_id_path = os.path.join(root_path, 'image_log', 'train', src.split('/')[-1][:27] + '_data_id.txt')
                f_id = open(data_id_path, "r")
                lines = f_id.readlines()
                lines_new = []
                for line in lines:
                    lines_new.append(line.strip())
                f_id.close()
                dst = os.path.join(dst_folder, 'coord')
                os.makedirs(dst, exist_ok=True)
                f_src = open(src, "r")
                lines = f_src.readlines()
                for idx, line in enumerate(lines):
                    line = line.strip()
                    coord_path = os.path.join(coord_folder, line + '_layout_coord.txt')
                    dst_path = os.path.join(dst, src_step + '_' + lines_new[idx] + '_batch_' + str(idx) + '.txt')
                    if os.path.isfile(coord_path):
                        shutil.copy(coord_path, dst_path)
                    else:
                        print('coord load error!')
                        sys.exit()
                f_src.close()