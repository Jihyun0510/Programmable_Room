from glob import glob
from tqdm import tqdm
import json
import cv2

depth_list = sorted(glob('dataset/control_db/control_v8/depth_np_img/*'))
layout_list = sorted(glob('dataset/control_db/control_v8/layout_gt_gray/*'))
target_list = sorted(glob('dataset/control_db/control_v8/empty/*'))
coord_list = sorted(glob('dataset/control_db/control_v8/layout_coord/*'))
content_list = sorted(glob('dataset/control_db/control_v8/content/*'))
prompt_list = []

prompt_paths = sorted(glob('dataset/control_db/control_v8/txt_upgrade/*'))
for prompt_path in tqdm(prompt_paths):
    with open(prompt_path, 'r') as f:
        prompt = f.readline()
        prompt_list.append(prompt)
    f.close()

print()
print('>>> train & test')
print(len(depth_list))
print(len(layout_list))
print(len(target_list))
print(len(coord_list))
print(len(content_list))
print(len(prompt_list))

# split train & test
depth_list_train = depth_list[:int(len(depth_list)*0.8)]
layout_list_train = layout_list[:int(len(layout_list)*0.8)]
target_list_train = target_list[:int(len(target_list)*0.8)]
coord_list_train = coord_list[:int(len(coord_list)*0.8)]
content_list_train = content_list[:int(len(content_list)*0.8)]
prompt_list_train = prompt_list[:int(len(prompt_list)*0.8)]

depth_list_test = depth_list[int(len(depth_list)*0.8):]
layout_list_test = layout_list[int(len(layout_list)*0.8):]
target_list_test = target_list[int(len(target_list)*0.8):]
coord_list_test = coord_list[int(len(coord_list)*0.8):]
content_list_test = content_list[int(len(content_list)*0.8):]
prompt_list_test = prompt_list[int(len(prompt_list)*0.8):]

print()
print('>>> train')
print(len(depth_list_train))
print(len(layout_list_train))
print(len(target_list_train))
print(len(coord_list_train))
print(len(content_list_train))
print(len(prompt_list_train))
print()
print('>>> test')
print(len(depth_list_test))
print(len(layout_list_test))
print(len(target_list_test))
print(len(coord_list_test))
print(len(content_list_test))
print(len(prompt_list_test))


with open('./anno_train.txt', 'w') as f:
    for idx in range(len(prompt_list_train)):
        data = {}
        data['depth'] = depth_list_train[idx]
        data['layout'] = layout_list_train[idx]
        data['target'] = target_list_train[idx]
        data['coord'] = coord_list_train[idx]
        data['content'] = content_list_train[idx]
        data['prompt'] = prompt_list_train[idx]
        data_str = str(data)
        f.write(data_str)
        f.write('\n')
f.close()

with open('./anno_test.txt', 'w') as f:
    for idx in range(len(prompt_list_test)):
        data = {}
        data['depth'] = depth_list_test[idx]
        data['layout'] = layout_list_test[idx]
        data['target'] = target_list_test[idx]
        data['coord'] = coord_list_test[idx]
        data['content'] = content_list_test[idx]
        data['prompt'] = prompt_list_test[idx]
        data_str = str(data)
        f.write(data_str)
        f.write('\n')
f.close()
