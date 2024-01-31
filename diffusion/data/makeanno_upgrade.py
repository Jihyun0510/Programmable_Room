from glob import glob
from tqdm import tqdm
import json
import cv2

fail_list = []
with open('fail_text.txt', 'r') as f:
    for line in f:
        line = line.strip()
        fail_id = line.split('/')[-1][:-10]
        fail_list.append(fail_id)
f.close()

depth_list = sorted(glob('diffusion_dataset/depth/*'))
layout_list = sorted(glob('diffusion_dataset/layout_image/*'))
semantic_list = sorted(glob('diffusion_dataset/new_semantic/*'))
target_list = sorted(glob('diffusion_dataset/empty/*'))
coord_list = sorted(glob('diffusion_dataset/layout_coord/*'))
content_list = sorted(glob('diffusion_dataset/content/*'))
prompt_list = []

prompt_paths = sorted(glob('diffusion_dataset/text/*'))
for prompt_path in tqdm(prompt_paths):
    with open(prompt_path, 'r') as f:
        prompt = f.readline()
        prompt_list.append(prompt)
    f.close()

print()
print('>>> train & test')
print(len(depth_list))
print(len(layout_list))
print(len(semantic_list))
print(len(target_list))
print(len(coord_list))
print(len(content_list))
print(len(prompt_list))

depth_list_upgrade = []
layout_list_upgrade = []
semantic_list_upgrade = []
target_list_upgrade = []
coord_list_upgrade = []
content_list_upgrade = []
prompt_list_upgrade = []

for depth, layout, semantic, target, coord, prompt, content in zip(depth_list, layout_list, semantic_list, target_list, coord_list, prompt_list, content_list):
    file_id = depth.split('/')[-1][:-16]
    if file_id in fail_list:
        continue
    scene_num = file_id[6:11]
    if int(scene_num) >= 1000:
        break
    
    depth_list_upgrade.append(depth)
    layout_list_upgrade.append(layout)
    semantic_list_upgrade.append(semantic)
    target_list_upgrade.append(target)
    coord_list_upgrade.append(coord)
    prompt_list_upgrade.append(prompt)
    content_list_upgrade.append(content)
  
    


# split train & test
depth_list_train = depth_list_upgrade[:int(len(depth_list_upgrade)*0.8)]
layout_list_train = layout_list_upgrade[:int(len(layout_list_upgrade)*0.8)]
semantic_list_train = semantic_list_upgrade[:int(len(semantic_list_upgrade)*0.8)]
target_list_train = target_list_upgrade[:int(len(target_list_upgrade)*0.8)]
coord_list_train = coord_list_upgrade[:int(len(coord_list_upgrade)*0.8)]
content_list_train = content_list_upgrade[:int(len(content_list_upgrade)*0.8)]
prompt_list_train = prompt_list_upgrade[:int(len(prompt_list_upgrade)*0.8)]

depth_list_test = depth_list_upgrade[int(len(depth_list_upgrade)*0.8):]
layout_list_test = layout_list_upgrade[int(len(layout_list_upgrade)*0.8):]
semantic_list_test = semantic_list_upgrade[int(len(semantic_list_upgrade)*0.8):]
target_list_test = target_list_upgrade[int(len(target_list_upgrade)*0.8):]
coord_list_test = coord_list_upgrade[int(len(coord_list_upgrade)*0.8):]
content_list_test = content_list_upgrade[int(len(content_list_upgrade)*0.8):]
prompt_list_test = prompt_list_upgrade[int(len(prompt_list_upgrade)*0.8):]

print()
print('>>> train')
print(len(depth_list_train))
print(len(layout_list_train))
print(len(semantic_list_train))
print(len(target_list_train))
print(len(coord_list_train))
print(len(content_list_train))
print(len(prompt_list_train))
print()
print('>>> test')
print(len(depth_list_test))
print(len(layout_list_test))
print(len(semantic_list_test))
print(len(target_list_test))
print(len(coord_list_test))
print(len(content_list_test))
print(len(prompt_list_test))


with open('ProgrammableRoom/diffusion_v2/data/anno_train.txt', 'w') as f:
    for idx in range(len(prompt_list_train)):
        data = {}
        data['depth'] = depth_list_train[idx]
        data['layout'] = layout_list_train[idx]
        data['semantic'] = semantic_list_train[idx]
        data['target'] = target_list_train[idx]
        data['coord'] = coord_list_train[idx]
        data['content'] = content_list_train[idx]
        data['prompt'] = prompt_list_train[idx]
        data_str = str(data)
        f.write(data_str)
        f.write('\n')
f.close()

with open('ProgrammableRoom/diffusion_v2/data/anno_test.txt', 'w') as f:
    for idx in range(len(prompt_list_test)):
        data = {}
        data['depth'] = depth_list_test[idx]
        data['layout'] = layout_list_test[idx]
        data['semantic'] = semantic_list_test[idx]
        data['target'] = target_list_test[idx]
        data['coord'] = coord_list_test[idx]
        data['content'] = content_list_test[idx]
        data['prompt'] = prompt_list_test[idx]
        data_str = str(data)
        f.write(data_str)
        f.write('\n')
f.close()
