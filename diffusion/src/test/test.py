import sys
if './' not in sys.path:
	sys.path.append('./')
from diffusion_v2.utils.share import *
import utils.config as config
from diffusion_v2.utils.paint import layout2semantic

import cv2
import einops
import numpy as np
import os
from datetime import datetime
import torch
from pytorch_lightning import seed_everything
from glob import glob
from tqdm import tqdm
import random
import shutil

from annotator.util import resize_image, HWC3
from diffusion_v2.models.util import create_model, load_state_dict
from diffusion_v2.models.ddim_hacked import DDIMSampler


model = create_model('./configs/local_v15_cond3.yaml').cpu()
model_path = sorted(glob('workspace/room/diffusion_v2/ckpt/*'))
model.load_state_dict(load_state_dict(model_path[0], location='cuda'), strict=False)
model = model.cuda()
ddim_sampler = DDIMSampler(model)


def process(layout_image, depth_image, semantic_image, content_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, strength, scale, seed, eta, low_threshold, high_threshold, value_threshold, distance_threshold, alpha, global_strength):
    
    seed_everything(seed)

    if layout_image is not None:
        anchor_image = layout_image
    elif depth_image is not None:
        anchor_image = depth_image
    elif content_image is not None:
        anchor_image = content_image
    else:
        anchor_image = np.zeros((image_resolution, image_resolution, 3)).astype(np.uint8)
    H, W, C = resize_image(HWC3(anchor_image), image_resolution).shape

    with torch.no_grad():
        if layout_image is not None:
            layout_image = cv2.resize(layout_image, (W, H))
            layout_detected_map = layout_image
        else:
            layout_detected_map = np.zeros((H, W, C)).astype(np.uint8)
        if depth_image is not None:
            depth_image = cv2.resize(depth_image, (W, H))
            depth_detected_map = depth_image
        else:
            depth_detected_map = np.zeros((H, W, C)).astype(np.uint8)
        if semantic_image is not None:
            semantic_image = cv2.resize(semantic_image, (W, H))
            sementic_detected_map = semantic_image
        else:
            sementic_detected_map = np.zeros((H, W, C)).astype(np.uint8)
        if False:
            #content_emb = apply_content(content_image)
            content_emb = np.load(content_image)
        else:
            content_emb = np.zeros((768))

        detected_maps_list = [
            layout_detected_map,
            depth_detected_map,
            sementic_detected_map
        ]
        detected_maps = np.concatenate(detected_maps_list, axis=2)

        local_control = torch.from_numpy(detected_maps.copy()).float().cuda() / 255.0
        local_control = torch.stack([local_control for _ in range(num_samples)], dim=0)
        local_control = einops.rearrange(local_control, 'b h w c -> b c h w').clone()
        global_control = torch.from_numpy(content_emb.copy()).float().cuda().clone()
        global_control = torch.stack([global_control for _ in range(num_samples)], dim=0)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        uc_local_control = local_control
        uc_global_control = torch.zeros_like(global_control)
        cond = {"local_control": [local_control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)], 'global_control': [global_control]}
        un_cond = {"local_control": [uc_local_control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)], 'global_control': [uc_global_control]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength] * 13
        samples, _ = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond, global_strength=global_strength)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        results = [x_samples[i] for i in range(num_samples)]

    return [results, detected_maps_list]

if __name__ == "__main__":
    now = datetime.now()
    now_str = now.strftime('%Y_%m_%d_%H_%M_%S')
    
    prompts = []
    layouts = []
    depths = []
    semantics = []
    coords = []
    targets = []
    contents = []
    with open('./data/anno_test.txt', 'r') as f:
        for line in f:
            line = line.strip()
            line_dict = eval(line)
            prompts.append(line_dict['prompt'])
            layouts.append(line_dict['layout'])
            depths.append(line_dict['depth'])
            semantics.append(line_dict['semantic'])
            coords.append(line_dict['coord'])
            targets.append(line_dict['target'])
            contents.append(line_dict['content'])
    f.close()
    
    for itr, (prompt, layout, depth, semantic, coord, target, content) in enumerate(zip(tqdm(prompts), layouts, depths, semantics, coords, targets, contents)):
        print()
        print('>>> Itr :' + str(itr+1) + ' / ' + str(len(prompts)))
        save_folder = os.path.join('results', now_str, 'itr_' + str(itr+1))
        os.makedirs(save_folder, exist_ok=True)
        
        layout_image = cv2.imread(layout)
        depth_image = cv2.imread(depth)
        depth_image = cv2.resize(depth_image, (1024, 512))
        #semantic_image = cv2.imread(semantic)
        _, semantic_image = layout2semantic(coord)
        content_image = None
        a_prompt = '' # 'best quality, extremely detailed'
        n_prompt = '' # 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'
        num_samples = 4
        image_resolution = 512
        ddim_steps = 50
        strength = 1
        scale = 7.5
        seed = 42
        eta = 0.0
        low_threshold = 100
        high_threshold = 200
        value_threshold = 0.1
        distance_threshold = 0.1
        alpha = 6.2
        global_strength = 1
        
        [results, detected_maps] = process(layout_image, depth_image, semantic_image, content_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, strength, scale, seed, eta, low_threshold, high_threshold, value_threshold, distance_threshold, alpha, global_strength)
        
        save_path = os.path.join(save_folder, 'prompt.txt')
        f = open(save_path, "w")
        f.write(prompt)
        f.close()
        
        save_path = os.path.join(save_folder, 'coord.txt')
        shutil.copy(coord, save_path)
        
        save_path = os.path.join(save_folder, 'origin.png')
        shutil.copy(target, save_path)
        
        for idx, detected_map in enumerate(detected_maps):
            save_path = os.path.join(save_folder, 'detected_map_' + str(idx) + '.png')
            cv2.imwrite(save_path, detected_map)
        
        for idx, result in enumerate(results):
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            save_path = os.path.join(save_folder, 'pano_' + str(idx) + '.png')
            cv2.imwrite(save_path, result)
    
    print("Finished.")