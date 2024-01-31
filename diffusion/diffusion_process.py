# import sys
# if './' not in sys.path:
#     sys.path.append('./')
from diffusion.utils.share import *
import diffusion.utils.config as config

import cv2
import einops
import numpy as np
import os
from datetime import datetime
import torch
from pytorch_lightning import seed_everything
from glob import glob
# import random
# import shutil

from diffusion_v2.annotator.util import resize_image, HWC3
from diffusion_v2.models.util import create_model, load_state_dict
from diffusion_v2.models.ddim_hacked import DDIMSampler





class PanoramaGenerator:
    def __init__(self):
        self.model = create_model('./diffusion_v2/configs/local_v15_cond3.yaml').cpu()
        self.model.load_state_dict(load_state_dict('./diffusion_v2/ckpt/room.ckpt', location='cuda'), strict=False)
        self.model = self.model.cuda()
        self.ddim_sampler = DDIMSampler(self.model)
        self.a = "a"

    def process(self,layout_image, depth, semantic_image, content_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, strength, scale, seed, eta, low_threshold, high_threshold, value_threshold, distance_threshold, alpha, global_strength):
        
        seed_everything(seed)

        if layout_image is not None:
            anchor_image = layout_image.astype(np.uint8)

        else:
            anchor_image = np.zeros((image_resolution, image_resolution, 3)).astype(np.uint8)
        
        
        mx = np.max(depth)
        mn = np.min(depth)
        depth = (depth - mn)/(mx - mn) 
        depth *= 255
        depth = np.expand_dims(depth, axis=2) #edited

        depth = np.repeat(depth, 3, axis=2) #edited

        H, W, C = resize_image(HWC3(anchor_image), image_resolution).shape

        with torch.no_grad():
            if layout_image is not None:
                layout_image = cv2.resize(layout_image, (W, H))
                layout_detected_map = layout_image
            else:
                layout_detected_map = np.zeros((H, W, C)).astype(np.uint8)

            
            if semantic_image is not None:
                semantic_image = cv2.resize(semantic_image, (W, H))
                sementic_detected_map = semantic_image
            else:
                sementic_detected_map = np.zeros((H, W, C)).astype(np.uint8)
            if False:
                content_emb = np.load(content_image)
            else:
                content_emb = np.zeros((768))

            detected_maps_list = [
                layout_detected_map,
                depth,
                sementic_detected_map
            ]
            detected_maps = np.concatenate(detected_maps_list, axis=2)

            local_control = torch.from_numpy(detected_maps.copy()).float().cuda() / 255.0
            local_control = torch.stack([local_control for _ in range(num_samples)], dim=0)
            local_control = einops.rearrange(local_control, 'b h w c -> b c h w').clone()
            global_control = torch.from_numpy(content_emb.copy()).float().cuda().clone()
            global_control = torch.stack([global_control for _ in range(num_samples)], dim=0)

            if config.save_memory:
                 self.model.low_vram_shift(is_diffusing=False)

            uc_local_control = local_control
            uc_global_control = torch.zeros_like(global_control)
            cond = {"local_control": [local_control], "c_crossattn": [self.model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)], 'global_control': [global_control]}
            un_cond = {"local_control": [uc_local_control], "c_crossattn": [self.model.get_learned_conditioning([n_prompt] * num_samples)], 'global_control': [uc_global_control]}
            shape = (4, H // 8, W // 8)

            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=True)

            self.model.control_scales = [strength] * 13
            samples, _ = self.ddim_sampler.sample(ddim_steps, num_samples,
                                                        shape, cond, verbose=False, eta=eta,
                                                        unconditional_guidance_scale=scale,
                                                        unconditional_conditioning=un_cond, global_strength=global_strength)

            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=False)

            x_samples = self.model.decode_first_stage(samples)
            x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
            results = [x_samples[i] for i in range(num_samples)]

        return [results, detected_maps_list]




    def generate_panorama(self, save_path, layout, depth, semantic_image, prompt):
        
        content_image = None
        a_prompt = '' # 'best quality, extremely detailed'
        n_prompt = '' # 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'
        num_samples = 3 #
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


        src_max = np.max(depth)
        src_min = np.min(depth)
        new_depth = depth / (src_max - src_min)
        new_depth *= 255



        [results, detected_maps] = self.process(layout, depth, semantic_image, content_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, strength, scale, seed, eta, low_threshold, high_threshold, value_threshold, distance_threshold, alpha, global_strength)        

        with open(os.path.join(save_path, 'texture_prompt.txt'), "w") as f:
            f.write(prompt)
        for idx, result in enumerate(results):
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            results[idx] = result

        return results
        
