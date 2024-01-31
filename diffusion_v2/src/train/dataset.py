import os
import random
import cv2
import numpy as np

from torch.utils.data import Dataset

from util import *

from diffusion_v2.utils import panostretch
from shapely.geometry import LineString
from PIL import Image

class UniDataset(Dataset):
    def __init__(self,
                 anno_path,
                 image_dir,
                 condition_root,
                 local_type_list,
                 global_type_list,
                 resolution,
                 drop_txt_prob,
                 keep_all_cond_prob,
                 drop_all_cond_prob,
                 drop_each_cond_prob):
        
        file_ids, depth_ids, layout_ids, semantic_ids, coord_ids, content_ids, self.annos = read_anno(anno_path)
        self.coord_paths = coord_ids
        self.image_paths = [file_id for file_id in file_ids]
        self.local_paths = {}
        for local_type in local_type_list:
            if local_type.startswith('depth'):
                self.local_paths[local_type] = [depth_id for depth_id in depth_ids]
            elif local_type.startswith('layout'):
                self.local_paths[local_type] = [layout_id for layout_id in layout_ids]
            elif local_type.startswith('semantic'):
                self.local_paths[local_type] = [semantic_id for semantic_id in semantic_ids]
        self.global_paths = {}
        for global_type in global_type_list:
            self.global_paths[global_type] = [content_id for content_id in content_ids]
        
        self.local_type_list = local_type_list
        self.global_type_list = global_type_list
        self.resolution = resolution
        self.drop_txt_prob = drop_txt_prob
        self.keep_all_cond_prob = keep_all_cond_prob
        self.drop_all_cond_prob = drop_all_cond_prob
        self.drop_each_cond_prob = drop_each_cond_prob
    
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        data_id = image_path.split('/')[-1][:-10]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.resolution, self.resolution))
        image = (image.astype(np.float32) / 127.5) - 1.0
        
        image_ori = cv2.imread(image_path)
        image_ori = cv2.cvtColor(image_ori, cv2.COLOR_BGR2RGB)
        image_ori = (image_ori.astype(np.float32) / 127.5) - 1.0

        anno = self.annos[index]
        local_files = []
        for local_type in self.local_type_list:
            local_files.append(self.local_paths[local_type][index])
        global_files = []
        for global_type in self.global_type_list:
            global_files.append(self.global_paths[global_type][index])

        local_conditions = []
        for local_file in local_files:
            condition = cv2.imread(local_file)
            condition = cv2.cvtColor(condition, cv2.COLOR_BGR2RGB)
            condition = cv2.resize(condition, (self.resolution, self.resolution))
            condition = condition.astype(np.float32) / 255.0
            local_conditions.append(condition)
        global_conditions = []
        for global_file in global_files:
            condition = np.load(global_file)
            global_conditions.append(condition)
        
        if False: # dropout
            if random.random() < self.drop_txt_prob:
                anno = ''
            local_conditions = keep_and_drop(local_conditions, self.keep_all_cond_prob, self.drop_all_cond_prob, self.drop_each_cond_prob)
            global_conditions = keep_and_drop(global_conditions, self.keep_all_cond_prob, self.drop_all_cond_prob, self.drop_each_cond_prob)
        
        if len(local_conditions) != 0:
            local_conditions = np.concatenate(local_conditions, axis=2)
        if len(global_conditions) != 0:
            global_conditions = np.concatenate(global_conditions)

        # Read ground truth corners
        img = np.array(Image.open(image_path), np.float32)[..., :3] / 255.
        H, W = img.shape[:2]
        with open(self.coord_paths[index]) as f:
            cor = np.array([line.strip().split() for line in f if line.strip()], np.float32)

            # Corner with minimum x should at the beginning
            cor = np.roll(cor[:, :2], -2 * np.argmin(cor[::2, 0]), 0)

            # Detect occlusion
            occlusion = find_occlusion(cor[::2].copy()).repeat(2)
            assert (np.abs(cor[0::2, 0] - cor[1::2, 0]) > W/100).sum() == 0, image_path
            assert (cor[0::2, 1] > cor[1::2, 1]).sum() == 0, image_path

        # Stretch augmentation
        if True:
            max_stretch = 2.0
            xmin, ymin, xmax, ymax = cor2xybound(cor)
            kx = np.random.uniform(1.0, max_stretch)
            ky = np.random.uniform(1.0, max_stretch)
            if np.random.randint(2) == 0:
                kx = max(1 / kx, min(0.5 / xmin, 1.0))
            else:
                kx = min(kx, max(10.0 / xmax, 1.0))
            if np.random.randint(2) == 0:
                ky = max(1 / ky, min(0.5 / ymin, 1.0))
            else:
                ky = min(ky, max(10.0 / ymax, 1.0))
            img, cor = panostretch.pano_stretch(img, cor, kx, ky)

        # Random flip
        if True and np.random.randint(2) == 0:
            cor[:, 0] = img.shape[1] - 1 - cor[:, 0]

        # Random horizontal rotate
        if True:
            dx = np.random.randint(img.shape[1])
            cor[:, 0] = (cor[:, 0] + dx) % img.shape[1]

        # Prepare 1d wall-wall probability
        corx = cor[~occlusion, 0]
        cal_A = corx.reshape(1, -1)
        cal_B = np.arange(img.shape[1]).reshape(-1, 1)

        dist_o = np.min(np.abs(cal_A - cal_B), axis=1)
        dist_r = np.min(np.abs(cal_A - (cal_B + img.shape[1])), axis=1)
        dist_l = np.min(np.abs(cal_A - (cal_B - img.shape[1])), axis=1)
        dist = np.min([dist_o, dist_r, dist_l], 0)
        p_base = 0.96
        y_cor = (p_base ** dist).reshape(1, -1)
        
        return dict(jpg=image, jpg_ori=image_ori, txt=anno, local_conditions=local_conditions, global_conditions=global_conditions, coord=y_cor, data_id=data_id)
        
    def __len__(self):
        return len(self.annos)

def find_occlusion(coor):
    u = panostretch.coorx2u(coor[:, 0])
    v = panostretch.coory2v(coor[:, 1])
    x, y = panostretch.uv2xy(u, v, z=-50)
    occlusion = []
    for i in range(len(x)):
        raycast = LineString([(0, 0), (x[i], y[i])])
        other_layout = []
        for j in range(i+1, len(x)):
            other_layout.append((x[j], y[j]))
        for j in range(0, i):
            other_layout.append((x[j], y[j]))
        other_layout = LineString(other_layout)
        occlusion.append(raycast.intersects(other_layout))
    return np.array(occlusion)


def cor2xybound(cor):
    ''' Helper function to clip max/min stretch factor '''
    corU = cor[0::2]
    corB = cor[1::2]
    zU = -50
    u = panostretch.coorx2u(corU[:, 0])
    vU = panostretch.coory2v(corU[:, 1])
    vB = panostretch.coory2v(corB[:, 1])

    x, y = panostretch.uv2xy(u, vU, z=zU)
    c = np.sqrt(x**2 + y**2)
    zB = c * np.tan(vB)
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    S = 3 / abs(zB.mean() - zU)
    dx = [abs(xmin * S), abs(xmax * S)]
    dy = [abs(ymin * S), abs(ymax * S)]

    return min(dx), min(dy), max(dx), max(dy)


def cor_2_1d(cor, H, W):
    bon_ceil_x, bon_ceil_y = [], []
    bon_floor_x, bon_floor_y = [], []
    n_cor = len(cor)
    for i in range(n_cor // 2):
        xys = panostretch.pano_connect_points(cor[i*2],
                                              cor[(i*2+2) % n_cor],
                                              z=-50, w=W, h=H)
        bon_ceil_x.extend(xys[:, 0])
        bon_ceil_y.extend(xys[:, 1])
    for i in range(n_cor // 2):
        xys = panostretch.pano_connect_points(cor[i*2+1],
                                              cor[(i*2+3) % n_cor],
                                              z=50, w=W, h=H)
        bon_floor_x.extend(xys[:, 0])
        bon_floor_y.extend(xys[:, 1])
    bon_ceil_x, bon_ceil_y = sort_xy_filter_unique(bon_ceil_x, bon_ceil_y, y_small_first=True)
    bon_floor_x, bon_floor_y = sort_xy_filter_unique(bon_floor_x, bon_floor_y, y_small_first=False)
    bon = np.zeros((2, W))
    bon[0] = np.interp(np.arange(W), bon_ceil_x, bon_ceil_y, period=W)
    bon[1] = np.interp(np.arange(W), bon_floor_x, bon_floor_y, period=W)
    bon = ((bon + 0.5) / H - 0.5) * np.pi
    return bon


def sort_xy_filter_unique(xs, ys, y_small_first=True):
    xs, ys = np.array(xs), np.array(ys)
    idx_sort = np.argsort(xs + ys / ys.max() * (int(y_small_first)*2-1))
    xs, ys = xs[idx_sort], ys[idx_sort]
    _, idx_unique = np.unique(xs, return_index=True)
    xs, ys = xs[idx_unique], ys[idx_unique]
    assert np.all(np.diff(xs) > 0)
    return xs, ys