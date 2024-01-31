import random

import numpy as np


def read_anno(anno_path):
    fi = open(anno_path)
    lines = fi.readlines()
    fi.close()
    file_ids, depth_ids, layout_ids, semantic_ids, coord_ids, content_ids, annos = [], [], [], [], [], [], []
    for line in lines:
        line_new = eval(line.strip())
        id = line_new["target"]
        id_depth = line_new["depth"]
        id_layout = line_new["layout"]
        id_semantic = line_new["semantic"]
        id_coord = line_new["coord"]
        id_content = line_new["content"]
        txt = line_new["prompt"]
        file_ids.append(id)
        depth_ids.append(id_depth)
        layout_ids.append(id_layout)
        semantic_ids.append(id_semantic)
        coord_ids.append(id_coord)
        content_ids.append(id_content)
        annos.append(txt)
    return file_ids, depth_ids, layout_ids, semantic_ids, coord_ids, content_ids, annos
    # return file_ids, depth_ids, layout_ids, semantic_ids, coord_ids, annos


def keep_and_drop(conditions, keep_all_prob, drop_all_prob, drop_each_prob):
    results = []
    seed = random.random()
    if seed < keep_all_prob:
        results = conditions
    elif seed < keep_all_prob + drop_all_prob:
        for condition in conditions:
            results.append(np.zeros(condition.shape))
    else:
        for i in range(len(conditions)):
            if random.random() < drop_each_prob[i]:
                results.append(np.zeros(conditions[i].shape))
            else:
                results.append(conditions[i])
    return results