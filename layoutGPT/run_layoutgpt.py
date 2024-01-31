import os
import os.path as op
import json
import clip
import torch
import numpy as np
from tqdm import tqdm
import time
from PIL import Image
import openai
import argparse
import sys

from transformers import GPT2TokenizerFast
from layoutGPT.utils import *
from layoutGPT.parse_llm_output import parse_3D_layout
from layoutGPT.find_largest_rectangle import find_rectangle


class LayoutGPTSceneSynthesis:
    def __init__(self, open_ai_key, floor, text, dataset_dir, icl_type='k-similar', K=8, room='bedroom', llm_type='gpt4', unit='px', regular_floor_plan=True):
        self.open_ai_key = open_ai_key
        self.floor = floor
        self.text = text
        self.dataset_dir = dataset_dir
        self.icl_type = icl_type
        self.K = K

        if "livingroom" in text:
            self.room = "livingroom"
        else:
            self.room = "bedroom"

        print("ROOM TYPE:", self.room)

        self.gpt_type = llm_type
        self.unit = unit
        self.regular_floor_plan = regular_floor_plan
        self.setup_openai()
        self.gpt_name = {
    'gpt3.5': 'text-davinci-003',
    'gpt3.5-chat': 'gpt-3.5-turbo',
    'gpt4': 'gpt-4',
}
    def setup_openai(self):
        openai.organization = ""
        openai.api_key = self.open_ai_key
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    def form_prompt_for_chatgpt_ori(self, text_input, rl,rw, top_k, stats, supporting_examples,
                                train_features=None, val_feature=None):
        
        norm = min(rl, rw)
        room_length, room_width = rl/norm, rw/norm

        scale_factor = 256
        room_length, room_width = int(room_length*scale_factor), int(room_width*scale_factor)

        message_list = []
        unit_name = 'pixel' if self.unit in ['px', ''] else 'meters'
        class_freq = [f"{obj}: {round(stats['class_frequencies'][obj], 4)}" for obj in stats['object_types']]
        rtn_prompt = 'You are a 3D indoor scene designer. \nInstruction: synthesize the 3D layout of an indoor scene. ' \
                    'The generated 3D layout should follow the CSS style, where each line starts with the furniture category ' \
                    'and is followed by the 3D size, orientation and absolute position. ' \
                    "Formally, each line should follow the template: \n" \
                    f"FURNITURE {{length: ?{self.unit}: width: ?{self.unit}; height: ?{self.unit}; orientation: ? degrees; left: ?{self.unit}; top: ?{self.unit}; depth: ?{self.unit};}}\n" \
                    f'All values are in {unit_name} but the orientation angle is in degrees.\n\n' \
                    f"Available furnitures: {', '.join(stats['object_types'])} \n" \
                    f"Overall furniture frequencies: ({'; '.join(class_freq)})\n\n"

        message_list.append({'role': 'system', 'content': rtn_prompt})
        last_example = 'Room type: {}, Room size: {} x {}, Layout:\n'.format(self.room, room_width, room_length)


        if self.icl_type == 'k-similar':
            assert train_features is not None
            sorted_ids = self.get_closest_room(train_features, val_feature)
            supporting_examples = [supporting_examples[id] for id in sorted_ids[:top_k]]


        # loop through the related supporting examples, check if the prompt length exceed limit
        for supporting_example in supporting_examples[:top_k]:

            current_messages = [
                {'role': 'user', 'content': supporting_example[0]+"Layout:\n"},
                {'role': 'assistant', 'content': supporting_example[1].lstrip("Layout:\n")},
            ]
            message_list = message_list + current_messages
        
        # concatename prompts for gpt4
        message_list.append({'role': 'user', 'content': last_example })

        return message_list


    def load_room_boxes(self, prefix, id, stats, unit):
        data = np.load(op.join(prefix, id, 'boxes.npz'))
        x_c, y_c = data['floor_plan_centroid'][0], data['floor_plan_centroid'][2]
        x_offset  = min(data['floor_plan_vertices'][:,0])
        y_offset = min(data['floor_plan_vertices'][:,2])
        room_length = max(data['floor_plan_vertices'][:,0]) - min(data['floor_plan_vertices'][:,0])
        room_width = max(data['floor_plan_vertices'][:,2]) - min(data['floor_plan_vertices'][:,2])    
        vertices = np.stack((data['floor_plan_vertices'][:,0]-x_offset, data['floor_plan_vertices'][:,2]-y_offset), axis=1)
        vertices = np.asarray([list(nxy) for nxy in set(tuple(xy) for xy in vertices)])
 
        norm = min(room_length, room_width)
        room_length, room_width = room_length/norm, room_width/norm
        vertices /= norm
        if self.unit in ['px', '']:
            scale_factor = 256
            room_length, room_width = int(room_length*scale_factor), int(room_width*scale_factor)

        vertices = [f'({v[0]:.2f}, {v[1]:.2f})' for v in vertices]

        if self.unit in ['px', '']:
            condition = f"Condition:\n"
            if self.room == 'livingroom':
                if 'dining' in id.lower():
                    condition += f"Room Type: living room & dining room\n"
                else:
                    condition += f"Room Type: living room\n"
            else:
                condition += f"Room Type: {self.room}\n"
            condition += f"Room Size: max length {room_length}{unit}, max width {room_width}{unit}\n"
        else:
            condition = f"Condition:\n" \
                        f"Room Type: {self.room}\n" \
                        f"Room Size: max length {room_length:.2f}{unit}, max width {room_width:.2f}{unit}\n"

        layout = 'Layout:\n'
        for label, size, angle, loc in zip(data['class_labels'], data['sizes'], data['angles'], data['translations']):
            label_idx = np.where(label)[0][0]
            if label_idx >= len(stats['object_types']): # NOTE:
                continue
            cat = stats['object_types'][label_idx]
            
            length, height, width = size # NOTE: half the actual size
            length, height, width = length*2, height*2, width*2
            orientation = round(angle[0] / 3.1415926 * 180)
            dx,dz,dy = loc # NOTE: center point
            dx = dx+x_c-x_offset
            dy = dy+y_c-y_offset

            # normalize
            length, width, height = length/norm, width/norm, height/norm
            dx, dy, dz = dx/norm, dy/norm, dz/norm
            if self.unit in ['px', '']:
                length, width, height = int(length*scale_factor), int(width*scale_factor), int(height*scale_factor)
                dx, dy, dz = int(dx*scale_factor), int(dy*scale_factor), int(dz*scale_factor)

            if self.unit in ['px', '']:
                layout += f"{cat} {{length: {length}{unit}; " \
                                    f"width: {width}{unit}; " \
                                    f"height: {height}{unit}; " \
                                    f"left: {dx}{unit}; " \
                                    f"top: {dy}{unit}; " \
                                    f"depth: {dz}{unit};" \
                                    f"orientation: {orientation} degrees;}}\n"                                
            else:
                layout += f"{cat} {{length: {length:.2f}{unit}; " \
                                    f"height: {height:.2f}{unit}; " \
                                    f"width: {width:.2f}{unit}; " \
                                    f"orientation: {orientation} degrees; " \
                                    f"left: {dx:.2f}{unit}; " \
                                    f"top: {dy:.2f}{unit}; " \
                                    f"depth: {dz:.2f}{unit};}}\n" 

        return condition, layout, dict(data)
    
    def load_set(self, prefix, ids, stats, unit):
        id2prompt = {}
        meta_data = {}
        for id in tqdm(ids):
            condition, layout, data = self.load_room_boxes(prefix, id, stats, unit)
            id2prompt[id] = [condition, layout]
            meta_data[id] = data
        return id2prompt, meta_data

    def load_features(self, meta_data, floor_plan=False):
        features = {}
        for id, data in meta_data.items():
            if floor_plan:
                features[id] = np.asarray(Image.fromarray(data['room_layout'].squeeze()).resize((64,64)))
            else:
                room_length = max(data['floor_plan_vertices'][:,0]) - min(data['floor_plan_vertices'][:,0])
                room_width = max(data['floor_plan_vertices'][:,2]) - min(data['floor_plan_vertices'][:,2])  
                features[id] = np.asarray([room_length, room_width])
        
        return features

    def get_closest_room(self, train_features, val_feature):
        '''
        train_features
        '''
        distances = [[id, ((feat-val_feature)**2).mean()] for id, feat in train_features.items()]
        distances = sorted(distances, key=lambda x: x[1])
        sorted_ids, _ = zip(*distances)
        return sorted_ids

    def create_prompt(self, sample):
        return sample[0] + sample[1] + "\n\n"

    def run_scene_synthesis(self):
        # dataset_prefix = f"{self.dataset_dir}"
        dataset_prefix = "/ProgrammableRoom/layoutGPT/original_result"

        with open("/ProgrammableRoom/layoutGPT/dataset/3D/{}_splits.json".format(self.room), "r") as file:
            splits = json.load(file)
        
        with open("/ProgrammableRoom/layoutGPT/ATISS/config/{}_dataset_stats.txt".format(self.room), "r") as file:
            stats = json.load(file) 

        if self.regular_floor_plan:
            suffix = ""
            suffix += '_regular'


        # load train examples
        train_ids = splits['rect_train'] if self.regular_floor_plan else splits['train']
        train_data, meta_train_data = self.load_set(dataset_prefix, train_ids, stats, self.unit)


        if self.icl_type == 'fixed-random':
            # load fixed supporting examples
            all_supporting_examples = list(train_data.values())
            supporting_examples = all_supporting_examples[:self.K]
            train_features = None

        elif self.icl_type == 'k-similar':
            supporting_examples = train_data
            train_features = self.load_features(meta_train_data)
        
        # GPT-3 prediction process
        gpt_name = self.gpt_name[self.gpt_type]
        top_k = self.K

        rectangle = find_rectangle(self.floor)
        
        room_length = max(rectangle, key=lambda x: x[0])[0] - min(rectangle, key=lambda x: x[0])[0]
        room_width = max(rectangle, key=lambda x: x[1])[1] - min(rectangle, key=lambda x: x[1])[1]
        val_features = np.asarray([room_length, room_width])
        

        # # Calculate centroid coordinates
        # centroid_x = sum(x for x, y in rectangle) / 4
        # centroid_y = sum(y for x, y in rectangle) / 4

        prompt_for_gpt3 = self.form_prompt_for_chatgpt_ori(
        # text_input=val_example,
        text_input=self.room,
        rl = room_length,
        rw = room_width,
        top_k=top_k,
        stats=stats,
        supporting_examples=supporting_examples,
        train_features=train_features,
        val_feature=val_features)

        response = openai.ChatCompletion.create(
            model=gpt_name,
            messages=prompt_for_gpt3,
            temperature=0.7,
            max_tokens=1024,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0)

        response = response["choices"][0]["message"]["content"]
        responses = response.split('\n')

        result = []
        for r in responses:
            if r == '':
                continue
            print("Response:", r)
            l, a = r.split("{")
            a = a.strip().strip("}").strip().strip(";").split(";")
            a = [b.strip().strip(";").rstrip("degrees").strip() for b in a]
            l = l.strip()
            a = [b.split(":") for b in a]
            a = {k.strip():float(v.lstrip().rstrip("px")) for k, v in a}
            result.append([l,a])

        write_json(op.join(self.dataset_dir, "furniture.json"), result) #edited
        return result
