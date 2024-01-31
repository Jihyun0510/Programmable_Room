import os
import os.path as op
import json
import pdb
import clip
import torch
import numpy as np
from tqdm import tqdm
import time
import random
from PIL import Image
import argparse
import openai
from layoutGPT.utils import *


from transformers import GPT2TokenizerFast

# from parse_llm_output import parse_3D_layout
import json

openai.organization = ""
openai.api_key = "" 
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")


parser = argparse.ArgumentParser(prog='LayoutGPT for scene synthesis', description='Use GPTs to predict 3D layout for indoor scenes.')
parser.add_argument('--room', type=str, default='bedroom', choices=['bedroom','livingroom'])
parser.add_argument('--dataset_dir', type=str)
parser.add_argument('--gpt_type', type=str, default='gpt4', choices=['gpt3.5', 'gpt3.5-chat', 'gpt4'])
parser.add_argument('--icl_type', type=str, default='k-similar', choices=['fixed-random', 'k-similar'])
parser.add_argument('--base_output_dir', type=str, default='./llm_output/3D/')
parser.add_argument('--K', type=int, default=8)
parser.add_argument('--gpt_input_length_limit', type=int, default=7000)
parser.add_argument('--unit', type=str, choices=['px', 'm', ''], default='px')
parser.add_argument("--n_iter", type=int, default=1)
parser.add_argument("--test", action='store_true')
parser.add_argument('--verbose', default=False, action='store_true')
parser.add_argument("--suffix", type=str, default="")
parser.add_argument("--normalize", action='store_true')
parser.add_argument("--regular_floor_plan", action='store_true')
parser.add_argument("--temperature", type=float, default=0.7)
parser.add_argument("--floor", type=str)
args = parser.parse_args()

# GPT Type
gpt_name = {
    'gpt3.5': 'text-davinci-003',
    'gpt3.5-chat': 'gpt-3.5-turbo',
    'gpt4': 'gpt-4',
}


def load_room_boxes(prefix, id, stats, unit):
    data = np.load(op.join(prefix, id, 'boxes.npz'))
    x_c, y_c = data['floor_plan_centroid'][0], data['floor_plan_centroid'][2]
    x_offset  = min(data['floor_plan_vertices'][:,0])
    y_offset = min(data['floor_plan_vertices'][:,2])
    room_length = max(data['floor_plan_vertices'][:,0]) - min(data['floor_plan_vertices'][:,0])
    room_width = max(data['floor_plan_vertices'][:,2]) - min(data['floor_plan_vertices'][:,2])    
    vertices = np.stack((data['floor_plan_vertices'][:,0]-x_offset, data['floor_plan_vertices'][:,2]-y_offset), axis=1)
    vertices = np.asarray([list(nxy) for nxy in set(tuple(xy) for xy in vertices)])
    # print("room_length:",room_length,"room_width", room_width)  
    # normalize
    if args.normalize:
        norm = min(room_length, room_width)
        room_length, room_width = room_length/norm, room_width/norm
        vertices /= norm
        if unit in ['px', '']:
            scale_factor = 256
            room_length, room_width = int(room_length*scale_factor), int(room_width*scale_factor)

    vertices = [f'({v[0]:.2f}, {v[1]:.2f})' for v in vertices]

    if unit in ['px', '']:
        condition = f"Condition:\n"
        if args.room == 'livingroom':
            if 'dining' in id.lower():
                condition += f"Room Type: living room & dining room\n"
            else:
                condition += f"Room Type: living room\n"
        else:
            condition += f"Room Type: {args.room}\n"
        condition += f"Room Size: max length {room_length}{unit}, max width {room_width}{unit}\n"
    else:
        condition = f"Condition:\n" \
                    f"Room Type: {args.room}\n" \
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
        if args.normalize:
            length, width, height = length/norm, width/norm, height/norm
            dx, dy, dz = dx/norm, dy/norm, dz/norm
            if unit in ['px', '']:
                length, width, height = int(length*scale_factor), int(width*scale_factor), int(height*scale_factor)
                dx, dy, dz = int(dx*scale_factor), int(dy*scale_factor), int(dz*scale_factor)

        if unit in ['px', '']:
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


def load_set(prefix, ids, stats, unit):
    id2prompt = {}
    meta_data = {}
    for id in tqdm(ids):
        condition, layout, data = load_room_boxes(prefix, id, stats, unit)
        id2prompt[id] = [condition, layout]
        meta_data[id] = data
    return id2prompt, meta_data


def load_features(meta_data, floor_plan=False):
    features = {}
    for id, data in meta_data.items():
        if floor_plan:
            features[id] = np.asarray(Image.fromarray(data['room_layout'].squeeze()).resize((64,64)))
        else:
            room_length = max(data['floor_plan_vertices'][:,0]) - min(data['floor_plan_vertices'][:,0])
            room_width = max(data['floor_plan_vertices'][:,2]) - min(data['floor_plan_vertices'][:,2])  
            features[id] = np.asarray([room_length, room_width])

    
    return features


def get_closest_room(train_features, val_feature):
    '''
    train_features
    '''
    distances = [[id, ((feat-val_feature)**2).mean()] for id, feat in train_features.items()]
    distances = sorted(distances, key=lambda x: x[1])
    sorted_ids, _ = zip(*distances)
    return sorted_ids


def create_prompt(sample):
    return sample[0] + sample[1] + "\n\n"

def convert_to_list_format(input_string):
    # Split input string into individual furniture entries based on line breaks
    furniture_entries = input_string.strip().split('\n')
    
    # Initialize an empty list to store furniture objects
    furniture_objects = []
    
    # Iterate through furniture entries and extract attributes
    for entry in furniture_entries:
        # Split entry based on specific delimiters to extract attributes
        parts = entry.split('{')
        # print(parts)
        furniture_name = parts[0].strip()
        # print(furniture_name)
        attributes = {}
        
        for attribute in parts[1].rstrip(";}").split(";"):
            # print(attribute)
            key, value = attribute.split(':')
            if key.strip() == "orientation":
                attributes[key.strip()] = float(value.strip(';').strip().replace('degrees', '').strip())
            else:
                attributes[key.strip()] = float(value.strip(';').strip().replace('px', '').strip())
        furniture_objects.append([furniture_name, attributes])
    
    return furniture_objects

def json_to_css(data):
   
    css_output = ""
    for item in data:
        furniture_name, attributes = item
        css_output += f"{furniture_name} {{length: {attributes['length']}px; width: {attributes['width']}px; height: {attributes['height']}px; left: {attributes['left']}px; top: {attributes['top']}px; depth: {attributes['depth']}px;orientation: {attributes['orientation']} degrees;}}\n"
    
    return css_output


def form_prompt_for_chatgpt(data, furniture, rl, rw, top_k, stats, supporting_examples,
                            train_features=None, val_feature=None):
    norm = min(rl, rw)
    room_length, room_width = rl/norm, rw/norm

    scale_factor = 256
    room_length, room_width = int(room_length*scale_factor), int(room_width*scale_factor)

    message_list = []
    unit_name = 'pixel' if args.unit in ['px', ''] else 'meters'

    rtn_prompt = 'You are a 3D indoor scene designer. \nInstruction: add an object to a 3D layout of an indoor scene.' \
                'The added object should follow the CSS style, where it starts with the furniture category ' \
                'and is followed by the 3D size, orientation and absolute position. ' \
                "Formally, the result should follow the template: \n" \
                f"FURNITURE {{length: ?{args.unit}: width: ?{args.unit}; height: ?{args.unit}; orientation: ? degrees; left: ?{args.unit}; top: ?{args.unit}; depth: ?{args.unit};}}\n" \
                f'All values are in {unit_name} but the orientation angle is in degrees.\n\n'

    message_list.append({'role': 'system', 'content': rtn_prompt})
    last_example = f'Add a {furniture} to a 3D scene. \nRoom size: max length {room_length}, max width {room_width}. \n3D scene layout: {json_to_css(data)}. \n'

    if args.icl_type == 'k-similar':
        assert train_features is not None
        sorted_ids = get_closest_room(train_features, val_feature)
        supporting_examples = [supporting_examples[id] for id in sorted_ids[:top_k]]

    for i, supporting_example in enumerate(supporting_examples[:top_k]):

        input_string = supporting_example[1].lstrip("Layout:\n")
        # Split the input string into lines
        lines = input_string.rstrip("\n").split('\n')

        # Filter out lines containing 'pendant_lamp'
        filtered_lines = [line for line in lines if not line.startswith('pendant_lamp')]

        # Join the filtered lines back into a string
        supporting_example[1] = '\n'.join(filtered_lines)


        room_dimensions = supporting_example[0].split(': max length ')[1].split('px, max width ')
        room_length = int(room_dimensions[0])
        room_width = int(room_dimensions[1].split('px')[0])

        # furniture_list = supporting_example[1].split('\n')[1:-1]
        furniture_list = supporting_example[1].split('\n')
        random_furniture_index = random.randint(0, len(furniture_list) - 1)
        # while "pendant" in furniture_list[random_furniture_index]:
        #     random_furniture_index = random.randint(0, len(furniture_list) - 1)

        selected_furniture = furniture_list.pop(random_furniture_index)

        first_statement = "Add a {} to a 3d scene. \nRoom size: max length {}px, max width {}px. \n3D scene layout: {}. \n".format(
            selected_furniture.split(' ')[0],
            room_length,
            room_width,
            '\n'.join(furniture_list))

        # Construct the second statement with the randomly selected furniture added at the end
        second_statement = '\n'.join(furniture_list) + f"\n{selected_furniture}\n"

        print("first statement")
        print(first_statement)
        print("second statement")
        print(second_statement)
        current_messages = [
            {'role': 'user', 'content': first_statement},
            {'role': 'assistant', 'content': second_statement},
        ]
        message_list = message_list + current_messages
    
    # Concatenate prompts for gpt4
    message_list.append({'role': 'user', 'content': last_example})

    return message_list
###############################################################edited#############################################################
def calculate_centroid(attr):
    x_centroid = attr['left']
    y_centroid = attr['top']
    return x_centroid, y_centroid

def calculate_distance(furn0_feat, fur1_feat):
    return ((furn0_feat[0] - fur1_feat[0])**2 + (furn0_feat[1] - fur1_feat[1])**2) ** 0.5

def parse_attributes_from_example(example, furniture1):
    attributes = {}
    _, line = example
    parts = line.lstrip("Layout:\n").rstrip(";}\n").split("}\n")
    for part in parts:
        furniture, attrs = part.split(" {")
        if furniture == furniture1:
            attrs = attrs.rstrip(";").split(";")
            for attr in attrs:
                key, value = attr.split(": ")
                key, value = key.strip(), value.strip()
                if key=="orientation":
                    attributes[key] = float(value[:-7])  # remove 'degrees' and convert to float for numerical attributes
                else:
                    attributes[key] = float(value[:-2])  # remove 'px' and convert to float for numerical attributes
            return attributes

def replace_fur(data, furniture, room_length, room_width, top_k, supporting_examples, train_features, val_feature):
    furn0_attributes = None
    for d in data:
        if d[0] == furniture[0]: 
            furn0_attributes = d[1]
            break

    furn0_feat = calculate_centroid(furn0_attributes)
    
    closest_example = None
    min_distance = float('inf')
    
    sorted_ids = get_closest_room(train_features, val_feature)
    supporting_examples = [supporting_examples[id] for id in sorted_ids]
    k = 0
    
    for supporting_example in supporting_examples:
        if furniture[1] in supporting_example[1] and k < top_k:

            example_attributes = parse_attributes_from_example(supporting_example, furniture[1])
            fur1_feat = calculate_centroid(example_attributes)
            distance = calculate_distance(furn0_feat, fur1_feat)
            
            # Check if the replacement item fits within the room dimensions
            if distance < min_distance:
                # update
                closest_example = example_attributes
                min_distance = distance
            
            k += 1
    
    if closest_example:
        # Remove furniture[0] from data
        updated_data = [item for item in data if item[0] != furniture[0]]

        # Add furniture[1] with updated attributes to data
        new_furniture = [furniture[1], {
            "depth": closest_example["depth"],
            "height": closest_example["height"],
            "left": furn0_feat[0],
            "length": closest_example["length"],
            "orientation": closest_example["orientation"],
            "top": furn0_feat[1],
            "width": closest_example["width"]
        }]

        updated_data.append(new_furniture)
        
        print("Closest example found:", closest_example)
        print("Updated data after replacing", furniture[0], "with closest example.")
    else:
        print("No suitable examples found.")
        updated_data = data  # If no suitable example is found, return the original data.
    
    return updated_data

def _main(args):
    dataset_prefix = f"{args.dataset_dir}"

    with open("./ProgrammableRoom/layoutGPT/dataset/3D/bedroom_splits.json", "r") as file:
        splits = json.load(file)
    
    with open("./ProgrammableRoom/layoutGPT/ATISS/config/dataset_stats.txt", "r") as file:
        stats = json.load(file)   

    if args.regular_floor_plan:
        args.suffix += '_regular'


    # load train examples
    train_ids = splits['rect_train'] if args.regular_floor_plan else splits['train']
    train_data, meta_train_data = load_set(dataset_prefix, train_ids, stats, args.unit)


    if args.icl_type == 'fixed-random':
        # load fixed supporting examples
        all_supporting_examples = list(train_data.values())
        supporting_examples = all_supporting_examples[:args.K]
        train_features = None

    elif args.icl_type == 'k-similar':
        supporting_examples = train_data
        train_features = load_features(meta_train_data)
    
    # GPT-3 prediction process
    args.gpt_name = gpt_name[args.gpt_type]
    top_k = args.K

    # Read the original JSON file
    output_directory, iteration = args.save_directory[:-1], int(args.save_directory[-1])
    with open(os.path.join(output_directory, "{}.furniture.json".format(iteration-1)), 'r') as file:
        data = json.load(file)
        
        if order.lower() in ["remove", "delete"]:
            # Remove "furniture" and its attributes from the data
            updated_data = [item for item in data if item[0] != furniture]


        elif order.lower() == 'replace':
            updated_data = replace_fur(data, furniture, 
                                       room_length,
                                        room_width,
                                        top_k=top_k,
                                        supporting_examples=supporting_examples,
                                        train_features=train_features,
                                        val_feature= np.asarray([room_length, room_width]))

        elif order.lower() == "add":

            prompt_for_gpt3 = form_prompt_for_chatgpt(
                data,
                furniture,
                rl = room_length,
                rw = room_width,
                top_k=top_k,
                stats=stats,
                supporting_examples=supporting_examples,
                train_features=train_features,
                val_feature=np.asarray([room_length, room_width]))
        
            # print(prompt_for_gpt3)
            response = openai.ChatCompletion.create(
                model=args.gpt_name,
                messages=prompt_for_gpt3,
                temperature=0.7,
                max_tokens=1024,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                stop="Instruction:",
                n=args.n_iter,
            )

            response = response["choices"][0]["message"]["content"]
            print(response)
            updated_data = convert_to_list_format(response)

        
        else:
            raise ValueError("Invalid order. Order must be 'replace', 'remove', 'delete', or 'add'.")       
        
        # Save the edited content to the new JSON file
        with open(os.path.join(args.save_directory, "furniture.json", 'w')) as new_file:
            json.dump(updated_data, new_file, indent=4)


if __name__ == '__main__':
    _main(args)