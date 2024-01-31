import openai
import json
import os
import numpy as np
import cv2
import open3d as o3d
from PIL import Image
import re
import tifffile as tiff
import shutil
from engine.boundary import draw_boundaries
from engine.conversion import final_uv
from engine.utils import estimate_depth, fold, fold2
from engine.edit_furniture import AddFurniture, ReplaceFurniture
from diffusion_v2.diffusion_process import PanoramaGenerator
from engine.utils import find_rectangle
from layoutGPT.run_layoutgpt import LayoutGPTSceneSynthesis
from Structured3D.segmentation import visualize_panorama_single
from engine.conversion import xyz2uv
from Structured3D.misc.panorama import draw



# from transformers import GPT2TokenizerFast
# tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

class ProgramInterpreter:
    def __init__(self, openai_api_key, gpt_name='gpt-4'):
        openai.api_key = openai_api_key
        self.gpt_name = gpt_name

    def form_prompt_for_chatgpt(self, text_input):
        message_list = []

        prompt = "You are a programmer. Your task is to code a program for the client's instruction step by step. " \
                    "The client wants to generate or edit a room. \n" \
                        "Generate a program for the client's instruction step by step.\n"\
                        "Currently, for EditFurniture(), only 'Remove', 'Add', and 'Replace'are available.\n"\
                        "The input texts for the 'prompt' of GenSemantic() and GenTexture() should include the number of the windows and doors in numerals if the client mentions them.\n"\
                        "If the client wants to generate an empty room, do not generate furniture using GenFurniture(). In this case, the output of GenEmptyRoom() becomes the 'RESULT'.\n"

        supporting_examples = [
                                 ["INSTRUCTION: Create a living room whose size is 4mx5mx3m.",
                                'SHAPE = GenShape(prompt = "A livingroom whose size is 4mx5mx3m")\n'\
                                'EMPTY = GenEmptyRoom(coordinates = SHAPE, texture = None)\n'\
                                'FURNITURE = GenFurniture(coordinates = SHAPE, roomtype = "livingroom")\n'
                                'RESULT = Merge(emptyroom = EMPTY, furniture = FURNITURE)'],

                                ["INSTRUCTION: Increase the height of the room.",
                                'SHAPE, LAYOUT, DEPTH, SEMANTIC, TEXTURE, TEXT, EMPTY, FURNITURE = LoadRoom()\n'\
                                'SHAPE0 = EditShape(load = SHAPE, prompt="Increase the height of the room.")\n'\
                                'LAYOUT0 = EditLayout(load = LAYOUT, coordinates = SHAPE0)\n'\
                                'DEPTH0 = EditDepth(load = DEPTH, coordinates = SHAPE0)\n'\
                                'SEMANTIC0 = EditSemantic(load = SEMANTIC, coordinates = SHAPE0, prompt = TEXT)\n'\
                                'TEXTURE0 = EditTexture(coordinates = SHAPE0, layout = LAYOUT0, depth = DEPTH0, semantic = SEMANTIC0, prompt = TEXT)\n'\
                                'EMPTY0 = EditEmptyRoom(load = EMPTY, coordinates = SHAPE0, texture = TEXTURE0)\n'\
                                'RESULT = Merge(EMPTY0, furniture = FURNITURE)'],

                                ["INSTRUCTION: Increase the size of the room by scale 2.",
                                'SHAPE, LAYOUT, DEPTH, SEMANTIC, TEXTURE, TEXT, EMPTY, FURNITURE = LoadRoom()\n'\
                                'SHAPE0 = EditShape(load = SHAPE, prompt="Increase the size of the room by scale 2.")\n'\
                                'LAYOUT0 = EditLayout(load = LAYOUT, coordinates = SHAPE0)\n'\
                                'DEPTH0 = EditDepth(load = DEPTH, coordinates = SHAPE0)\n'\
                                'SEMANTIC0 = EditSemantic(load = SEMANTIC, coordinates = SHAPE0, prompt = TEXT)\n'\
                                'TEXTURE0 = EditTexture(coordinates = SHAPE0, layout = LAYOUT0, depth = DEPTH0, semantic = SEMANTIC0, prompt = TEXT)\n'\
                                'EMPTY0 = EditEmptyRoom(load = EMPTY, coordinates = SHAPE0, texture = TEXTURE0)\n'\
                                'RESULT = Merge(EMPTY0, furniture = FURNITURE)'],

                                ["INSTRUCTION: I want to replace the sofa with a chair.",
                                'SHAPE, LAYOUT, DEPTH, SEMANTIC, TEXTURE, TEXT, EMPTY, FURNITURE = LoadRoom()\n'\
                                'FURNITURE0 = EditFurniture(load = FURNITURE, coordinates = SHAPE, replace = "sofa -> chair")\n'\
                                'RESULT = Merge(emptyroom = EMPTY, furniture = FURNITURE0)'],
        
                                ["INSTRUCTION: Delete the single bed in the room.",
                                'SHAPE, LAYOUT, DEPTH, SEMANTIC, TEXTURE, TEXT, EMPTY, FURNITURE = LoadRoom()\n'\
                                'FURNITURE0 = EditFurniture(load = FURNITURE, coordinates = SHAPE, remove = "single bed")\n'\
                                'RESULT = Merge(emptyroom = EMPTY, furniture = FURNITURE0)'],

                                ["INSTRUCTION: Delete the desk and add a wardrobe",
                                'SHAPE, LAYOUT, DEPTH, SEMANTIC, TEXTURE, TEXT, EMPTY, FURNITURE = LoadRoom()\n'\
                                'FURNITURE0 = EditFurniture((load = FURNITURE, coordinates = SHAPE, remove = "desk")\n'\
                                'FURNITURE1 = EditFurniture(load = FURNITURE0, coordinates = SHAPE, add = "wardrobe")\n'\
                                'RESULT = Merge(emptyroom = EMPTY, furniture = FURNITURE1)'], 

                                ["INSTRUCTION: Delete the sofa. Then, change the texture of the room to bricks.",
                                'SHAPE, LAYOUT, DEPTH, SEMANTIC, TEXTURE, TEXT, EMPTY, FURNITURE = LoadRoom()\n'\
                                'FURNITURE0 = EditFurniture(load = FURNITURE, coordinates = SHAPE, remove = "sofa")\n'\
                                'TEXTURE0 = EditTexture(load = TEXTURE, coordinates = SHAPE, layout = LAYOUT, depth = DEPTH, semantic = SEMANTIC, prompt= "the texture of the room is bricks.")\n'\
                                'EMPTY0 = EditEmptyRoom(load = EMPTY, coordinates = SHAPE, texture = TEXTURE0)\n'\
                                'RESULT = Merge(emptyroom = EMPTY0, furniture = FURNITURE)'],

                                ["INSTRUCTION: Change the color of the room to red.",
                                'SHAPE, LAYOUT, DEPTH, SEMANTIC, TEXTURE, TEXT, EMPTY, FURNITURE = LoadRoom()\n'\
                                'TEXTURE0 = EditTexture(load = TEXTURE, coordinates = SHAPE, layout = LAYOUT, depth = DEPTH, semantic = SEMANTIC, prompt= "the room is painted in red color")\n'\
                                'EMPTY0 = EditEmptyRoom(load = EMPTY, coordinates = SHAPE, texture = TEXTURE0 )\n'\
                                'RESULT = Merge(emptyroom = EMPTY0, furniture = FURNITURE)'],

                                ["INSTRUCTION: Generate furniture in the empty living room.",
                                'SHAPE, LAYOUT, DEPTH, SEMANTIC, TEXTURE, TEXT, EMPTY, FURNITURE = LoadRoom()\n'\
                                'FURNITURE0 = GenFurniture(coordinates = SHAPE, roomtype = "livingroom")\n'\
                                'RESULT = Merge(emptyroom = EMPTY, furniture = FURNITURE0)'],

                                ["INSTRUCTION: Create a pink bedroom with furniture. The floor area is 35m^2, and the height is 3m. There are two windows and a door.",
                                'SHAPE = GenShape(prompt = "A bedroom whose floor area is 35m^2, and the height is 3m.")\n'\
                                'LAYOUT = GenLayout(coordinates = SHAPE)\n'\
                                'DEPTH = GenDepth(coordinates = SHAPE)\n'\
                                'SEMANTIC = GenSemantic(coordinates = SHAPE, prompt = "the room is painted in pink. There are 2 windows and 1 door in the room.")\n'\
                                'TEXTURE = GenTexture(coordinates = SHAPE, layout = LAYOUT, depth = DEPTH, semantic = SEMANTIC, prompt= "the room is painted in pink. There are 2 windows and 1 door.")\n'\
                                'EMPTY1 = GenEmptyRoom(coordinates = SHAPE, texture = TEXTURE)\n'\
                                'FURNITURE = GenFurniture(coordinates=SHAPE, roomtype = "bedroom")\n'\
                                'RESULT = Merge(emptyroom = OBJ1, furniture = OBJ2)'],

                                ["INSTRUCTION: Create an empty room of which the walls, ceiling, and floor are all covered with light yellow tiles. The tiles have a wave pattern, creating a sense of continuity and elegance. The floor follows template no.1, and the height is 2m.",
                                'SHAPE = GenShape(prompt = "template 1, and the height is 2m.")\n'\
                                'LAYOUT = GenLayout(coordinates = SHAPE)\n'\
                                'DEPTH = GenDepth(coordinates = SHAPE)\n'\
                                'SEMANTIC = GenSemantic(coordinates=SHAPE, prompt = "the walls, ceiling, and floor are all covered with light yellow tiles. The tiles have a wave pattern, creating a sense of continuity and elegance.")\n'\
                                'TEXTURE = GenTexture(coordinates = SHAPE, layout = LAYOUT, depth = DEPTH, semantic = SEMANTIC, prompt= "the walls, ceiling, and floor are all covered with light yellow tiles. The tiles have a wave pattern, creating a sense of continuity and elegance.")\n'\
                                'RESULT = GenEmptyRoom(coordinates = SHAPE, texture = TEXTURE)'],

                                ["INSTRUCTION: Double the width of the room, and change the color of the room to blue.",
                                'SHAPE, LAYOUT, DEPTH, SEMANTIC, TEXTURE, TEXT, EMPTY, FURNITURE = LoadRoom()\n'\
                                'SHAPE0 = EditShape(load = SHAPE, prompt="Double the width of the room.")\n'\
                                'LAYOUT0 = EditLayout(load = LAYOUT, coordinates = SHAPE0)\n'\
                                'DEPTH0 = EditDepth(load = DEPTH, coordinates = SHAPE0)\n'\
                                'SEMANTIC0 = EditSemantic(load = SEMANTIC, coordinates = SHAPE0, prompt = "Color of the room is blue.")\n'\
                                'TEXTURE0 = EditTexture(coordinates = SHAPE0, layout = LAYOUT0, depth = DEPTH0, semantic = SEMANTIC0, prompt = "Color of the room is blue.")\n'\
                                'EMPTY0 = EditEmptyRoom(load = EMPTY, coordinates = SHAPE0, texture = TEXTURE0)\n'\
                                'RESULT = Merge(emptyroom = EMPTY0, furniture = FURNITURE)']
                                ]
            

            
        message_list.append({'role': 'system', 'content': prompt +"\n PROGRAM: "})

        for example in supporting_examples:
            current_messages = [
                {'role': 'user', 'content': example[0] +"\n PROGRAM: "},
                {'role': 'assistant', 'content': example[1]},
            ]
            message_list = message_list + current_messages
        
        
        message_list.append({'role': 'user', 'content': "INSTRUCTION: {}".format(text_input)})
        
        return message_list

    def generate_response(self, text_input):
        prompt_for_gpt4 = self.form_prompt_for_chatgpt(text_input)
        response = openai.ChatCompletion.create(
            model=self.gpt_name,
            messages=prompt_for_gpt4,
            temperature=0.7,
            max_tokens=1024,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        
        return response["choices"][0]["message"]["content"]

class GenShape:
    def __init__(self, openai_api_key, save_directory, prompt):
        openai.organization = ""
        openai.api_key = openai_api_key
        self.gpt_name = 'gpt-4'
        self.save_directory = save_directory
        self.prompt = prompt

    def form_prompt_for_chatgpt(self, prompt):

        message_list = []
        instruction = "You are a 2D room designer. Your task is to create floor plans based on specific client instructions." \
            "The floor plans represent the top-down view of rooms. The centroid of each floor plan should be at coordinates (0, 0)." \
                "The clients provide instructions regarding the shape of the rooms. \n"\
                "If the height information is not given, set the default height value as 4."\
                "Remember that the 2D coordinate represents (width, length).\n"\
                "When calculating the area of a reactangular floor, remember that the area is equal to width * length. m^2 is equal to m².\n"\
                "Never include 0 in any of the corners coordinates.\n"

        supporting_examples = [["Generate a bedroom with a square-shaped floor.","floor = (-3,3),(3,3),(-3,3),(-3,-3), height = 4"],
                            ["Create a rectangular livingroom. The maximum width, length, height of the room is 3m,5m,5m.","floor = (-1.5,2.5),(1.5,2.5),(1.5,-2.5),(-1.5,-2.5), height = 5"],
                            ["Generate a cross-shaped dining room. The ratio of maximum width, maximum length, height is 1:2:3.",
                                "floor = (1.25, 5), (1.25, 2.5), (2.5, 2.5), (2.5, -2.5), (1.25, -2.5), (1.25, -5), (-1.25, -5), (-1.25, -2.5), (-2.5, -2.5), (-2.5, 2.5), (-1.25, 2.5), (-1.25, 5), height = 15 "],
                                ["Generate a living room in the shape of 'U'. The total floor area should be 25m².", "floor = (-0.5,0.5),(0.5,0.5),(0.5,1.5),(1.5,1.5),(1.5,-0.5),(-1.5,-0.5),(-1.5,1.5),(-0.5,1.5), height=4"],
                                ["Generate a livingroom whose shape is like 'L' with unequal arms. Length of longer arm is 5 meters, length of shorter arm is 3 meters.", 
                                "floor = (-2.5,0.5),(-2.5,2.5),(2.5,2.5),(2.5,-0.5),(-0.7,-0.3),(-0.5,0.5), height = 4"],
                                ["Generate a livingroom whose shape is like 'L' whose height is 3m.", 
                                "floor = (-4,4),(4,4),(4,-2),(-3,-2),(-3.2,0.8),(-4,1), height = 3"],
                                ["template 15, and the height is 3m", 
                                "floor = template 15, height = 3"],
                                ["I want a bedroom",'floor = (-3.5, 2.5),(3.5, 2.5),(3.5, -2.5),(-3.5, -2.5), height = 4'],
                                ["Generate a triangular room.", "floor = (-3,-2),(-2,3),(2,-1.5), height = 4"],
                                ["Generate a room whose floor is equal to 35m^2, and the height is 3.5.", "floor = (-2.5,3.5),(2.5,3.5),(2.5,-3.5),(-2.5,-3.5), height = 3.5"],
                                ["Generate a room whose floor is equal to 40m^2, and the height is 3m.", "floor = (-2.5,4),(2.5,4),(2.5,-4),(-2.5,-4), height = 3"],
                                ["A livingroom whose area is 45m^2", "floor = (-2.5,4.5),(2.5,4.5),(2.5,-4.5),(-2.5,4.5), height = 4"]]

        message_list.append({'role': 'system', 'content': instruction})

        for example in supporting_examples:
            current_messages = [
                {'role': 'user', 'content': "Instruction: {} \n".format(example[0]) +"Result 2D coordinates: \n"},
                {'role': 'assistant', 'content': example[1]},
            ]
            message_list = message_list + current_messages
        
        message_list.append({'role': 'user', 'content': prompt+"Result 2D coordinates: \n"})
        
        return message_list

    def run(self):
       
        prompt_for_gpt3 = self.form_prompt_for_chatgpt(self.prompt)
        response = openai.ChatCompletion.create(
            model=self.gpt_name,
            messages=prompt_for_gpt3,
            temperature=0.7,
            max_tokens=1024,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )

        response = response["choices"][0]["message"]["content"]
        gt_floor_corners = [] 

        if len(response.split("height =")) > 1:
            coordinates_str, height = response.split("height =")
            
        else:
            coordinates_str = response
            height = '4'

        coordinates_str= coordinates_str.lstrip("floor =").rstrip('),').strip()
        height = float(height.strip().strip().strip())

        if "template" in coordinates_str.lower():
            n = ""
            for char in coordinates_str:
                if char.isdigit():
                    n += char
            file_name = "template_{}_floor_coord.txt".format(str(n).zfill(2))
            texmplate_path = "Path to template"
            with open(os.path.join(texmplate_path, file_name), 'r') as file:
                for line in file:
                    # Split each line into a list of strings
                    w, l = line.split()
                    
                    # Convert each string to a float and create a list of floats
                    point = [float(w), float(l)]
                    point.insert(1, -height/2)
                    gt_floor_corners.append(point)

            FILE_NAME = "floor.json"
            with open(os.path.join(self.save_directory, FILE_NAME), "w") as floor:
                json.dump(gt_floor_corners, floor)
                return np.array(gt_floor_corners)


        coordinates_list = coordinates_str.strip().rstrip("),").split("),")
        for point in coordinates_list:
            w, l = point.strip().lstrip('(').strip().split(",")
            point = [float(w), float(l)]
            point.insert(1, -height/2)
            gt_floor_corners.append(point)

        FILE_NAME = "floor.json"
        with open(os.path.join(self.save_directory, FILE_NAME), "w") as floor:
            json.dump(gt_floor_corners, floor)
        
        return np.array(gt_floor_corners)


class GenLayout:
    def __init__(self, openai_api_key, save_directory, coordinates):

        self.save_directory = save_directory
        self.xyz = coordinates

    def height2ratio(self, height, camera_height=1.6):
        camera_height = height/2 #edited
        ceil_height = height - camera_height
        ratio = ceil_height / camera_height
        return ratio
    
    def run(self):

        height = np.abs(self.xyz[0][1]*2)
        pano_bd, visible_corners = draw_boundaries(np.zeros([512, 1024, 3]), corners_list=[self.xyz],
                                    boundary_color=[0, 255, 0], ratio=self.height2ratio(height))
        for floor, ceiling in zip(visible_corners[0], visible_corners[1]):
            floor[1] *= pano_bd.shape[0]
            ceiling[1] *= pano_bd.shape[0]
            floor[0] *= pano_bd.shape[1]
            ceiling[0] *= pano_bd.shape[1]
            
            pano_bd[int(ceiling[1]):int(floor[1]), int(floor[0]), 1] = 255
        
        result_array = np.concatenate(list(zip(visible_corners[1], visible_corners[0])), axis=0)   

        layout, semantic, _ = visualize_panorama_single(result_array)

        cv2.imwrite(os.path.join(self.save_directory,"layout.png"), layout)

        return layout

class GenDepth:
    def __init__(self, openai_api_key, save_directory, coordinates):

        self.save_directory = save_directory
        self.xyz = coordinates

   
    def run(self):
        uv = final_uv(self.xyz)
        
        height = np.abs(self.xyz[0][1])
        
        depth = estimate_depth(uv, height)

        Image.fromarray(depth).save(os.path.join(self.save_directory,'depth.tif'))

        tif_max = np.max(depth)
        tif_new = depth / tif_max
        tif_new *= 255
        cv2.imwrite(os.path.join(self.save_directory,"depth.png"), tif_new)

        return depth


class GenSemantic:

    def __init__(self, openai_api_key, save_directory, coordinates, prompt):

        self.save_directory = save_directory
        self.xyz = coordinates
        self.prompt = prompt

    def height2ratio(self, height, camera_height=1.6):
        camera_height = height/2 #edited
        ceil_height = height - camera_height
        ratio = ceil_height / camera_height
        return ratio

    def add_window_door(self, semantic, walls, door=None, window=None):
        # Create a new array with absolute y values
        abs_y_xyz = self.xyz.copy()
        abs_y_xyz[:, 1] = np.abs(self.xyz[:, 1])
        # Concatenate the original and new arrays
        concat_xyz = np.concatenate(list(zip(self.xyz, abs_y_xyz)), axis=0) 

        # Convert xyz coordinates to uv coordinates
        uv = xyz2uv(concat_xyz)

        # Create a dictionary
        uv_xyz_dict = {(uv[i, 0]*1024, uv[i, 1]*512): concat_xyz[i, :] for i in range(len(uv))}
        result = draw(semantic, walls, uv_xyz_dict, door, window)
        
        return result

    def run(self):

        height = np.abs(self.xyz[0][1]*2)
        pano_bd, visible_corners = draw_boundaries(np.zeros([512, 1024, 3]), corners_list=[self.xyz],
                                    boundary_color=[0, 255, 0], ratio=self.height2ratio(height))
        
        for floor, ceiling in zip(visible_corners[0], visible_corners[1]):
            floor[1] *= pano_bd.shape[0]
            ceiling[1] *= pano_bd.shape[0]
            floor[0] *= pano_bd.shape[1]
            ceiling[0] *= pano_bd.shape[1]
            
            pano_bd[int(ceiling[1]):int(floor[1]), int(floor[0]), 1] = 255
            
        result_array = np.concatenate(list(zip(visible_corners[1], visible_corners[0])), axis=0)   
        
        layout, semantic, walls = visualize_panorama_single(result_array)

        # Count window and door
        window = 0
        door = 0

        # Regex patterns to find windows and doors
        window_pattern = r"(\d+)\s*window"
        door_pattern = r"(\d+)\s*door"

        # Search for windows and doors in self.prompt
        window_match = re.search(window_pattern, self.prompt)
        door_match = re.search(door_pattern, self.prompt)

        if window_match:
            window = int(window_match.group(1))
        
        if door_match:
            door = int(door_match.group(1))


        semantic = self.add_window_door(semantic, walls, door, window)
        vis_semantic = cv2.cvtColor(semantic, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(self.save_directory,"semantic.png"), vis_semantic)

        return semantic   

class GenTexture:
    def __init__(self, openai_api_key, save_directory, layout, depth, semantic, prompt, coordinates):

        self.save_directory = save_directory
        self.layout = layout
        self.prompt = prompt
        self.depth = depth
        self.semantic = semantic
        self.xyz = coordinates

    def run(self):

        panorama_generator = PanoramaGenerator()

        images = panorama_generator.generate_panorama(self.save_directory, self.layout, self.depth, self.semantic, self.prompt)
        cv2.imwrite(os.path.join(self.save_directory, "texture.png"), images[-1])
        return images[-1] 

class GenEmptyRoom:
    def __init__(self, openai_api_key, save_directory,  coordinates, texture):

        self.save_directory = save_directory
        self.texture = texture
        self.coordinates = coordinates
    
    def run(self):
        
        height = np.abs(self.coordinates[0][1])
        uv = final_uv(self.coordinates)
        
        geometry = fold(self.texture, uv, height*2)
        geometry2 = fold2(self.texture, uv, height*2)
              

        o3d.io.write_triangle_mesh(os.path.join(self.save_directory, "room_without_ceiling.ply"), geometry[0], write_ascii=False, compressed=False, write_vertex_normals=True, write_vertex_colors=True, write_triangle_uvs=True, print_progress=False)
        o3d.io.write_triangle_mesh(os.path.join(self.save_directory, "room_with_ceiling.ply"), geometry2[0], write_ascii=False, compressed=False, write_vertex_normals=True, write_vertex_colors=True, write_triangle_uvs=True, print_progress=False)

        return geometry
    
class GenFurniture:
    def __init__(self, openai_api_key, save_directory, coordinates, roomtype):

        self.openai_api_key =  openai_api_key
        self.save_directory = save_directory
        self.furniture = roomtype
        self.coordinates = coordinates 

    def run(self):
        # List of 2D coordinates representing corners of the shape [(x1, y1), (x2, y2), ...]
        coordinates = [[sublist[0], sublist[2]] for sublist in self.coordinates]
        floor =  find_rectangle(coordinates)
        dataset_dir = self.save_directory
        layout_gpt = LayoutGPTSceneSynthesis(self.openai_api_key, floor, self.furniture, dataset_dir)
        json_file = layout_gpt.run_scene_synthesis()
        return json_file

class EditFurniture:
    def __init__(self, openai_api_key, save_directory, load, coordinates, remove=None, add=None, replace=None):

        self.openai_api_key =  openai_api_key
        self.save_directory = save_directory
        self.remove = remove
        self.add = add
        self.replace = replace
        self.furniture = load
        self.coordinates = coordinates 

    def run(self):

        # room = "livingroom"/
        room = "bedroom" 
        coordinates = [[sublist[0], sublist[2]] for sublist in self.coordinates]
        floor =  find_rectangle(coordinates)
        updated_data = self.furniture.copy()
        if self.remove:
            furniture_list = [item.strip() for item in self.remove.split(',')]
            for f in furniture_list:
                updated_data = [item for item in updated_data if item[0] != f]


        elif self.replace:
            furniture_list = [item.strip().split("->") for item in self.replace.split(',')]
            for f in furniture_list:
                funiture_editor = ReplaceFurniture(self.furniture, floor, f, room)
                updated_data = funiture_editor.replace_fur(self.furniture, f)

        elif self.add:
            furniture_list = [item.strip() for item in self.add.split(',')]
            for f in furniture_list:
                funiture_editor = AddFurniture(self.openai_api_key, self.furniture, floor, f, room)
                updated_data = funiture_editor.run_scene_synthesis()
                
        else:
            raise ValueError("Invalid order. Order must be 'replace', 'remove', 'delete', or 'add'.")   

        with open(os.path.join(self.save_directory, "furniture.json"), 'w') as new_file:
            json.dump(updated_data, new_file, indent=4)

        
        return updated_data


class Merge:
    def __init__(self, openai_api_key, save_directory, emptyroom, furniture):
        self.openai_api_key = openai_api_key
        self.save_directory = save_directory
        self.emptyroom = emptyroom
        self.furniture = furniture
    def run(self):
        
        parent_directory = os.path.dirname(self.save_directory)
        new_path = os.path.join(parent_directory, "results")
        
        os.makedirs(new_path, exist_ok=True)

        try:
            room_with_ceiling = os.path.join(self.save_directory, "room_with_ceiling.ply")
            room_without_ceiling = os.path.join(self.save_directory, "room_without_ceiling.ply")
            shutil.copy2(room_with_ceiling, os.path.join(new_path, "room_with_ceiling.ply"))
            shutil.copy2(room_without_ceiling, os.path.join(new_path, "room_without_ceiling.ply"))
        except:
            pass

        return new_path

class LoadRoom:
    def __init__(self, openai_api_key, save_directory):
        self.openai_api_key = openai_api_key
        self.save_directory = save_directory   

    def run(self):
        current_step = int(os.path.basename(self.save_directory))
        floor_list = None
        layout_image = None
        depth_image = None
        semantic_image = None
        texture_prompt = None
        empty_room = None
        furniture = None

        # Iterate through previous steps to find the file
        for step in range(current_step - 1, -1, -1):
            list_path = os.path.join(os.path.dirname(self.save_directory), str(step), "floor.json")
 
            if os.path.exists(list_path):
                with open(list_path, "r") as file:
                    floor_list = json.load(file)
                break  # Stop iterating if the file is found
          
        for step in range(current_step - 1, -1, -1):
            layout_path = os.path.join(os.path.dirname(self.save_directory), str(step), "layout.png")
            depth_path = os.path.join(os.path.dirname(self.save_directory), str(step), "depth.tif")
            semantic_path = os.path.join(os.path.dirname(self.save_directory), str(step), "semantic.png")
            if os.path.exists(layout_path):
                layout_image = cv2.imread(layout_path)
                semantic_image = cv2.imread(semantic_path)
                semantic_image = cv2.cvtColor(semantic_image, cv2.COLOR_BGR2RGB)
                depth_image = tiff.imread(depth_path)
                break  # Stop iterating if the file is found

        for step in range(current_step - 1, -1, -1):
            texture_path = os.path.join(os.path.dirname(self.save_directory), str(step), "texture.png")
            if os.path.exists(texture_path):
                texture_image = cv2.imread(texture_path)
                break  # Stop iterating if the file is found 
  
        for step in range(current_step - 1, -1, -1):
            text_path = os.path.join(os.path.dirname(self.save_directory), str(step), "texture_prompt.txt")
            if os.path.exists(text_path):
                with open(text_path, "r") as file:
                    texture_prompt = file.read()
                break  # Stop iterating if the file is found 

        for step in range(current_step - 1, -1, -1):
            room_path = os.path.join(os.path.dirname(self.save_directory), str(step), "room_with_ceiling.ply")
            if os.path.exists(room_path):
                empty_room = os.path.join(os.path.dirname(self.save_directory), str(step))

        for step in range(current_step - 1, -1, -1):
            furniture_path = os.path.join(os.path.dirname(self.save_directory), str(step), "furniture.json")
 
            if os.path.exists(furniture_path):
                with open(furniture_path, "r") as file:
                    furniture = json.load(file)
                break  # Stop iterating if the file is found

        if floor_list is None:
            raise ValueError("floor coordinates not found in previous steps.")
        if layout_image is None:
            raise ValueError("layout image not found in previous steps.")
        if depth_image is None:
            raise ValueError("depth image not found in previous steps.")
        if semantic_image is None:
            raise ValueError("semantic image not found in previous steps.")
        if texture_prompt is None:
            raise ValueError("texture prompt not found in previous steps.")
          
        return floor_list, layout_image, depth_image, semantic_image, texture_image, texture_prompt, empty_room, furniture

class EditShape():
    def __init__(self, openai_api_key, save_directory,load, prompt):
        openai.organization = ""
        openai.api_key = openai_api_key
        self.gpt_name = 'gpt-4'
        self.save_directory = save_directory
        self.coordinates = load
        self.prompt = prompt


    def form_prompt_for_chatgpt(self, coordinates, prompt):
        message_list = []
        instruction = "You are a 3D room designer. Your task is to edit floor plans following client's instructions." \
            "The floor plans represent the top-down view of rooms. " \
                "The coordites include (x,y,z) values. 'y' represents the negative half of the original height of the room. "\
                "The clients provide instructions regarding the shape of the rooms. \n"\
                "Remember that the coordinate represents (width, -height*0.5, length).\n"\
                "When calculating the area of a reactangular floor, remember that the area is equal to width * length. m^2 is equal to m²."\
                "When scaling the room, the ratio of the width, height, length should be the same or as similar as possible."\
                "Ensure that none of the corner's coordinates includes the value 0. \n\n"

        supporting_examples = [["Increase the height of the room by 2m. The 3D coordinates of the room's floor is (-3,-2,3),(3,-2,3),(-3,-2,3),(-3,-2,-3).","(-3,-3,3),(3,-3,3),(-3,-3,3),(-3,-3,-3)"],
                            ["Increase the width of the room by 1. The 3D coordinates of the room's floor is (-1.5,-1.5,2.5),(1.5,-1.5,2.5),(1.5,-1.5,-2.5),(-1.5,-1.5,-2.5)."," (-2,-1.5,2.5),(2,-1.5,2.5),(2,-1.5,-2.5),(-2,-1.5,-2.5)"],
                            ["Decrease the length of the room by 2m. The 3D coordinates of the room's floor is (1.25,-2, 5), (1.25,-2,2.5), (2.5,-2,2.5), (2.5,-2,-2.5), (1.25,-2,-2.5), (1.25,-2,-5), (-1.25,-2,-5), (-1.25,-2,-2.5), (-2.5,-2,-2.5), (-2.5,-2,2.5), (-1.25,-2,2.5), (-1.25,-2,5).",
                                "(1.25, -2, 4), (1.25, -2, 1.5), (2.5, -2, 1.5), (2.5, -2, -1.5), (1.25, -2, -1.5), (1.25, -2, -4), (-1.25, -2, -4), (-1.25, -2, -1.5), (-2.5, -2, -1.5), (-2.5, -2, 1.5), (-1.25, -2, 1.5),(-1.25, -2, 4)"],
                            ["Decrease the height of the room by 0.5m. The 3D coordinates of the room's floor is (-3,-1.5,3),(3,-1.5,3),(-3,-1.5,3),(-3,-1.5,-3).","(-3,-1.75,2.5),(3,-1.75,2.5),(-3,-1.75,2.5),(-3,-1.75,-2.5)"],
                                ["Decrease the  width of the room by 0.7m. The 3D coordinates of the room's floor is (-3,-1,3),(3,-1,3),(-3,-1,3),(-3,-1,-3).","(-2.65,-1,2.5),(2.65,-1,2.5),(-2.65,-1,2.5),(-2.65,-1,-2.5)"],
                                ["Decrease the length of the room by 1m. The 3D coordinates of the room's floor is (-3,-2,3),(3,-2,3),(-3,-2,3),(-3,-2,-3).","(-3,-2,2.5),(3,-2,2.5),(-3,-2,2.5),(-3,-2,-2.5)"]]
                             

        message_list.append({'role': 'system', 'content': instruction})

        for example in supporting_examples:
            current_messages = [
                {'role': 'user', 'content': "Instruction: {}\n".format(example[0]) +"Result 3D coordinates: \n"},
                {'role': 'assistant', 'content': example[1]},
            ]
            message_list = message_list + current_messages
        
        message_list.append({'role': 'user', 'content': "Instruction: {} The 3D coordinates of the room's floor is {}\n".format(prompt, coordinates) +"Result 3D coordinates: \n"})
        
        return message_list

    def run(self):
       
        prompt_for_gpt3 = self.form_prompt_for_chatgpt(self.coordinates, self.prompt)
        
        response = openai.ChatCompletion.create(
            model=self.gpt_name,
            messages=prompt_for_gpt3,
            temperature=0.7,
            max_tokens=1024,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )

        response = response["choices"][0]["message"]["content"]
        gt_floor_corners = [] 
        coordinates_str= response.lstrip('(').rstrip(')').strip()


        coordinates_list = coordinates_str.strip().rstrip("],").split(",[")
        
        for point in coordinates_list:
            
            w, h, l = point.strip().lstrip('[').strip().split(",")            
            
            point = [float(w), float(h), float(l)]
            
            gt_floor_corners.append(point)

        FILE_NAME = "floor.json"
        with open(os.path.join(self.save_directory, FILE_NAME), "w") as floor:
            json.dump(gt_floor_corners, floor)
        
        return np.array(gt_floor_corners)
    
class EditLayout:
    def __init__(self, openai_api_key, save_directory, load, coordinates):

        self.save_directory = save_directory
        self.xyz = coordinates
        self.initial_layout = load
        
    def height2ratio(self, height, camera_height=1.6):
        camera_height = height/2 #edited
        ceil_height = height - camera_height
        ratio = ceil_height / camera_height
        return ratio
    
    def run(self):

        height = np.abs(self.xyz[0][1]*2)
        pano_bd, visible_corners = draw_boundaries(np.zeros([512, 1024, 3]), corners_list=[self.xyz],
                                    boundary_color=[0, 255, 0], ratio=self.height2ratio(height))
        for floor, ceiling in zip(visible_corners[0], visible_corners[1]):
            floor[1] *= pano_bd.shape[0]
            ceiling[1] *= pano_bd.shape[0]
            floor[0] *= pano_bd.shape[1]
            ceiling[0] *= pano_bd.shape[1]
            
            pano_bd[int(ceiling[1]):int(floor[1]), int(floor[0]), 1] = 255
        
        result_array = np.concatenate(list(zip(visible_corners[1], visible_corners[0])), axis=0)   

        layout, semantic, _ = visualize_panorama_single(result_array)

        cv2.imwrite(os.path.join(self.save_directory,"layout.png"), layout)

        return layout

class EditDepth:
    def __init__(self, openai_api_key, save_directory, load, coordinates):

        self.save_directory = save_directory
        self.xyz = coordinates
        self.initial_depth = load

   
    def run(self):
        uv = final_uv(self.xyz)
        
        height = np.abs(self.xyz[0][1])
        
        depth = estimate_depth(uv, height)

        Image.fromarray(depth).save(os.path.join(self.save_directory,'depth.tif'))

        tif_max = np.max(depth)
        
        tif_new = depth / tif_max
        tif_new *= 255
        cv2.imwrite(os.path.join(self.save_directory,"depth.png"), tif_new)

        return depth
    
class EditSemantic:

    def __init__(self, openai_api_key, save_directory, load, coordinates, prompt):

        self.save_directory = save_directory
        self.xyz = coordinates
        self.prompt = prompt
        self.load = load
    def height2ratio(self, height, camera_height=1.6):
        camera_height = height/2 #edited
        ceil_height = height - camera_height
        ratio = ceil_height / camera_height
        return ratio

    def add_window_door(self, semantic, walls, door=None, window=None):
        # Create a new array with absolute y values
        abs_y_xyz = self.xyz.copy()
        abs_y_xyz[:, 1] = np.abs(self.xyz[:, 1])
        # Concatenate the original and new arrays
        concat_xyz = np.concatenate(list(zip(self.xyz, abs_y_xyz)), axis=0) 

        # Convert xyz coordinates to uv coordinates
        uv = xyz2uv(concat_xyz)

        # Create a dictionary
        uv_xyz_dict = {(uv[i, 0]*1024, uv[i, 1]*512): concat_xyz[i, :] for i in range(len(uv))}
        result = draw(semantic, walls, uv_xyz_dict, door, window)
        
        return result

    def run(self):

        height = np.abs(self.xyz[0][1]*2)
        pano_bd, visible_corners = draw_boundaries(np.zeros([512, 1024, 3]), corners_list=[self.xyz],
                                    boundary_color=[0, 255, 0], ratio=self.height2ratio(height))
        
        for floor, ceiling in zip(visible_corners[0], visible_corners[1]):
            floor[1] *= pano_bd.shape[0]
            ceiling[1] *= pano_bd.shape[0]
            floor[0] *= pano_bd.shape[1]
            ceiling[0] *= pano_bd.shape[1]
            
            pano_bd[int(ceiling[1]):int(floor[1]), int(floor[0]), 1] = 255
            
        result_array = np.concatenate(list(zip(visible_corners[1], visible_corners[0])), axis=0)   
        
        layout, semantic, walls = visualize_panorama_single(result_array)

        # Count window and door
        window = 0
        door = 0

        # Regex patterns to find windows and doors
        window_pattern = r"(\d+)\s*window"
        door_pattern = r"(\d+)\s*door"

        # Search for windows and doors in self.prompt
        window_match = re.search(window_pattern, self.prompt)
        door_match = re.search(door_pattern, self.prompt)

        if window_match:
            window = int(window_match.group(1))
        
        if door_match:
            door = int(door_match.group(1))


        semantic = self.add_window_door(semantic, walls, door, window)
        vis_semantic = cv2.cvtColor(semantic, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(self.save_directory,"semantic.png"), vis_semantic)

        return semantic   
    
class EditTexture:
    def __init__(self, openai_api_key, save_directory, load, coordinates, layout, depth, semantic, prompt):

        self.save_directory = save_directory
        self.layout = layout
        self.prompt = prompt
        self.depth = depth
        self.semantic = semantic
        self.xyz = coordinates
        self.load = load

    def run(self):

        panorama_generator = PanoramaGenerator()

        images = panorama_generator.generate_panorama(self.save_directory, self.layout, self.depth, self.semantic, self.prompt)
        cv2.imwrite(os.path.join(self.save_directory, "texture.png"), images[-1])
        return images[-1] 

class EditEmptyRoom:
    def __init__(self, openai_api_key, save_directory,  load, coordinates, texture):

        self.save_directory = save_directory
        self.texture = texture
        self.coordinates = coordinates
        self.load = load
    
    def run(self):
        
        height = np.abs(self.coordinates[0][1])
        uv = final_uv(self.coordinates)
        
        geometry = fold(self.texture, uv, height*2)
        geometry2 = fold2(self.texture, uv, height*2)

        o3d.io.write_triangle_mesh(os.path.join(self.save_directory, "room_without_ceiling.ply"), geometry[0], write_ascii=False, compressed=False, write_vertex_normals=True, write_vertex_colors=True, write_triangle_uvs=True, print_progress=False)
        o3d.io.write_triangle_mesh(os.path.join(self.save_directory, "room_with_ceiling.ply"), geometry2[0], write_ascii=False, compressed=False, write_vertex_normals=True, write_vertex_colors=True, write_triangle_uvs=True, print_progress=False)

        return geometry