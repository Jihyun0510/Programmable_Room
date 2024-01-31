import os
import argparse
from engine.modules import ProgramInterpreter, GenShape, GenLayout, GenDepth, GenSemantic, GenTexture, GenEmptyRoom, GenFurniture, EditFurniture, EditShape, EditLayout, EditDepth, EditSemantic, EditTexture, EditEmptyRoom, Merge, LoadRoom
import time
import sys
from glob import glob
from datetime import datetime
from tqdm import tqdm


class VisualProgrammer:
    def __init__(self, openai_api_key):
        self.openai_api_key = openai_api_key

    def parse_response(self, response):
        functions_list = response.split('\n')
        parsed_functions = []

        for function_str in functions_list:
            
            f = {}
            index = function_str.find('=')

            if index != -1:
                variable_name = function_str[:index].strip()
                function_call = function_str[index+2:].strip()

            function_name, parameters = function_call.split('(')
            
            parameters = parameters.rstrip(')')
            function_params = [param.strip() for param in parameters.split(',')]
            f["variable_name"] = variable_name
            f["function_name"] = function_name
            f["parameters"] = function_params
            parsed_functions.append(f)
        return parsed_functions


    def generate_response(self, input_text):
        assistant = ProgramInterpreter(self.openai_api_key)
        response = assistant.generate_response(input_text)
        return response


def main():

    parser = argparse.ArgumentParser(description="ChatGPT Assistant")
    parser.add_argument("--instruction", type=str, help="Instruction for the room")
    args = parser.parse_args()

    now = datetime.now()
    now_str = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_directory = os.path.join("results", now_str)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    openai_api_key = "YOUR_OPENAI_API_KEY"
    assistant = VisualProgrammer(openai_api_key)

    
     #Keep receiving texts
    iteration = 0
    instruction = args.instruction
    while True:
        if instruction.lower() == "stop":
            print("Stopping the process.")
            break

        response = assistant.generate_response(instruction)

        output_file = os.path.join(output_directory, f"{iteration}.txt")
        with open(output_file, "w") as file:
            file.write(response)

        lines = assistant.parse_response(response)
        save_directory = os.path.join(output_directory, f"{iteration}")
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        
        try:
            for function_dict in lines:

                variable_name = function_dict["variable_name"]
                function_name = function_dict["function_name"]
                function_params = function_dict["parameters"]

                function_call = f"{function_name}(openai_api_key=openai_api_key, save_directory=save_directory, "
                                
                for param in function_params:
                    function_call += param + ', '

                function_call = function_call.rstrip(', ') + ')'
                
                print(function_call)
                instance = eval(function_call)
                result = instance.run()    
                globals()[variable_name] = result

        except Exception as e:
            print(f"An error occurred: {e}")


        # Get next instruction from user
        instruction = input("Enter next instruction (or 'stop' to end): ")
        iteration += 1



if __name__ == "__main__":
    main()