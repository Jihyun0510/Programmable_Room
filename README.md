# Programmable-Room: Interactive Textured 3D Room Meshes Generation Empowered by Large Language Models

We present Programmable-Room, a framework which interactively generates and edits a 3D room mesh, given natural language instructions. 

## Description

This repo is the official PyTorch implementation of our Programmable-Room paper.

![Main Image](images/main.png)

Programmable-Room interprets user-provided descriptions to create plausible 3D coordinates for room meshes, to generate panorama images for the texture, to construct 3D meshes by integrating the coordinates and panorama texture images, and to arrange furniture, allowing users to specify single or combined actions as needed. 
Inspired by visual programming (VP), Programmable-Room utilizes a large language model (LLM) to write a python program which is an ordered list of necessary modules for the various tasks given in natural language. 

We developed most of the modules. For the texture generating module, we utilize a pretrained large-scale diffusion model to generate panorama images conditioned on text and visual prompts (i.e., layout, depth, and semantic map) simultaneously. Specifically, we accelerate the performance of panorama image generation by optimizing the training objective with 1D representation of panorama scene obtained from bidirectional LSTM.

## Setup

### Environment
Generate a conda environment.

```bash
conda env create -f environment.yaml
conda activate PR
```
### Download
Download the [pretrained weights](https://drive.google.com/file/d/1zU6xGu9DK4OKGUVTS65zL45TEX78sbAC/view?usp=sharing) for the texture generation model. Then place it under ```diffusion/ckpt```.

## Interactive Scene Generation

![Editing Example](images/editing.png)

To generate a scene, you can simpy run the ```inference.py``` scrips. For example,

```bash
python inference.py --instruction "Generate a bedroom whose area of the floor is equal to 30m^2. The walls, where there are a door and a window, are covered with light green fabric with stripes."
```

Then, in the terminal, you can continue inserting additional instructions. For example,

```bash
"Remove the ceiling lamp and chair from the room."
"Replace the desk with a coffee table."
"Decrease the width of the room by 1.2m."
"Add an armchair in the room."
"Cover the room with dark wood to make the room look like an old cottage."
```

The code will stop if you enter "stop".

```bash
"stop"
```

## Acknowledgements
This code is built on the codes from the [Stable Diffusion](https://github.com/CompVis/stable-diffusion.git), [ControlNet](https://github.com/lllyasviel/ControlNet.git), [LayoutGPT](https://github.com/weixi-feng/LayoutGPT.git), and [Structured3D](https://github.com/bertjiazheng/Structured3D.git).
