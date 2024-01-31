# Programmable-Room: Interactive Textured 3D Room Meshes Generation Empowered by Large Language Models

We present Programmable-Room, a framework which interactively generates and edits a 3D room mesh, given natural language instructions. 
More specifically, our Programmable-Room interprets user-provided descriptions to create plausible 3D coordinates for room meshes, to generate panorama images for the texture, to construct 3D meshes by integrating the coordinates and panorama texture images, and to arrange furniture, allowing users to specify single or combined actions as needed. 
Inspired by visual programming (VP), Programmable-Room utilizes a large language model (LLM) to write a python program which is an ordered list of necessary modules for the various tasks given in natural language. 
We developed most of the modules. For the texture generating module, we utilize a pretrained large-scale diffusion model to generate panorama images conditioned on text and visual prompts (i.e., layout, depth, and semantic map) simultaneously. Specifically, we accelerate the performance of panorama image generation by optimizing the training objective with 1D representation of panorama scene obtained from bidirectional LSTM. We demonstrate Programmable-Room's flexibility in generating and editing 3D room meshes, and prove our framework's superiority to an existing model quantitatively and qualitatively.


## Description

This repo is the official PyTorch implementation of our Programmable-Room paper.

## Setup

bash
conda env create -f environment/environment.yaml
conda activate ldm

