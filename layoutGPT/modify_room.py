import numpy as np
from plyfile import PlyData, PlyElement
import tripy

def load_ply_mesh(file_path):
    plydata = PlyData.read(file_path)
    vertices = np.vstack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).T
    faces = np.vstack(plydata['face']['vertex_indices'])
    return vertices, faces

def calculate_mesh_size(vertices):
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    size = max_coords - min_coords
    return size

if __name__ == "__main__":
    # Replace 'your_mesh_file.ply' with the path to your PLY file
    mesh_file_path = 'your_mesh_file.ply'

    # Load the textured mesh from the PLY file
    vertices, faces = load_ply_mesh(mesh_file_path)

    # Calculate the size of the mesh
    mesh_size = calculate_mesh_size(vertices)

    print(f"Mesh Size (X, Y, Z): {mesh_size}")