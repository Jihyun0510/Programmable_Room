import numpy as np
from engine.conversion import final_uv, cor_2_1d, np_coor2xy, np_coorx2u, np_coory2v
import open3d as o3d
from PIL import Image
from scipy.signal import correlate2d
from scipy.ndimage import shift
from plyfile import PlyData, PlyElement
import cv2
from Structured3D.segmentation import visualize_panorama_single
import largestinteriorrectangle as lir
import openai


def layout_2_depth(cor_id, h, w, height, return_mask=False):
    # Convert corners to per-column boundary first
    # Up -pi/2,  Down pi/2
    vc, vf = cor_2_1d(cor_id, h, w)
    vc = vc[None, :]  # [1, w]
    vf = vf[None, :]  # [1, w]
    assert (vc > 0).sum() == 0
    assert (vf < 0).sum() == 0

    # Per-pixel v coordinate (vertical angle)
    vs = ((np.arange(h) + 0.5) / h - 0.5) * np.pi
    vs = np.repeat(vs[:, None], w, axis=1)  # [h, w]

    # Floor-plane to depth
    # floor_h = 1.6
    floor_h = height
    floor_d = np.abs(floor_h / np.sin(vs))

    # wall to camera distance on horizontal plane at cross camera center
    cs = floor_h / np.tan(vf)

    # Ceiling-plane to depth
    ceil_h = np.abs(cs * np.tan(vc))      # [1, w]
    ceil_d = np.abs(ceil_h / np.sin(vs))  # [h, w]

    # Wall to depth
    wall_d = np.abs(cs / np.cos(vs))  # [h, w]

    # Recover layout depth
    floor_mask = (vs > vf)
    ceil_mask = (vs < vc)
    wall_mask = (~floor_mask) & (~ceil_mask)
    depth = np.zeros([h, w], np.float32)    # [h, w]
    depth[floor_mask] = floor_d[floor_mask]
    depth[ceil_mask] = ceil_d[ceil_mask]
    depth[wall_mask] = wall_d[wall_mask]

    assert (depth == 0).sum() == 0
    if return_mask:
        return depth, floor_mask, ceil_mask, wall_mask
    return depth

def estimate_depth(uv, height):
    H, W = 512, 1024
    cor_id = []
    ceiling_uv = uv[0]
    floor_uv = uv[1]
    for floor, ceiling in zip(ceiling_uv, floor_uv):
        cor_id.append(floor)
        cor_id.append(ceiling)

    cor_id = np.array(cor_id, np.float32)
    cor_id[:, 0] *= W
    cor_id[:, 1] *= H
    depth, floor_mask, ceil_mask, wall_mask = layout_2_depth(cor_id, H, W, height, return_mask=True)


    return depth

def np_coor2xy(coor, z=50, coorW=1024, coorH=512, floorW=1024, floorH=512):
    '''
    coor: N x 2, index of array in (col, row) format
    '''
    coor = np.array(coor)
    u = np_coorx2u(coor[:, 0], coorW)
    v = np_coory2v(coor[:, 1], coorH)
    c = z / np.tan(v)
    x = c * np.sin(u) + floorW / 2 - 0.5
    y = -c * np.cos(u) + floorH / 2 - 0.5
    return np.hstack([x[:, None], y[:, None]])



def fold(texture, uv, height):
    # equirect_texture = np.array(Image.open(args.img))
    equirect_texture = texture

    H, W = equirect_texture.shape[:2]
    cor_id = []
    ceiling_uv = uv[0]
    floor_uv = uv[1]
    for floor, ceiling in zip(ceiling_uv, floor_uv):
        cor_id.append(floor)
        cor_id.append(ceiling)

    cor_id = np.array(cor_id, np.float32)
    cor_id[:, 0] *= W
    cor_id[:, 1] *= H

    # Convert corners to layout
    depth, floor_mask, ceil_mask, wall_mask = layout_2_depth(cor_id, H, W, height, return_mask=True)



    coorx, coory = np.meshgrid(np.arange(W), np.arange(H))
    us = np_coorx2u(coorx, W)
    vs = np_coory2v(coory, H)
    zs = depth * np.sin(vs)
    cs = depth * np.cos(vs)
    xs = cs * np.sin(us)
    ys = -cs * np.cos(us)

    # Aggregate mask
    mask = np.ones_like(floor_mask)
    mask &= ~ceil_mask #edited


    # Prepare ply's points and faces
    xyzrgb = np.concatenate([
        xs[...,None], ys[...,None], zs[...,None],
        equirect_texture], -1)
    xyzrgb = np.concatenate([xyzrgb, xyzrgb[:,[0]]], 1)
    mask = np.concatenate([mask, mask[:,[0]]], 1)
    lo_tri_template = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 1, 1]])
    up_tri_template = np.array([
        [0, 0, 0],
        [0, 1, 1],
        [0, 0, 1]])
    ma_tri_template = np.array([
        [0, 0, 0],
        [0, 1, 1],
        [0, 1, 0]])
    lo_mask = (correlate2d(mask, lo_tri_template, mode='same') == 3)
    up_mask = (correlate2d(mask, up_tri_template, mode='same') == 3)
    ma_mask = (correlate2d(mask, ma_tri_template, mode='same') == 3) & (~lo_mask) & (~up_mask)
    ref_mask = (
        lo_mask | (correlate2d(lo_mask, np.flip(lo_tri_template, (0,1)), mode='same') > 0) |\
        up_mask | (correlate2d(up_mask, np.flip(up_tri_template, (0,1)), mode='same') > 0) |\
        ma_mask | (correlate2d(ma_mask, np.flip(ma_tri_template, (0,1)), mode='same') > 0)
    )
    points = xyzrgb[ref_mask]

    ref_id = np.full(ref_mask.shape, -1, np.int32)
    ref_id[ref_mask] = np.arange(ref_mask.sum())
    faces_lo_tri = np.stack([
        ref_id[lo_mask],
        ref_id[shift(lo_mask, [1, 0], cval=False, order=0)],
        ref_id[shift(lo_mask, [1, 1], cval=False, order=0)],
    ], 1)
    faces_up_tri = np.stack([
        ref_id[up_mask],
        ref_id[shift(up_mask, [1, 1], cval=False, order=0)],
        ref_id[shift(up_mask, [0, 1], cval=False, order=0)],
    ], 1)
    faces_ma_tri = np.stack([
        ref_id[ma_mask],
        ref_id[shift(ma_mask, [1, 0], cval=False, order=0)],
        ref_id[shift(ma_mask, [0, 1], cval=False, order=0)],
    ], 1)
    faces = np.concatenate([faces_lo_tri, faces_up_tri, faces_ma_tri])
   

    points[:, [3, 5]] = points[:, [5, 3]] #Edited
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points[:, :3])
    mesh.vertex_colors = o3d.utility.Vector3dVector(points[:, 3:] / 255.)
    mesh.triangles = o3d.utility.Vector3iVector(faces)


    vertices = np.asarray(mesh.vertices)
    min_z = np.min(vertices[:, 2])
    shifted_vertices = vertices - [0, 0, min_z]
    current_height = np.max(shifted_vertices, axis=0)[2]  # Z dimension
    scaling_factor = height*1.2 / current_height

    scaled_vertices = shifted_vertices * scaling_factor    

    
    mesh.vertices = o3d.utility.Vector3dVector(scaled_vertices)


    draw_geometries = [mesh]


    return draw_geometries

def fold2(texture, uv, height):
    equirect_texture = texture

    H, W = equirect_texture.shape[:2]
    cor_id = []
    ceiling_uv = uv[0]
    floor_uv = uv[1]
    for floor, ceiling in zip(ceiling_uv, floor_uv):
        
        cor_id.append(floor)
        cor_id.append(ceiling)

    cor_id = np.array(cor_id, np.float32)
    cor_id[:, 0] *= W
    cor_id[:, 1] *= H



    # Convert corners to layout
    depth, floor_mask, ceil_mask, wall_mask = layout_2_depth(cor_id, H, W, height, return_mask=True)

  
    coorx, coory = np.meshgrid(np.arange(W), np.arange(H))
    us = np_coorx2u(coorx, W)
    vs = np_coory2v(coory, H)
    zs = depth * np.sin(vs)
    cs = depth * np.cos(vs)
    xs = cs * np.sin(us)
    ys = -cs * np.cos(us)

    mask = np.ones_like(floor_mask)

    # Prepare ply's points and faces
    xyzrgb = np.concatenate([
        xs[...,None], ys[...,None], zs[...,None],
        equirect_texture], -1)
    xyzrgb = np.concatenate([xyzrgb, xyzrgb[:,[0]]], 1)
    mask = np.concatenate([mask, mask[:,[0]]], 1)
    lo_tri_template = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 1, 1]])
    up_tri_template = np.array([
        [0, 0, 0],
        [0, 1, 1],
        [0, 0, 1]])
    ma_tri_template = np.array([
        [0, 0, 0],
        [0, 1, 1],
        [0, 1, 0]])
    lo_mask = (correlate2d(mask, lo_tri_template, mode='same') == 3)
    up_mask = (correlate2d(mask, up_tri_template, mode='same') == 3)
    ma_mask = (correlate2d(mask, ma_tri_template, mode='same') == 3) & (~lo_mask) & (~up_mask)
    ref_mask = (
        lo_mask | (correlate2d(lo_mask, np.flip(lo_tri_template, (0,1)), mode='same') > 0) |\
        up_mask | (correlate2d(up_mask, np.flip(up_tri_template, (0,1)), mode='same') > 0) |\
        ma_mask | (correlate2d(ma_mask, np.flip(ma_tri_template, (0,1)), mode='same') > 0)
    )
    points = xyzrgb[ref_mask]

    ref_id = np.full(ref_mask.shape, -1, np.int32)
    ref_id[ref_mask] = np.arange(ref_mask.sum())
    faces_lo_tri = np.stack([
        ref_id[lo_mask],
        ref_id[shift(lo_mask, [1, 0], cval=False, order=0)],
        ref_id[shift(lo_mask, [1, 1], cval=False, order=0)],
    ], 1)
    faces_up_tri = np.stack([
        ref_id[up_mask],
        ref_id[shift(up_mask, [1, 1], cval=False, order=0)],
        ref_id[shift(up_mask, [0, 1], cval=False, order=0)],
    ], 1)
    faces_ma_tri = np.stack([
        ref_id[ma_mask],
        ref_id[shift(ma_mask, [1, 0], cval=False, order=0)],
        ref_id[shift(ma_mask, [0, 1], cval=False, order=0)],
    ], 1)
    faces = np.concatenate([faces_lo_tri, faces_up_tri, faces_ma_tri])
    
    points[:, [3, 5]] = points[:, [5, 3]] #Edited
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points[:, :3])
    mesh.vertex_colors = o3d.utility.Vector3dVector(points[:, 3:] / 255.)
    mesh.triangles = o3d.utility.Vector3iVector(faces)


    vertices = np.asarray(mesh.vertices)
    min_z = np.min(vertices[:, 2])
    shifted_vertices = vertices - [0, 0, min_z]

    current_height = np.max(shifted_vertices, axis=0)[2]  # Z dimension
    scaling_factor = height*1.2 / current_height

    scaled_vertices = shifted_vertices * scaling_factor    

    
    mesh.vertices = o3d.utility.Vector3dVector(scaled_vertices)


    draw_geometries = [mesh]


    return draw_geometries

def load_ply_mesh(file_path):
    plydata = PlyData.read(file_path)
    vertices = np.vstack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).T

    colors = np.vstack([plydata['vertex']['red'], plydata['vertex']['green'], plydata['vertex']['blue']]).T

    faces = np.vstack(plydata['face']['vertex_indices'])
    num_faces = len(plydata['face']['vertex_indices'])

    faces_tuple = np.zeros((num_faces,), dtype=[('vertex_indices', 'i4', (3,))])

    for i in range(0, num_faces):
        faces_tuple[i] = faces[i, :].tolist()

    return vertices, colors, faces_tuple

def shift_and_scale_mesh(vertices, colors, faces, desired_height):
    min_z = np.min(vertices[:, 2])
    shifted_vertices = vertices - [0, 0, min_z]
    current_height = np.max(shifted_vertices, axis=0)[2]  # Z dimension
    scaling_factor = desired_height*1.2 / current_height
    ori_factor = desired_height / current_height

    scaled_vertices = shifted_vertices * scaling_factor
    
    return scaled_vertices, colors, faces


def save_ply_mesh(output_file_path, vertices, colors, faces):
    vertex = np.array([tuple(v) + tuple(c) for v, c in zip(vertices, colors)], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    
    PlyData([PlyElement.describe(vertex, 'vertex'), PlyElement.describe(faces, 'face')], text=True).write(output_file_path)




def normalize_coordinates(corners, s=1000):
    # Step 1: Find minimum x and y coordinates
    min_x = min(coord[0] for coord in corners)
    min_y = min(coord[1] for coord in corners)

    # Step 2: Shift coordinates to make them non-negative
    shifted_coords = [(x - min_x, y - min_y) for x, y in corners]

    # Step 3: Find maximum x and y coordinates after shifting
    max_x = max(coord[0] for coord in shifted_coords)
    max_y = max(coord[1] for coord in shifted_coords)

    # Step 4: Scale coordinates to have maximum value of s
    scale_factor = max(max_x, max_y) / s
    normalized_coords = [(int(x / scale_factor), int(y / scale_factor)) for x, y in shifted_coords]

    return normalized_coords


def denormalize_coordinates(normalized_coords, corners, s=1000):
    
    # Step 1: Find minimum x and y coordinates
    min_x = min(coord[0] for coord in corners)
    min_y = min(coord[1] for coord in corners)

    # Step 2: Shift coordinates to make them non-negative
    shifted_coords = [(x - min_x, y - min_y) for x, y in corners]

    # Step 3: Find maximum x and y coordinates after shifting
    max_x = max(coord[0] for coord in shifted_coords)
    max_y = max(coord[1] for coord in shifted_coords)

    # Step 4: Scale coordinates to have maximum value of s
    scale_factor = max(max_x, max_y) / s

    # Step 5: Scale coordinates back to the original range
    scaled_coords = [(x * scale_factor, y * scale_factor) for x, y in normalized_coords]
    
    # Step 6: Shift coordinates back to their original positions
    denormalized_coords = [(x + min_x, y + min_y) for x, y in scaled_coords]
    
    return denormalized_coords
    
    # return denormalized_coords

def find_rectangle(corners):

    int_corners = normalize_coordinates(corners)

  
    rectangle = lir.lir(np.array([int_corners]))
    rectangle_corrdinates = [[rectangle[0],rectangle[1]], [rectangle[0]+rectangle[2],rectangle[1]], [rectangle[0]+rectangle[2],rectangle[1]+rectangle[3]], [rectangle[0],rectangle[1]+rectangle[3]]]
    # print(rectangle_corrdinates)
    denormalized_rectangle_coordinates = denormalize_coordinates(rectangle_corrdinates,corners)
    return denormalized_rectangle_coordinates

