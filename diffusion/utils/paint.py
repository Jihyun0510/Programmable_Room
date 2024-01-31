import os
import json
import argparse

import cv2
import numpy as np
import numpy.matlib as matlib
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from descartes.patch import PolygonPatch

from tqdm import tqdm
from glob import glob

def visualize_panorama(args):
    """visualize panorama layout
    """
    scene_path = os.path.join(args.path, f"scene_{args.scene:05d}", "2D_rendering")

    for room_id in np.sort(os.listdir(scene_path)):
        room_path = os.path.join(scene_path, room_id, "panorama")

        cor_id = np.loadtxt(os.path.join(room_path, "layout.txt"))
        img_src = cv2.imread(os.path.join(room_path, "full", "rgb_rawlight.png"))
        img_src = cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB)
        img_viz = draw_boundary_from_cor_id(cor_id, img_src)

        plt.axis('off')
        plt.imshow(img_viz)
        plt.show()

def coords2uv(coords, width, height):
    """
    Image coordinates (xy) to uv
    """
    middleX = width / 2 + 0.5
    middleY = height / 2 + 0.5
    uv = np.hstack([
        (coords[:, [0]] - middleX) / width * 2 * np.pi,
        -(coords[:, [1]] - middleY) / height * np.pi])
    return uv

def uv2xyzN(uv, planeID=1):
    ID1 = (int(planeID) - 1 + 0) % 3
    ID2 = (int(planeID) - 1 + 1) % 3
    ID3 = (int(planeID) - 1 + 2) % 3
    xyz = np.zeros((uv.shape[0], 3))
    xyz[:, ID1] = np.cos(uv[:, 1]) * np.sin(uv[:, 0])
    xyz[:, ID2] = np.cos(uv[:, 1]) * np.cos(uv[:, 0])
    xyz[:, ID3] = np.sin(uv[:, 1])
    return xyz

def lineFromTwoPoint(pt1, pt2):
    """
    Generate line segment based on two points on panorama
    pt1, pt2: two points on panorama
    line:
        1~3-th dim: normal of the line
        4-th dim: the projection dimension ID
        5~6-th dim: the u of line segment endpoints in projection plane
    """
    numLine = pt1.shape[0]
    lines = np.zeros((numLine, 6))
    n = np.cross(pt1, pt2)
    n = n / (matlib.repmat(np.sqrt(np.sum(n ** 2, 1, keepdims=True)), 1, 3) + 1e-9)
    lines[:, 0:3] = n

    areaXY = np.abs(np.sum(n * matlib.repmat([0, 0, 1], numLine, 1), 1, keepdims=True))
    areaYZ = np.abs(np.sum(n * matlib.repmat([1, 0, 0], numLine, 1), 1, keepdims=True))
    areaZX = np.abs(np.sum(n * matlib.repmat([0, 1, 0], numLine, 1), 1, keepdims=True))
    planeIDs = np.argmax(np.hstack([areaXY, areaYZ, areaZX]), axis=1) + 1
    lines[:, 3] = planeIDs

    for i in range(numLine):
        uv = xyz2uvN(np.vstack([pt1[i, :], pt2[i, :]]), lines[i, 3])
        umax = uv[:, 0].max() + np.pi
        umin = uv[:, 0].min() + np.pi
        if umax - umin > np.pi:
            lines[i, 4:6] = np.array([umax, umin]) / 2 / np.pi
        else:
            lines[i, 4:6] = np.array([umin, umax]) / 2 / np.pi

    return lines

def computeUVN(n, in_, planeID):
    """
    compute v given u and normal.
    """
    if planeID == 2:
        n = np.array([n[1], n[2], n[0]])
    elif planeID == 3:
        n = np.array([n[2], n[0], n[1]])
    bc = n[0] * np.sin(in_) + n[1] * np.cos(in_)
    bs = n[2]
    out = np.arctan(-bc / (bs + 1e-9))
    return out

def xyz2uvN(xyz, planeID=1):
    ID1 = (int(planeID) - 1 + 0) % 3
    ID2 = (int(planeID) - 1 + 1) % 3
    ID3 = (int(planeID) - 1 + 2) % 3
    normXY = np.sqrt(xyz[:, [ID1]] ** 2 + xyz[:, [ID2]] ** 2)
    normXY[normXY < 0.000001] = 0.000001
    normXYZ = np.sqrt(xyz[:, [ID1]] ** 2 + xyz[:, [ID2]] ** 2 + xyz[:, [ID3]] ** 2)
    v = np.arcsin(xyz[:, [ID3]] / normXYZ)
    u = np.arcsin(xyz[:, [ID1]] / normXY)
    valid = (xyz[:, [ID2]] < 0) & (u >= 0)
    u[valid] = np.pi - u[valid]
    valid = (xyz[:, [ID2]] < 0) & (u <= 0)
    u[valid] = -np.pi - u[valid]
    uv = np.hstack([u, v])
    uv[np.isnan(uv[:, 0]), 0] = 0
    return uv

def lineIdxFromCors(cor_all, im_w, im_h):
    assert len(cor_all) % 2 == 0
    uv = coords2uv(cor_all, im_w, im_h)
    xyz = uv2xyzN(uv)
    lines = lineFromTwoPoint(xyz[0::2], xyz[1::2])
    num_sample = max(im_h, im_w)

    cs, rs = [], []
    for i in range(lines.shape[0]):
        n = lines[i, 0:3]
        sid = lines[i, 4] * 2 * np.pi
        eid = lines[i, 5] * 2 * np.pi
        if eid < sid:
            x = np.linspace(sid, eid + 2 * np.pi, num_sample)
            x = x % (2 * np.pi)
        else:
            x = np.linspace(sid, eid, num_sample)

        u = -np.pi + x.reshape(-1, 1)
        v = computeUVN(n, u, lines[i, 3])
        xyz = uv2xyzN(np.hstack([u, v]), lines[i, 3])
        uv = xyz2uvN(xyz, 1)

        r = np.minimum(np.floor((uv[:, 0] + np.pi) / (2 * np.pi) * im_w) + 1,
                       im_w).astype(np.int32)
        c = np.minimum(np.floor((np.pi / 2 - uv[:, 1]) / np.pi * im_h) + 1,
                       im_h).astype(np.int32)
        cs.extend(r - 1)
        rs.extend(c - 1)
    return rs, cs

def draw_boundary_from_cor_id(cor_id, img_src):
    im_h, im_w = img_src.shape[:2]
    cor_all = [cor_id] #edited
    # cor_all = cor_id
    for i in range(len(cor_id)):
        # print(cor_id[i])
        cor_all.append(cor_id[i, :])
        cor_all.append(cor_id[(i+2) % len(cor_id), :])
    cor_all = np.vstack(cor_all)

    rs, cs = lineIdxFromCors(cor_all, im_w, im_h)
    rs = np.array(rs)
    cs = np.array(cs)

    panoEdgeC = img_src.astype(np.uint8)
    # for dx, dy in [[-1, 0], [1, 0], [0, 0], [0, 1], [0, -1]]:
    # for dx, dy in [[0, 0]]:
        # panoEdgeC[np.clip(rs + dx, 0, im_h - 1), np.clip(cs + dy, 0, im_w - 1), 0] = 0
        # panoEdgeC[np.clip(rs + dx, 0, im_h - 1), np.clip(cs + dy, 0, im_w - 1), 1] = 0
        # panoEdgeC[np.clip(rs + dx, 0, im_h - 1), np.clip(cs + dy, 0, im_w - 1), 2] = 255

    panoEdgeC[np.clip(rs, 0, im_h - 1), np.clip(cs, 0, im_w - 1), 0] = 0
    panoEdgeC[np.clip(rs, 0, im_h - 1), np.clip(cs, 0, im_w - 1), 1] = 0
    panoEdgeC[np.clip(rs, 0, im_h - 1), np.clip(cs, 0, im_w - 1), 2] = 255

    # Convert the image to grayscale
    gray = cv2.cvtColor(panoEdgeC, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to the grayscale image
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding to obtain binary image
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Perform morphological operations to clean up the binary image
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Find contours in the binary image
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(len(contours))

    # Sort contours based on their bounding box's top-left y-coordinate
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])

    # Allocate colors based on the vertical position of the segments
    result = np.zeros_like(panoEdgeC)
    for i, contour in enumerate(contours):
        if i == 0:
            color = [183, 71, 78]  # Red for the highest segment
        elif i == len(contours) - 1:
            color = [138, 223, 152]  # Blue for the lowest segment
        else:
            color = [232, 199, 174]  # Green for the rest of the segments

        cv2.drawContours(result, [contour], -1, color, thickness=cv2.FILLED)
    # Apply morphological closing to fill the gaps between regions
    kernel_closing = np.ones((15, 15), np.uint8)
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel_closing)   
    return panoEdgeC, result

def layout2semantic(coord):
    """visualize panorama layout
    """

    coord_npy = np.loadtxt(coord)
    img_src = np.zeros((512,1024,3), dtype=np.uint8)
    img_layout, img_semantic = draw_boundary_from_cor_id(coord_npy, img_src)


    cv2.imwrite("/workspace/room/uni_v9/utils/test_layout.png", img_layout)
    cv2.imwrite("/workspace/room/uni_v9/utils/test_semantic.png", img_semantic)
    print(img_semantic.shape)
    return img_layout, img_semantic

