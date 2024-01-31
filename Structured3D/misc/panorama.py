"""
Copy from https://github.com/sunset1995/pytorch-layoutnet/blob/master/pano.py
"""
import numpy as np
import numpy.matlib as matlib
import cv2
from engine.conversion import lonlat2uv, xyz2uv


def xyz_2_coorxy(xs, ys, zs, H=512, W=1024):
    us = np.arctan2(xs, ys)
    vs = -np.arctan(zs / np.sqrt(xs**2 + ys**2))
    coorx = (us / (2 * np.pi) + 0.5) * W
    coory = (vs / np.pi + 0.5) * H
    return coorx, coory


def coords2uv(coords, width, height):
    """
    Image coordinates (xy) to uv
    """
    middleX = width / 2 + 0.5
    middleY = height / 2 + 0.5
    uv = np.hstack([
        (coords[:, [0]] - middleX) / width * 2 * np.pi,
        -(coords[:, [1]] - middleY) / height * np.pi])
    
    # print(uv)
    return uv


def uv2xyzN(uv, planeID=1):
    #UNIT SPHERE
    ID1 = (int(planeID) - 1 + 0) % 3
    ID2 = (int(planeID) - 1 + 1) % 3
    ID3 = (int(planeID) - 1 + 2) % 3
    xyz = np.zeros((uv.shape[0], 3))
    xyz[:, ID1] = np.cos(uv[:, 1]) * np.sin(uv[:, 0])
    xyz[:, ID2] = np.cos(uv[:, 1]) * np.cos(uv[:, 0])
    xyz[:, ID3] = np.sin(uv[:, 1])
    return xyz


def uv2xyzN_vec(uv, planeID):
    """
    vectorization version of uv2xyzN
    @uv       N x 2
    @planeID  N
    """
    assert (planeID.astype(int) != planeID).sum() == 0
    planeID = planeID.astype(int)
    ID1 = (planeID - 1 + 0) % 3
    ID2 = (planeID - 1 + 1) % 3
    ID3 = (planeID - 1 + 2) % 3
    ID = np.arange(len(uv))
    xyz = np.zeros((len(uv), 3))
    xyz[ID, ID1] = np.cos(uv[:, 1]) * np.sin(uv[:, 0])
    xyz[ID, ID2] = np.cos(uv[:, 1]) * np.cos(uv[:, 0])
    xyz[ID, ID3] = np.sin(uv[:, 1])
    return xyz


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


def computeUVN_vec(n, in_, planeID):
    """
    vectorization version of computeUVN
    @n         N x 3
    @in_      MN x 1
    @planeID   N
    """
    n = n.copy()
    if (planeID == 2).sum():
        n[planeID == 2] = np.roll(n[planeID == 2], 2, axis=1)
    if (planeID == 3).sum():
        n[planeID == 3] = np.roll(n[planeID == 3], 1, axis=1)
    n = np.repeat(n, in_.shape[0] // n.shape[0], axis=0)
    assert n.shape[0] == in_.shape[0]
    bc = n[:, [0]] * np.sin(in_) + n[:, [1]] * np.cos(in_)
    bs = n[:, [2]]
    out = np.arctan(-bc / (bs + 1e-9))
    return out


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
    # print("!!!!!!!!!!!!!!!!!!!!!!!!!11",numLine, pt1.shape)
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
        # print("point1,", pt1[i], "point2,", pt2[i])
        uv = xyz2uvN(np.vstack([pt1[i, :], pt2[i, :]]), lines[i, 3])
        umax = uv[:, 0].max() + np.pi
        umin = uv[:, 0].min() + np.pi
        if umax - umin > np.pi:
            lines[i, 4:6] = np.array([umax, umin]) / 2 / np.pi
        else:
            lines[i, 4:6] = np.array([umin, umax]) / 2 / np.pi
    # print(lines[:,4:])
    return lines


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
    # print(contours[0])

    # Allocate colors based on the vertical position of the segments
    result = np.zeros_like(panoEdgeC)
    for i, contour in enumerate(contours):
        if i == 0:
            # color = [183, 71, 78]  # Red for the highest segment
            color = [78, 71, 183]  # Red for the highest segment #RGB
        elif i == len(contours) - 1:
            # color = [138, 223, 152]  # Blue for the lowest segment
            color = [152, 223, 138]  # Blue for the lowest segment #RGB
        else:
            # color = [232, 199, 174]  # Green for the rest of the segments
            color = [174, 199, 232]  # Green for the rest of the segments #RGB

        cv2.drawContours(result, [contour], -1, color, thickness=cv2.FILLED)
    # Apply morphological closing to fill the gaps between regions
    kernel_closing = np.ones((15, 15), np.uint8)
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel_closing)   
    return panoEdgeC, result


def coorx2u(x, w=1024):
    return ((x + 0.5) / w - 0.5) * 2 * np.pi


def coory2v(y, h=512):
    return ((y + 0.5) / h - 0.5) * np.pi


def u2coorx(u, w=1024):
    return (u / (2 * np.pi) + 0.5) * w - 0.5


def v2coory(v, h=512):
    return (v / np.pi + 0.5) * h - 0.5


def uv2xy(u, v, z=-50):
    c = z / np.tan(v)
    x = c * np.cos(u)
    y = c * np.sin(u)
    return x, y


def pano_connect_points(p1, p2, z=-50, w=1024, h=512):
    u1 = coorx2u(p1[0], w)
    v1 = coory2v(p1[1], h)
    u2 = coorx2u(p2[0], w)
    v2 = coory2v(p2[1], h)

    x1, y1 = uv2xy(u1, v1, z)
    x2, y2 = uv2xy(u2, v2, z)

    if abs(p1[0] - p2[0]) < w / 2:
        pstart = np.ceil(min(p1[0], p2[0]))
        pend = np.floor(max(p1[0], p2[0]))
    else:
        pstart = np.ceil(max(p1[0], p2[0]))
        pend = np.floor(min(p1[0], p2[0]) + w)
    coorxs = (np.arange(pstart, pend + 1) % w).astype(np.float64)
    vx = x2 - x1
    vy = y2 - y1
    us = coorx2u(coorxs, w)
    ps = (np.tan(us) * x1 - y1) / (vy - np.tan(us) * vx)
    cs = np.sqrt((x1 + ps * vx) ** 2 + (y1 + ps * vy) ** 2)
    vs = np.arctan2(z, cs)
    coorys = v2coory(vs)

    return np.stack([coorxs, coorys], axis=-1)
##############################################################################
# import numpy as np
# import cv2


def find_contour_corners(contour, uv_xyz_dict, tolerance=10):
    # Convert the contour to a numpy array
    contour = np.squeeze(contour)

    # Find left-top, left-bottom, right-top, and right-bottom corners
    left_top = min(contour, key=lambda point: (point[0], -point[1]))
    left_bottom = min(contour, key=lambda point: (point[0], point[1]))
    right_top = max(contour, key=lambda point: (point[0], point[1]))
    right_bottom = max(contour, key=lambda point: (point[0], -point[1]))

    # print("#######################UV############################")
    # print("Left-Top Corner:", left_top)
    # print("Left-Bottom Corner:", left_bottom)
    # print("Right-Top Corner:", right_top)
    # print("Right-Bottom Corner:", right_bottom)


    # Find left-top, left-bottom, right-top, and right-bottom corners in XYZ coordinates
    left_top_xyz = find_matching_xyz(uv_xyz_dict, tuple(left_top), tolerance)
    left_bottom_xyz = find_matching_xyz(uv_xyz_dict, tuple(left_bottom), tolerance)
    right_top_xyz = find_matching_xyz(uv_xyz_dict, tuple(right_top), tolerance)
    right_bottom_xyz = find_matching_xyz(uv_xyz_dict, tuple(right_bottom), tolerance)
    
    # print("#######################XYZ############################")
    # print("Left-Top Corner:", left_top_xyz)
    # print("Left-Bottom Corner:", left_bottom_xyz)
    # print("Right-Top Corner:", right_top_xyz)
    # print("Right-Bottom Corner:", right_bottom_xyz)

    return np.array([left_top_xyz, left_bottom_xyz, right_top_xyz, right_bottom_xyz])


def find_matching_xyz(uv_xyz_dict, uv, tolerance):
    for key in uv_xyz_dict:
        if all(abs(key[i] - uv[i]) < tolerance for i in range(len(uv))):
            return uv_xyz_dict[key]
    raise ValueError(f"No matching value found for UV {uv}")

def interpolate_points(p1, p2, num_points):
    """Interpolate num_points evenly spaced points between p1 and p2."""
    return [p1 + t*(p2 - p1) for t in np.linspace(0, 1, num_points)]


def draw_window(image, selected_contour, uv_xyz_dict):
    # Find xyz coordinates of corners of the selected wall
    wall_corners_xyz = find_contour_corners(selected_contour, uv_xyz_dict)
    # print("############################################Window#################################################")
    # print("wall corners:", wall_corners_xyz)
    # Calculating the center of the plane
    center = (wall_corners_xyz[0] + wall_corners_xyz[1] + wall_corners_xyz[2] + wall_corners_xyz[3]) / 4

    # Calculating the dimensions of the plane
    width = np.linalg.norm(wall_corners_xyz[2] - wall_corners_xyz[0]) #right_top - left_top
    height = np.linalg.norm(wall_corners_xyz[0] - wall_corners_xyz[1]) #left_top - left_bottom
    # print("wall_width",width)
    # print("wall_height",height)



    # Generating random dimensions for the window
    window_width = np.random.uniform(0.4, 0.5) * width
    window_height = np.random.uniform(0.3, 0.5) * height

    # print("window_width",window_width)
    # print("window_height",window_height)

    # Ensure the window does not exceed or touch the boundaries of the wall
    # Setting a minimum gap from the edges
    min_gap = 0.05  # This can be adjusted as needed
    max_window_width = width - 2 * min_gap
    max_window_height = height - 2 * min_gap

    # # Adjust if the window size exceeds the maximum allowed size
    # window_width = min(window_width, max_window_width)
    # window_height = min(window_height, max_window_height)

    # Randomly adjust the vertical position
    # The vertical shift should be such that the window remains within the wall boundaries
    # max_vertical_shift = (height - window_height) / 2 - min_gap
    max_vertical_shift = 0.5
    vertical_shift = np.random.uniform(0.1, max_vertical_shift)

    # Calculating the corners of the window
    half_window_width = window_width / 2
    half_window_height = window_height / 2

    # Calculating the corners of the square
    # Assuming the square is parallel to the sides of the rectangular plane
    right_top = wall_corners_xyz[2]
    left_top = wall_corners_xyz[0]
    right_bottom = wall_corners_xyz[3]
    left_bottom = wall_corners_xyz[1]

    # print("right_top", right_top)
    # print("left_top", left_top)
    # print("right_bottom", right_bottom)
    # print("left_bottom", left_bottom)


    v1 = (right_top - left_top) / np.linalg.norm(right_top - left_top) # direction vector along the top edge
    v2 = (left_bottom - left_top) / np.linalg.norm(left_bottom - left_top) # direction vector 
    
    # print("vertical_shift", vertical_shift)
    # print("v1", v1)
    # print("v2", v2)

    # print("half_window_height + vertical_shift", half_window_height + vertical_shift)
    window_top_left = center + half_window_width * -v1 + (half_window_height + vertical_shift) * v2
    window_top_right = center + half_window_width * v1 + (half_window_height + vertical_shift) * v2
    window_bottom_left = center + half_window_width * -v1 + (-half_window_height + vertical_shift) * v2
    window_bottom_right = center + half_window_width * v1 + (-half_window_height + vertical_shift) * v2

    # print("window_top_left", window_top_left)
    # print(" window_top_right",  window_top_right)
    # print("window_bottom_left", window_bottom_left)
    # print(" window_bottom_right",  window_bottom_right)

    # Increased number of points per edge for denser representation
    num_points_per_edge_dense = 100

    # Interpolating dense points along each edge of the window
    top_edge_dense = interpolate_points(window_top_left, window_top_right, num_points_per_edge_dense)
    right_edge_dense = interpolate_points(window_top_right, window_bottom_right, num_points_per_edge_dense)
    bottom_edge_dense = interpolate_points(window_bottom_right, window_bottom_left, num_points_per_edge_dense)
    left_edge_dense = interpolate_points(window_bottom_left, window_top_left, num_points_per_edge_dense)

    # Combine all dense points (excluding the duplicates at the corners)
    window_points_xyz = top_edge_dense + right_edge_dense[1:] + bottom_edge_dense[1:] + left_edge_dense[1:]

    return window_points_xyz

def draw_door(image, selected_contour, uv_xyz_dict):
    # Find xyz coordinates of corners of the selected wall
    wall_corners_xyz = find_contour_corners(selected_contour, uv_xyz_dict)
    # print("############################################Door#################################################")
    # print("wall corners:", wall_corners_xyz)
    # Calculating the center of the plane
    center = (wall_corners_xyz[0] + wall_corners_xyz[1] + wall_corners_xyz[2] + wall_corners_xyz[3]) / 4

    # Calculating the dimensions of the plane
    width = np.linalg.norm(wall_corners_xyz[2] - wall_corners_xyz[0]) #right_top - left_top
    height = np.linalg.norm(wall_corners_xyz[0] - wall_corners_xyz[1]) #left_top - left_bottom
    # print("wall_width",width)
    # print("wall_height",height)

    # Generating dimensions for the door
    door_width = np.random.uniform(0.2, 0.3) * width
    door_height = 0.7 * height
    # print("door_width",door_width)
    # print("door_height",door_height)

    # Calculating the corners of the door
    half_door_width = door_width / 2

    # Assuming the door is parallel to the sides of the rectangular plane
    right_top = wall_corners_xyz[2]
    left_top = wall_corners_xyz[0]
    left_bottom = wall_corners_xyz[1]
    right_bottom = wall_corners_xyz[3]

    v1 = (right_top - left_top) / np.linalg.norm(right_top - left_top) # direction vector along the top edge
    v2 = (left_bottom - left_top) / np.linalg.norm(left_bottom - left_top)

    # print("v1", v1)
    # print("v2", v2)
    # window_top_left = center + half_window_width * -v1 + (half_window_height + vertical_shift) * v2
    # window_top_right = center + half_window_width * v1 + (half_window_height + vertical_shift) * v2
    # window_bottom_left = center + half_window_width * -v1 + (-half_window_height + vertical_shift) * v2
    # window_bottom_right = center + half_window_width * v1 + (-half_window_height + vertical_shift) * v2
    
    center = (wall_corners_xyz[0] + wall_corners_xyz[1] + wall_corners_xyz[2] + wall_corners_xyz[3]) / 4


    half_door_width = door_width/2
    half_door_height = door_height/2
    vertical_shift = - (height/2 - half_door_height)

    door_top_left = center + half_door_width * -v1 + (half_door_height + vertical_shift) * v2
    door_top_right = center + half_door_width * v1 + (half_door_height + vertical_shift) * v2
    door_bottom_left = center + half_door_width * -v1 + (-half_door_height + vertical_shift) * v2
    door_bottom_right = center + half_door_width * v1 + (-half_door_height + vertical_shift) * v2


    # print("door_top_left", door_top_left)
    # print("door_top_right", door_top_right)
    # print("door_bottom_left", door_bottom_left)
    # print("door_bottom_right", door_bottom_right)

    # Increased number of points per edge for denser representation
    num_points_per_edge_dense = 100

    # Interpolating dense points along each edge of the door
    top_edge_dense = interpolate_points(door_top_left, door_top_right, num_points_per_edge_dense)
    right_edge_dense = interpolate_points(door_top_right, door_bottom_right, num_points_per_edge_dense)
    bottom_edge_dense = interpolate_points(door_bottom_right, door_bottom_left, num_points_per_edge_dense)
    left_edge_dense = interpolate_points(door_bottom_left, door_top_left, num_points_per_edge_dense)

    # Combine all dense points (excluding the duplicates at the corners)
    door_points_xyz = top_edge_dense + right_edge_dense[1:] + bottom_edge_dense[1:] + left_edge_dense[1:]

    return door_points_xyz



def draw(image, wall_contours, uv_xyz_dict, door, window):
   # Check if the combined number of windows and doors exceeds the length of wall_contours
    if window + door > len(wall_contours):
        raise ValueError("Not enough walls! Decrease the number of windows or doors.")
    
    window_color = (197, 176, 213)
    door_color = (214, 39, 40)
    
    n_window = window
    window_wall_index = []
    if window:
        # Randomly choose a wall index to add a window
        while True:
            window_wall = np.random.randint(0, len(wall_contours))
            if not n_window:
                break
            if window_wall not in window_wall_index:
                window_wall_index.append(window_wall)
                n_window -= 1
        window_selected_contours = [wall_contours[n] for n in window_wall_index]
        for window_selected_contour in window_selected_contours:
            window_points_xyz = draw_window(image, window_selected_contour, uv_xyz_dict)
            window_points_uv = xyz2uv(window_points_xyz) * np.array([1026, 512])
            window_hull = cv2.convexHull(window_points_uv.astype(np.int32)) # Calculate the convex hull
            cv2.drawContours(image, [window_hull], -1, window_color, thickness=cv2.FILLED) # Draw and fill the convex hull on the image
    
    door_wall_index = []
    n_door = door
    if door:
        # Randomly choose a wall index to add a door
        while True:
            door_wall = np.random.randint(0, len(wall_contours))
            if not n_door:
                break
            if door_wall not in window_wall_index and door_wall not in door_wall_index:
                door_wall_index.append(door_wall)
                n_door -= 1
        door_selected_contours = [wall_contours[n] for n in door_wall_index]
        for door_selected_contour in door_selected_contours:
            door_points_xyz = draw_door(image, door_selected_contour, uv_xyz_dict)
            door_points_uv = xyz2uv(door_points_xyz) * np.array([1026, 512])
            door_hull = cv2.convexHull(door_points_uv.astype(np.int32)) # Calculate the convex hull
            cv2.drawContours(image, [door_hull], -1, door_color, thickness=cv2.FILLED) # Draw and fill the convex hull on the image  

    return image


# def draw_boundary_from_cor_id_window(cor_id, img_src, uv_xyz_dict):
#     im_h, im_w = img_src.shape[:2]
#     cor_all = [cor_id]  # edited
    
#     for i in range(len(cor_id)):
#         cor_all.append(cor_id[i, :])
#         cor_all.append(cor_id[(i+2) % len(cor_id), :]) #Appends the coordinate that is two indices ahead in a circular manner. This ensures that the last coordinate is connected to the first one, forming a closed shape
#     cor_all = np.vstack(cor_all) #Stacks all the coordinates in cor_all vertically to create a 2D array. Each row of this array represents a pair of connected coordinates.
    
#     # print("COR_id shape", cor_id.shape)
#     # print("COR_ALL shape", cor_all.shape)
#     # print('COR_ID', cor_id)
#     # print('COR_ALL', cor_all)
#     rs, cs = lineIdxFromCors(cor_all, im_w, im_h)
#     rs = np.array(rs)
#     cs = np.array(cs)

#     import os
#     panoEdgeC = img_src.astype(np.uint8)
#     panoEdgeC[np.clip(rs, 0, im_h - 1), np.clip(cs, 0, im_w - 1), 0] = 255
#     panoEdgeC[np.clip(rs, 0, im_h - 1), np.clip(cs, 0, im_w - 1), 1] = 255
#     panoEdgeC[np.clip(rs, 0, im_h - 1), np.clip(cs, 0, im_w - 1), 2] = 255
    
#     # Convert the image to grayscale
#     gray = cv2.cvtColor(panoEdgeC, cv2.COLOR_BGR2GRAY)

#     # Apply GaussianBlur to the grayscale image
#     blur = cv2.GaussianBlur(gray, (5, 5), 0)

#     # Apply adaptive thresholding to obtain a binary image
#     _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
#     # Perform morphological operations to clean up the binary image
#     kernel = np.ones((3, 3), np.uint8)
#     opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
#     # Find contours in the binary image
#     contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     # Sort contours based on their bounding box's top-left y-coordinate
#     contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])
#     wall_contours = sorted(contours[1:-1], key=lambda c: cv2.boundingRect(c)[0])

#     # Allocate colors based on the vertical position of the segments
#     result = np.zeros_like(panoEdgeC)

#     for i, contour in enumerate(contours):
#         if i == 0:
#             color = [78, 71, 183]  # Red for the highest segment #RGB
#         elif i == len(contours) - 1:
#             color = [152, 223, 138]  # Blue for the lowest segment #RGB
#         else:
#             color = [174, 199, 232]  # Green for the rest of the segments #RGB

#         cv2.drawContours(result, [contour], -1, color, thickness=cv2.FILLED)

#     # Apply morphological closing to fill the gaps between regions
#     kernel_closing = np.ones((15, 15), np.uint8)
#     result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel_closing)   
#     # Draw the window on the result image
#     result_with_window = draw(result, im_w, im_h, wall_contours[1:-1], uv_xyz_dict)

#     return panoEdgeC, result_with_window

def draw_boundary_from_cor_id(cor_id, img_src):
    im_h, im_w = img_src.shape[:2]
    cor_all = [cor_id]  # edited
    for i in range(len(cor_id)):
        cor_all.append(cor_id[i, :])
        cor_all.append(cor_id[(i+2) % len(cor_id), :]) #Appends the coordinate that is two indices ahead in a circular manner. This ensures that the last coordinate is connected to the first one, forming a closed shape
    cor_all = np.vstack(cor_all) #Stacks all the coordinates in cor_all vertically to create a 2D array. Each row of this array represents a pair of connected coordinates.
    
    rs, cs = lineIdxFromCors(cor_all, im_w, im_h)
    rs = np.array(rs)
    cs = np.array(cs)

    import os
    panoEdgeC = img_src.astype(np.uint8)
    panoEdgeC[np.clip(rs, 0, im_h - 1), np.clip(cs, 0, im_w - 1), 0] = 255
    panoEdgeC[np.clip(rs, 0, im_h - 1), np.clip(cs, 0, im_w - 1), 1] = 255
    panoEdgeC[np.clip(rs, 0, im_h - 1), np.clip(cs, 0, im_w - 1), 2] = 255
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(panoEdgeC, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to the grayscale image
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding to obtain a binary image
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Perform morphological operations to clean up the binary image
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Find contours in the binary image
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Sort contours based on their bounding box's top-left y-coordinate
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])
    wall_contours = sorted(contours[1:-1], key=lambda c: cv2.boundingRect(c)[0])
    # Allocate colors based on the vertical position of the segments
    result = np.zeros_like(panoEdgeC)

    for i, contour in enumerate(contours):
        if i == 0:
            color = [78, 71, 183]  # Red for the highest segment #RGB
        elif i == len(contours) - 1:
            color = [152, 223, 138]  # Blue for the lowest segment #RGB
        else:
            color = [174, 199, 232]  # Green for the rest of the segments #RGB

        cv2.drawContours(result, [contour], -1, color, thickness=cv2.FILLED)

    # Apply morphological closing to fill the gaps between regions
    kernel_closing = np.ones((15, 15), np.uint8)
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel_closing)   
    # # Draw the window on the result image
    # result_with_window = draw(result, wall_contours[1:-1], uv_xyz_dict)

    return panoEdgeC, result, wall_contours[1:-1]
