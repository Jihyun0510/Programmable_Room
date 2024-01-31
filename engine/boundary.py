import numpy as np
import math
import cv2
from engine.conversion import xyz2uv, uv2xyz, uv2pixel
from functools import cmp_to_key as ctk


def is_ceil_boundary(corners: np.ndarray) -> bool:
    m = corners[..., 1].max()
    return m < 0.5


def is_floor_boundary(corners: np.ndarray) -> bool:
    m = corners[..., 1].min()
    return m > 0.5


def boundary_type(corners: np.ndarray) -> int:
    """
    Returns the boundary type that also represents the projection plane
    :param corners:
    :return:
    """
    if is_ceil_boundary(corners):
        plan_y = -1
    elif is_floor_boundary(corners):
        plan_y = 1
    else:
        # An intersection occurs and an exception is considered
        assert False, 'corners error!'
    return plan_y

def polygon_to_segments(polygon: np.array) -> np.array:
    segments = []
    polygon = np.concatenate((polygon, [polygon[0]]))
    for i in range(len(polygon) - 1):
        p1 = polygon[i]
        p2 = polygon[i + 1]
        segments.append([p1, p2])
    segments = np.array(segments)
    return segments

class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

class EndPoint(Point):
    def __init__(self, x: float, y: float, begins_segment: bool = None, segment=None, angle: float = None):
        super().__init__(x, y)
        self.begins_segment = begins_segment
        self.segment = segment
        self.angle = angle

class Segment:
    def __init__(self, x1: float, y1: float, x2: float, y2: float, d: float = None):
        self.p1 = EndPoint(x1, y1)
        self.p2 = EndPoint(x2, y2)
        self.p1.segment = self
        self.p2.segment = self
        self.d = d


def calculate_end_point_angles(light_source: Point, segment: Segment) -> None:
    x = light_source.x
    y = light_source.y
    dx = 0.5 * (segment.p1.x + segment.p2.x) - x
    dy = 0.5 * (segment.p1.y + segment.p2.y) - y
    segment.d = (dx * dx) + (dy * dy)
    segment.p1.angle = math.atan2(segment.p1.y - y, segment.p1.x - x)
    segment.p2.angle = math.atan2(segment.p2.y - y, segment.p2.x - x)

def set_segment_beginning(segment: Segment) -> None:
    d_angle = segment.p2.angle - segment.p1.angle
    if d_angle <= -math.pi:
        d_angle += 2 * math.pi
    if d_angle > math.pi:
        d_angle -= 2 * math.pi
    segment.p1.begins_segment = d_angle > 0
    segment.p2.begins_segment = not segment.p1.begins_segment


def endpoint_compare(point_a: EndPoint, point_b: EndPoint):
    if point_a.angle > point_b.angle:
        return 1
    if point_a.angle < point_b.angle:
        return -1
    if not point_a.begins_segment and point_b.begins_segment:
        return 1
    if point_a.begins_segment and not point_b.begins_segment:
        return -1
    return 0

def segment_in_front_of(segment_a: Segment, segment_b: Segment, relative_point: Point):
    def left_of(segment: Segment, point: Point):
        cross = (segment.p2.x - segment.p1.x) * (point.y - segment.p1.y) - (segment.p2.y - segment.p1.y) * (
                point.x - segment.p1.x)
        return cross < 0

    def interpolate(point_a: Point, point_b: Point, f: float):
        point = Point(x=point_a.x * (1 - f) + point_b.x * f,
                      y=point_a.y * (1 - f) + point_b.y * f)
        return point

    a1 = left_of(segment_a, interpolate(segment_b.p1, segment_b.p2, 0.01))
    a2 = left_of(segment_a, interpolate(segment_b.p2, segment_b.p1, 0.01))
    a3 = left_of(segment_a, relative_point)
    b1 = left_of(segment_b, interpolate(segment_a.p1, segment_a.p2, 0.01))
    b2 = left_of(segment_b, interpolate(segment_a.p2, segment_a.p1, 0.01))
    b3 = left_of(segment_b, relative_point)
    if b1 == b2 and not (b2 == b3):
        return True
    if a1 == a2 and a2 == a3:
        return True
    if a1 == a2 and not (a2 == a3):
        return False
    if b1 == b2 and b2 == b3:
        return False
    return False

def line_intersection(point1: Point, point2: Point, point3: Point, point4: Point):
    a = (point4.y - point3.y) * (point2.x - point1.x) - (point4.x - point3.x) * (point2.y - point1.y)
    b = (point4.x - point3.x) * (point1.y - point3.y) - (point4.y - point3.y) * (point1.x - point3.x)
    assert a != 0 or a == b, "center on polygon, it not support!"
    if a == 0:
        s = 1
    else:
        s = b / a

    return Point(
        point1.x + s * (point2.x - point1.x),
        point1.y + s * (point2.y - point1.y)
    )

def get_triangle_points(origin: Point, angle1: float, angle2: float, segment: Segment):
    p1 = origin
    p2 = Point(origin.x + math.cos(angle1), origin.y + math.sin(angle1))
    p3 = Point(0, 0)
    p4 = Point(0, 0)

    if segment:
        p3.x = segment.p1.x
        p3.y = segment.p1.y
        p4.x = segment.p2.x
        p4.y = segment.p2.y
    else:
        p3.x = origin.x + math.cos(angle1) * 2000
        p3.y = origin.y + math.sin(angle1) * 2000
        p4.x = origin.x + math.cos(angle2) * 2000
        p4.y = origin.y + math.sin(angle2) * 2000

    #  use the endpoint directly when the rays are parallel to segment
    if abs(segment.p1.angle - segment.p2.angle) < 1e-6:
        return [p4, p3]

    # it's maybe generate error coordinate when the rays are parallel to segment
    p_begin = line_intersection(p3, p4, p1, p2)
    p2.x = origin.x + math.cos(angle2)
    p2.y = origin.y + math.sin(angle2)
    p_end = line_intersection(p3, p4, p1, p2)

    return [p_begin, p_end]

def calc_visible_polygon(center: np.array, polygon: np.array = None, segments: np.array = None, show: bool = False):
    if segments is None and polygon is not None:
        segments = polygon_to_segments(polygon)

    origin = Point(x=center[0], y=center[1])
    endpoints = []
    for s in segments:
        p1 = s[0]
        p2 = s[1]
        segment = Segment(x1=p1[0], y1=p1[1], x2=p2[0], y2=p2[1])
        calculate_end_point_angles(origin, segment)
        set_segment_beginning(segment)
        endpoints.extend([segment.p1, segment.p2])

    open_segments = []
    output = []
    begin_angle = 0
    endpoints = sorted(endpoints, key=ctk(endpoint_compare))

    for pas in range(2):
        for endpoint in endpoints:
            open_segment = open_segments[0] if len(open_segments) else None
            if endpoint.begins_segment:
                index = 0
                segment = open_segments[index] if index < len(open_segments) else None
                while segment and segment_in_front_of(endpoint.segment, segment, origin):
                    index += 1
                    segment = open_segments[index] if index < len(open_segments) else None

                if not segment:
                    open_segments.append(endpoint.segment)
                else:
                    open_segments.insert(index, endpoint.segment)
            else:
                if endpoint.segment in open_segments:
                    open_segments.remove(endpoint.segment)

            if open_segment is not (open_segments[0] if len(open_segments) else None):
                if pas == 1 and open_segment:
                    triangle_points = get_triangle_points(origin, begin_angle, endpoint.angle, open_segment)
                    output.extend(triangle_points)
                begin_angle = endpoint.angle

    output_polygon = []
    # Remove duplicate
    for i, p in enumerate(output):
        q = output[(i + 1) % len(output)]
        if int(p.x * 10000) == int(q.x * 10000) and int(p.y * 10000) == int(q.y * 10000):
            continue
        output_polygon.append([p.x, p.y])

    output_polygon.reverse()
    output_polygon = np.array(output_polygon)

    return output_polygon

def visibility_corners(corners):
    plan_y = boundary_type(corners)
    # print("PLAN_Y", plan_y)
    # print("CORNERS", corners)
    xyz = uv2xyz(corners, plan_y)
    # print("@@@@@@@@@@@@@@@@@@@@@@@@@@", corners)
    # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", xyz)
    xz = xyz[:, ::2]
    # print("##############################", xz)
    xz = calc_visible_polygon(center=np.array([0, 0]), polygon=xz, show=False)
    xyz = np.insert(xz, 1, plan_y, axis=1)
    output = xyz2uv(xyz).astype(np.float32)
    # print(output)
    return output







def connect_corners_uv(uv1: np.ndarray, uv2: np.ndarray, length=256) -> np.ndarray:
    """
    :param uv1: [u, v]
    :param uv2: [u, v]
    :param length: Fix the total length in pixel coordinates
    :return:
    """
    # why -0.5? Check out the uv2Pixel function
    p_u1 = uv1[0] * length - 0.5
    p_u2 = uv2[0] * length - 0.5
    
    if abs(p_u1 - p_u2) < length / 2:
        start = np.ceil(min(p_u1, p_u2))
        p = max(p_u1, p_u2)
        end = np.floor(p)
        if end == np.ceil(p):
            end = end - 1
    else:
        start = np.ceil(max(p_u1, p_u2))
        p = min(p_u1, p_u2) + length
        end = np.floor(p)
        if end == np.ceil(p):
            end = end - 1
    
    p_us = (np.arange(start, end + 1) % length).astype(np.float64)
    
    if len(p_us) == 0:
        return None
    us = (p_us + 0.5) / length  # why +0.5? Check out the uv2Pixel function

    plan_y = boundary_type(np.array([uv1, uv2]))
    xyz1 = uv2xyz(np.array(uv1), plan_y)
    xyz2 = uv2xyz(np.array(uv2), plan_y)
    x1 = xyz1[0]
    z1 = xyz1[2]
    x2 = xyz2[0]
    z2 = xyz2[2]

    d_x = x2 - x1
    d_z = z2 - z1

    lon_s = (us - 0.5) * 2 * np.pi
    k = np.tan(lon_s)
    ps = (k * z1 - x1) / (d_x - k * d_z)
    cs = np.sqrt((z1 + ps * d_z) ** 2 + (x1 + ps * d_x) ** 2)

    lats = np.arctan2(plan_y, cs)
    vs = lats / np.pi + 0.5
    uv = np.stack([us, vs], axis=-1)

    if start == end:
        return uv[0:1]
    return uv


def connect_corners_xyz(uv1: np.ndarray, uv2: np.ndarray, step=0.01) -> np.ndarray:
    """
    :param uv1: [u, v]
    :param uv2: [u, v]
    :param step: Fixed step size in xyz coordinates
    :return:
    """
    plan_y = boundary_type(np.array([uv1, uv2]))
    xyz1 = uv2xyz(np.array(uv1), plan_y)
    xyz2 = uv2xyz(np.array(uv2), plan_y)

    vec = xyz2 - xyz1
    norm = np.linalg.norm(vec, ord=2)
    # print("!!!!!!!!!!!!!!",norm)
    direct = vec / norm
    xyz = np.array([xyz1 + direct * dis for dis in np.linspace(0, norm, int(norm / step))])
    if len(xyz) == 0:
        xyz = np.array([xyz2])
    uv = xyz2uv(xyz)
    return uv


def connect_corners(uv1: np.ndarray, uv2: np.ndarray, step=0.01, length=None) -> np.ndarray:
    """
    :param uv1: [u, v]
    :param uv2: [u, v]
    :param step:
    :param length:
    :return: [[u1, v1], [u2, v2]....] if length!=None，length of return result = length
    """
    if length is not None:
        uv = connect_corners_uv(uv1, uv2, length)
    elif step is not None:
        uv = connect_corners_xyz(uv1, uv2, step)
    else:
        uv = np.array([uv1])
    return uv


def corners2boundary(corners: np.ndarray, step=0.01, length=None, visible=True) -> np.ndarray:
    """
    When there is occlusion, even if the length is fixed, the final output length may be greater than the given length,
     which is more defined as the fixed step size under UV
    :param length:
    :param step:
    :param corners: [[u1, v1], [u2, v2]....]
    :param visible:
    :return:  [[u1, v1], [u2, v2]....] if length!=None，length of return result = length
    """
    assert step is not None or length is not None, "the step and length parameters cannot be null at the same time"
    if len(corners) < 3:
        return corners

    if visible:
        # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@", corners)
        corners = visibility_corners(corners)
        
        # with open(os.path.join(args.output_dir, k + '.json'), 'w') as f:
        #     json.dump({
        #         'z0': float(z0),
        #         'z1': float(z1),
        #         'uv': [[float(u), float(v)] for u, v in cor_id],
        #     }, f)

    n_con = len(corners)
    boundary = None
    for j in range(n_con):
        uv = connect_corners(corners[j], corners[(j + 1) % n_con], step, length)
        if uv is None:
            continue
        if boundary is None:
            boundary = uv
        else:
            boundary = np.concatenate((boundary, uv))
    boundary = np.roll(boundary, -boundary.argmin(axis=0)[0], axis=0)

    output_polygon = []
    for i, p in enumerate(boundary):
        q = boundary[(i + 1) % len(boundary)]
        if int(p[0] * 10000) == int(q[0] * 10000):
            continue
        output_polygon.append(p)
    output_polygon = np.array(output_polygon, dtype=np.float32)
    # print("OUTPUT_POLYGON_SHAPE", output_polygon.shape) # (560, 2)
    # return output_polygon
    return output_polygon, corners #edited


def corners2boundaries(ratio: float, corners_xyz: np.ndarray = None, corners_uv: np.ndarray = None, step=0.01,
                       length=None, visible=True):
    """
    When both step and length are None, corners are also returned
    :param ratio:
    :param corners_xyz:
    :param corners_uv:
    :param step:
    :param length:
    :param visible:
    :return: floor_boundary, ceil_boundary
    """
    if corners_xyz is None:
        plan_y = boundary_type(corners_uv)
        xyz = uv2xyz(corners_uv, plan_y)
        floor_xyz = xyz.copy()
        ceil_xyz = xyz.copy()
        if plan_y > 0:
            ceil_xyz[:, 1] *= -ratio
        else:
            floor_xyz[:, 1] /= -ratio
    else:
        floor_xyz = corners_xyz.copy()
        ceil_xyz = corners_xyz.copy()
        if corners_xyz[0][1] > 0:
            ceil_xyz[:, 1] *= -ratio
        else:
            floor_xyz = floor_xyz.astype("float") #edited
            floor_xyz[:, 1] /= -ratio

    floor_uv = xyz2uv(floor_xyz)
    ceil_uv = xyz2uv(ceil_xyz)
    if step is None and length is None: #edited
        return floor_uv, ceil_uv
    # print(floor_uv)
    # print(ceil_uv)
    

    floor_boundary = corners2boundary(floor_uv, step, length, visible)
    ceil_boundary = corners2boundary(ceil_uv, step, length, visible)
    # print("@@@@@@@@@@@@@@@@@@@@@@@@@", ceil_boundary)
    return floor_boundary, ceil_boundary

def draw_boundary(pano_img, corners: np.ndarray = None, boundary: np.ndarray = None, draw_corners=True, show=False,
                  step=0.01, length=None, boundary_color=None, marker_color=None, title=None, visible=True):
    if marker_color is None:
        marker_color = [0, 0, 1]
    if boundary_color is None:
        boundary_color = [0, 1, 0]

    assert corners is not None or boundary is not None, "corners or boundary error"
    
    shape = sorted(pano_img.shape)
    assert len(shape) > 1, "pano_img shape error"
    w = shape[-1]
    h = shape[-2]

    pano_img = pano_img.copy()
    if (corners is not None and len(corners) > 2) or \
            (boundary is not None and len(boundary) > 2):
        if isinstance(boundary_color, list) or isinstance(boundary_color, np.array):
            if boundary is None:
                boundary, visible_corners = corners2boundary(corners, step, length, visible)
                # print(corners)
            
            boundary = uv2pixel(boundary, w, h)
            # enhance boundary
            pano_img[np.clip(boundary[:, 1] + 1, 0, h - 1), boundary[:, 0]] = boundary_color  # right point
            pano_img[np.clip(boundary[:, 1] - 1, 0, h - 1), boundary[:, 0]] = boundary_color  # left point

            if pano_img.shape[1] > 512:
                pano_img[np.clip(boundary[:, 1] + 1, 0, h - 1), np.clip(boundary[:, 0] + 1, 0, w - 1)] = boundary_color
                pano_img[np.clip(boundary[:, 1] + 1, 0, h - 1), np.clip(boundary[:, 0] - 1, 0, w - 1)] = boundary_color
                pano_img[np.clip(boundary[:, 1] - 1, 0, h - 1), np.clip(boundary[:, 0] + 1, 0, w - 1)] = boundary_color
                pano_img[np.clip(boundary[:, 1] - 1, 0, h - 1), np.clip(boundary[:, 0] - 1, 0, w - 1)] = boundary_color

            pano_img[boundary[:, 1], np.clip(boundary[:, 0] + 1, 0, w - 1)] = boundary_color  # up point
            pano_img[boundary[:, 1], np.clip(boundary[:, 0] - 1, 0, w - 1)] = boundary_color  # down point

        if corners is not None and draw_corners:
            if visible:
                corners = visibility_corners(corners)
            corners = uv2pixel(corners, w, h)
            for corner in corners:
                cv2.drawMarker(pano_img, tuple(corner), marker_color, markerType=0, markerSize=10, thickness=2)

    # return pano_img
    return pano_img, visible_corners

def draw_boundaries(pano_img, corners_list: list = None, boundary_list: list = None, draw_corners=True, show=False,
                    step=0.01, length=None, boundary_color=None, marker_color=None, title=None, ratio=None, visible=True):
    """

    :param visible:
    :param pano_img:
    :param corners_list:
    :param boundary_list:
    :param draw_corners:
    :param show:
    :param step:
    :param length:
    :param boundary_color: RGB color
    :param marker_color: RGB color
    :param title:
    :param ratio: ceil_height/camera_height
    :return:
    """
    assert corners_list is not None or boundary_list is not None, "corners_list or boundary_list error"

    if corners_list is not None:
        if ratio is not None and len(corners_list) == 1:
            # corners_list = corners2boundaries(ratio, corners_uv=corners_list[0], step=None, visible=visible)
            # corners_list = corners2boundaries(ratio, corners_xyz=corners_list[0], step=None, visible=visible) #edited #convert to uv and calc visible corners
            corners_list= corners2boundaries(ratio, corners_xyz=corners_list[0], step=None, visible=visible) #edited #
        # print(corners_list)    
        visible_corners = []
        for i, corners in enumerate(corners_list):
            # print("@@@@@@@@@@@@@@@@@@@@@@", corners)
            pano_img, vc = draw_boundary(pano_img, corners=corners, draw_corners=draw_corners,
                                     show=show if i == len(corners_list) - 1 else False,
                                     step=step, length=length, boundary_color=boundary_color, marker_color=marker_color,
                                     title=title, visible=visible)
            visible_corners.append(vc)
    elif boundary_list is not None:
        if ratio is not None and len(boundary_list) == 1:
            boundary_list = corners2boundaries(ratio, corners_uv=boundary_list[0], step=None, visible=visible)
            
        for i, boundary in enumerate(boundary_list):
            pano_img = draw_boundary(pano_img, boundary=boundary, draw_corners=draw_corners,
                                     show=show if i == len(boundary_list) - 1 else False,
                                     step=step, length=length, boundary_color=boundary_color, marker_color=marker_color,
                                     title=title, visible=visible)

    return pano_img, visible_corners