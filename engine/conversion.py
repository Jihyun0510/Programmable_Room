import numpy as np
from engine.panostretch import pano_connect_points


def lonlat2uv(lonlat, axis=None):
    if axis is None:
        u = lonlat[..., 0:1] / (2 * np.pi) + 0.5
        v = lonlat[..., 1:] / np.pi + 0.5
    elif axis == 0:
        u = lonlat / (2 * np.pi) + 0.5
        return u
    elif axis == 1:
        v = lonlat / np.pi + 0.5
        return v
    else:
        assert False, "axis error"

    lst = [u, v]
    uv = np.concatenate(lst, axis=-1)
    return uv

def xyz2lonlat(xyz):
    atan2 = np.arctan2 
    asin = np.arcsin 
    norm = np.linalg.norm(xyz, axis=-1)
    # print("XYZ", np.array(xyz).shape)
    # print("NORM",norm.shape) 
    xyz_norm = xyz / norm[..., None]
    x = xyz_norm[..., 0:1]
    y = xyz_norm[..., 1:2]
    z = xyz_norm[..., 2:]
    lon = atan2(x, z)
    lat = asin(y)
    lst = [lon, lat]
    lonlat = np.concatenate(lst, axis=-1) 
    return lonlat


def xyz2uv(xyz):
    lonlat = xyz2lonlat(xyz)
    uv = lonlat2uv(lonlat)
    return uv



def final_uv(xyz):

    ratio = 1
    coordinates = np.array(xyz)
    floor_xyz = coordinates.copy()
    ceil_xyz = coordinates.copy()
    if coordinates[0][1] > 0:
        ceil_xyz[:, 1] *= -ratio
    else:
        floor_xyz[:, 1] /= -ratio

    ceil_uv = xyz2uv(floor_xyz)
    floor_uv = xyz2uv(ceil_xyz)


    return  floor_uv, ceil_uv


def uv2pixel(uv, w=1024, h=512, axis=None, need_round=True):
    """
    :param uv: [[u1, v1], [u2, v2] ...]
    :param w: width of panorama image
    :param h: height of panorama image
    :param axis: sometimes the input data is only u(axis =0) or only v(axis=1)
    :param need_round:
    :return:
    """
    if axis is None:
        pu = uv[..., 0:1] * w - 0.5
        pv = uv[..., 1:] * h - 0.5
    elif axis == 0:
        pu = uv * w - 0.5
        if need_round:
            pu = pu.round().astype(int) if isinstance(uv, np.ndarray) else pu.round().int()
        return pu
    elif axis == 1:
        pv = uv * h - 0.5
        if need_round:
            pv = pv.round().astype(np.int) if isinstance(uv, np.ndarray) else pv.round().int()
        return pv
    else:
        assert False, "axis error"

    lst = [pu, pv]
    if need_round:
        pixel = np.concatenate(lst, axis=-1).round().astype(int)
                                                                   
    else:
        pixel = np.concatenate(lst, axis=-1)
    pixel[..., 0] = pixel[..., 0] % w
    pixel[..., 1] = pixel[..., 1] % h

    return pixel

def uv2lonlat(uv, axis=None):
    if axis is None:
        lon = (uv[..., 0:1] - 0.5) * 2 * np.pi
        lat = (uv[..., 1:] - 0.5) * np.pi
    elif axis == 0:
        lon = (uv - 0.5) * 2 * np.pi
        return lon
    elif axis == 1:
        lat = (uv - 0.5) * np.pi
        return lat
    else:
        assert False, "axis error"

    lst = [lon, lat]
    lonlat = np.concatenate(lst, axis=-1) 
    return lonlat

def lonlat2xyz(lonlat, plan_y=None):
    lon = lonlat[..., 0:1]
    lat = lonlat[..., 1:]
    cos = np.cos 
    sin = np.sin 
    x = cos(lat) * sin(lon)
    y = sin(lat)
    z = cos(lat) * cos(lon)
    lst = [x, y, z]
    xyz = np.concatenate(lst, axis=-1) 

    if plan_y is not None:
        xyz = xyz * (plan_y / xyz[..., 1])[..., None]
    return xyz

def uv2xyz(uv, plan_y=None, spherical=False):
    lonlat = uv2lonlat(uv)
    xyz = lonlat2xyz(lonlat)

    if spherical:
        # Projection onto the sphere
        return xyz

    if plan_y is None:
        from boundary import boundary_type
        plan_y = boundary_type(uv)

    xyz = xyz * (plan_y / xyz[..., 1])[..., None]

    return xyz

def uv2pixel(uv, w=1024, h=512, axis=None, need_round=True):
    """
    :param uv: [[u1, v1], [u2, v2] ...]
    :param w: width of panorama image
    :param h: height of panorama image
    :param axis: sometimes the input data is only u(axis =0) or only v(axis=1)
    :param need_round:
    :return:
    """
    if axis is None:
        pu = uv[..., 0:1] * w - 0.5
        pv = uv[..., 1:] * h - 0.5
    elif axis == 0:
        pu = uv * w - 0.5
        if need_round:
            pu = pu.round().astype(int) if isinstance(uv, np.ndarray) else pu.round().int()
        return pu
    elif axis == 1:
        pv = uv * h - 0.5
        if need_round:
            pv = pv.round().astype(int) if isinstance(uv, np.ndarray) else pv.round().int()
        return pv
    else:
        assert False, "axis error"

    lst = [pu, pv]
    if need_round:
        pixel = np.concatenate(lst, axis=-1).round().astype(int) 
    else:
        pixel = np.concatenate(lst, axis=-1)
    pixel[..., 0] = pixel[..., 0] % w
    pixel[..., 1] = pixel[..., 1] % h

    return pixel

def sort_xy_filter_unique(xs, ys, y_small_first=True):
    xs, ys = np.array(xs), np.array(ys)
    idx_sort = np.argsort(xs + ys / ys.max() * (int(y_small_first)*2-1))
    xs, ys = xs[idx_sort], ys[idx_sort]
    _, idx_unique = np.unique(xs, return_index=True)
    xs, ys = xs[idx_unique], ys[idx_unique]
    assert np.all(np.diff(xs) > 0)
    return xs, ys

def cor_2_1d(cor, H, W):
    bon_ceil_x, bon_ceil_y = [], []
    bon_floor_x, bon_floor_y = [], []
    n_cor = len(cor)
    for i in range(n_cor // 2):
        xys = pano_connect_points(cor[i*2],
                                              cor[(i*2+2) % n_cor],
                                              z=-50, w=W, h=H)
        bon_ceil_x.extend(xys[:, 0])
        bon_ceil_y.extend(xys[:, 1])
    for i in range(n_cor // 2):
        xys = pano_connect_points(cor[i*2+1],
                                              cor[(i*2+3) % n_cor],
                                              z=50, w=W, h=H)
        bon_floor_x.extend(xys[:, 0])
        bon_floor_y.extend(xys[:, 1])
    bon_ceil_x, bon_ceil_y = sort_xy_filter_unique(bon_ceil_x, bon_ceil_y, y_small_first=True)
    bon_floor_x, bon_floor_y = sort_xy_filter_unique(bon_floor_x, bon_floor_y, y_small_first=False)
    bon = np.zeros((2, W))
    bon[0] = np.interp(np.arange(W), bon_ceil_x, bon_ceil_y, period=W)
    bon[1] = np.interp(np.arange(W), bon_floor_x, bon_floor_y, period=W)
    bon = ((bon + 0.5) / H - 0.5) * np.pi
    return bon

def np_coorx2u(coorx, coorW=1024):
    PI = float(np.pi)
    return ((coorx + 0.5) / coorW - 0.5) * 2 * PI


def np_coory2v(coory, coorH=512):
    PI = float(np.pi)
    return -((coory + 0.5) / coorH - 0.5) * PI


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