import cv2
import numpy as np
import largestinteriorrectangle as lir

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
    denormalized_rectangle_coordinates = denormalize_coordinates(rectangle_corrdinates,corners)

    return denormalized_rectangle_coordinates
