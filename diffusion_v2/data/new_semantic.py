import os
import cv2
import numpy as np

# Source and target directories
source_directory = "diffusion_dataset/semantic/"
target_directory = "diffusion_dataset/new_semantic/"

# Color mapping dictionary: old_color -> new_color
color_mapping = {
    (40, 39, 214): (232, 199, 174),
    (213, 176, 197): (232, 199, 174),
    (0, 0, 0): (232, 199, 174)
}

# Ensure the target directory exists
os.makedirs(target_directory, exist_ok=True)

# Process each image in the source directory
for filename in os.listdir(source_directory):
    if filename.endswith(('.jpg', '.png', '.jpeg')):
        # Read the image
        image_path = os.path.join(source_directory, filename)
        original_image = cv2.imread(image_path)

        # Iterate through pixels and replace colors
        for old_color, new_color in color_mapping.items():
            mask = np.all(original_image == old_color, axis=-1)
            original_image[mask] = new_color

        # Save the modified image
        target_path = os.path.join(target_directory, filename)
        cv2.imwrite(target_path, original_image)

        print(f"Processed: {filename}")

print("Processing complete.")