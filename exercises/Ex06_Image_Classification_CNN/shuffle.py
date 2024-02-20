import os
import random

# Set the path to the directory containing the images
img_dir = "Ex06_images/Brain_Tumor_Dataset/test/test"

# Set the path to the labels.txt file
labels_file = "labels.txt"

# Get the list of image files with all extensions in the directory
img_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]

# Load the list of image names and classes from the gabarito.txt file
with open(labels_file, "r") as f:
    lines = f.readlines()
    img_classes = {}
    for line in lines:
        name, img_class = line.strip().split()
        img_classes[name] = img_class

# Shuffle the list of image files
random.shuffle(img_files)

# Rename the image files and update the labels.txt file
for i, img_file in enumerate(img_files):
    old_path, old_name = os.path.split(img_file)
    name, ext = os.path.splitext(old_name)
    new_name = f"{i:04d}_test"
    new_path = os.path.join(img_dir, new_name + ext)
    os.rename(img_file, new_path)

    # Update the labels.txt file
    with open(labels_file, "r+") as f:
        lines = f.readlines()
        f.seek(0)
        f.truncate()
        for line in lines:
            img_name, img_class = line.strip().split()
            if img_name == name:
                f.write(f"{new_name} {img_class}\n")
            else:
                f.write(line)
