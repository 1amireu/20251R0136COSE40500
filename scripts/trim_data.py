import os
from pathlib import Path
import shutil
import re

# Define the source and destination directories
PARENT_DIR = Path.cwd()  # Assuming current working directory is the project root
SOURCE_DIR = PARENT_DIR / "images" / "dataset"
DEST_DIR = PARENT_DIR / "images" / "trimmed_dataset"

# Create destination directory if it doesn't exist
os.makedirs(DEST_DIR, exist_ok=True)

# Generate the sequence
def generate_sequence(max_limit):
    indices = []
    start = 9
    while start <= max_limit:
        indices.extend([range(start, start + 9)])
        start += 16
    return indices

# Extract number from filename (e.g., "Image5.jpg" -> 5)
def extract_number(filename):
    match = re.search(r'(\d+)', filename)
    if match:
        return int(match.group(1))
    return 0

try:
    # List all image files in the source directory
    all_images = sorted([f for f in os.listdir(SOURCE_DIR) if f.lower().endswith('.jpg')])
    print(f"Found {len(all_images)} images in source directory")
    
    # Create a dictionary mapping image numbers to filenames
    image_dict = {extract_number(img): img for img in all_images}
    
    # Generate the sequence
    max_number = max(image_dict.keys()) if image_dict else 0
    selected_indices = generate_sequence(max_number)
    
    # Select images with the specified indices
    selected_images = [image_dict[idx] for idx in selected_indices if idx in image_dict]
    print(f"Selected {len(selected_images)} images")
    
    # Print the selected indices for verification
    selected_numbers = [extract_number(img) for img in selected_images]
    #print(f"Selected image numbers: {sorted(selected_numbers)}")
    
    # Copy selected images to destination directory
    for img_name in selected_images:
        src_path = SOURCE_DIR / img_name
        dst_path = DEST_DIR / img_name
        shutil.copy2(src_path, dst_path)
    
    print(f"Selected images copied to: {DEST_DIR}")

except FileNotFoundError:
    print(f"Error: Source directory not found: {SOURCE_DIR}")
    print("Please make sure the path to your generated images is correct.")
