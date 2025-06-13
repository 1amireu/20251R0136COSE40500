# install arabic_reshaper, Pillow, numpy
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
import os
import csv
import random
import numpy as np
import glob
from pathlib import Path

from arabic_reshaper import ArabicReshaper
from arabic_reshaper.letters import LETTERS_ARABIC, connects_with_letters_before_and_after

# Create a copy of the letters dictionary
CUSTOM_LETTERS = LETTERS_ARABIC.copy()

# Define Jawi-specific letters with their presentation forms
JAWI_LETTERS = {
    '\u0686': ('\u0686', '\u0686\u0640', '\u0640\u0686\u0640', '\u0640\u0686'),  # Cha
    '\u06A0': ('\u06A0', '\u06A0\u0640', '\u0640\u06A0\u0640', '\u0640\u06A0'),  # Nga
    '\u06A4': ('\u06A4', '\u06A4\u0640', '\u0640\u06A4\u0640', '\u0640\u06A4'),  # Pa
    '\u06AC': ('\u0762', '\u0762\u0640', '\u0640\u0762\u0640', '\u0640\u0762'),  # Gaf
    '\u0762': ('\u0762', '\u0762\u0640', '\u0640\u0762\u0640', '\u0640\u0762'),  # Gaf
    '\u06CF': ('\u06CF', '', '', '\u0640\u06CF'),                                # Vau
    '\u06BD': ('\u06BD', '\u06BD\u0640', '\u0640\u06BD\u0640', '\u0640\u06BD'),  # Nya
}

# Update the letters dictionary with Jawi letters
CUSTOM_LETTERS.update(JAWI_LETTERS)

# Create custom reshaper for Jawi
class JawiReshaper(ArabicReshaper):
    def __init__(self, configuration=None, configuration_file=None):
        super().__init__(configuration, configuration_file)
        # Override the letters dictionary
        self.letters = CUSTOM_LETTERS

# Initialize the custom reshaper
jawi_reshaper = JawiReshaper()

# ─── AUGMENTATION CONFIG ────────────────────────────────────────────────
FONT_SIZE = 60
PADDING = 20
BG_COLOR = "white"
FG_COLOR = "black"
AUGMENTATIONS_PER_IMAGE = 1  # How many variants to create per original
ROTATION_ANGLES = [0, 30, 20, -30, -20]
NOISE_LEVELS = [0.01, 0.03, 0.05]  # Gaussian noise variances
BLUR_RADIUS = [0.5, 1.0]  # Gaussian blur radius
CONTRAST_FACTORS = [0.8, 1.0, 1.2]  # Contrast adjustment

# ─── AUGMENTATION TOGGLES ──────────────────────────────────────────────
# Master toggle for all augmentations
ENABLE_AUGMENTATION = True

# Individual toggles for specific augmentation types
ENABLE_ROTATION = True
ENABLE_BLUR = True
ENABLE_NOISE = True
ENABLE_CONTRAST = True

# Random color and padding configuration
ENABLE_RANDOM_COLORS = False
ENABLE_RANDOM_PADDING = False
MIN_PADDING = 20
MAX_PADDING = 150

# Naming convention toggle
USE_DESCRIPTIVE_NAMES = True

# Global counter for image numbering
image_counter = 1

# ─── FONT DISCOVERY ────────────────────────────────────────────────────
def get_all_fonts(font_dir):
    """Get all font files from the fonts directory"""
    font_extensions = ['*.ttf', '*.otf']
    fonts = []
    for ext in font_extensions:
        fonts.extend(glob.glob(os.path.join(font_dir, ext)))
    return fonts

# ─── TEXT TO IMAGE FUNCTION ────────────────────────────────────────────
def text_to_image_enhanced(jawi_text, rumi_name, font_path, apply_augmentation_flag):
    """Generate a single image for the Jawi text, optionally augmented, using the specified font.
    Normal images use fixed colors. Augmented images use random colors if ENABLE_RANDOM_COLORS is True.
    """
    global image_counter
    
    shaped = jawi_reshaper.reshape(jawi_text)
    
    try:
        font_name = os.path.basename(font_path).split('.')[0]
        font = ImageFont.truetype(font_path, FONT_SIZE)
        
        # Measure text dimensions
        dummy_img = Image.new("RGB", (1, 1))
        draw_dummy = ImageDraw.Draw(dummy_img)
        l, t, r, b = draw_dummy.textbbox((0, 0), shaped, font=font)
        w, h = r - l, b - t
        
        # Determine background and text colors
        # Apply random colors ONLY if it's an augmented image AND random colors are globally enabled
        if apply_augmentation_flag and ENABLE_RANDOM_COLORS:
            bg_color, text_color = generate_contrasting_colors()
        else:
            # For normal images, or if random colors are globally disabled, use fixed colors
            bg_color_tuple = tuple(int(BG_COLOR.strip('#')[i:i+2], 16) for i in (0, 2, 4)) if isinstance(BG_COLOR, str) and BG_COLOR.startswith('#') else (255, 255, 255) if BG_COLOR == "white" else (0,0,0)
            fg_color_tuple = tuple(int(FG_COLOR.strip('#')[i:i+2], 16) for i in (0, 2, 4)) if isinstance(FG_COLOR, str) and FG_COLOR.startswith('#') else (0, 0, 0) if FG_COLOR == "black" else (255,255,255)
            bg_color, text_color = bg_color_tuple, fg_color_tuple

        # Create base image
        base_image = Image.new("RGB", (w + 3*PADDING, h + 3*PADDING), bg_color)
        draw_context = ImageDraw.Draw(base_image)
        draw_context.text((PADDING, PADDING), shaped, fill=text_color, font=font)
        
        final_image = base_image
        augmentation_details_for_filename = None

        if apply_augmentation_flag and ENABLE_AUGMENTATION: # Check master augmentation toggle
            # Apply the configured set of augmentations
            # Note: apply_augmentations_pil might change the background if certain augmentations like perspective transform introduce new areas.
            # The current color of these new areas is determined by the `bg` parameter passed to it.
            final_image, augmentation_details_for_filename = apply_augmentations_pil(base_image, bg=bg_color)
        
        save_image(final_image, rumi_name, jawi_text,
                    font_name=font_name,
                    augmentation_info=augmentation_details_for_filename)
            
    except Exception as e:
        print(f"Error processing text '{jawi_text}' with font {font_path}: {e}")

# ─── AUGMENTATION HELPERS ───────────────────────────────────────────────
def rotate_image_pil(img, bg_color):
    """Rotate image with white background and expand canvas to fit full rotated image"""
    if ROTATION_ANGLES:
        angle = random.choice(ROTATION_ANGLES)
    else:
        angle = random.uniform(0, 360)
        
    return img.rotate(angle, expand=True, fillcolor=bg_color), angle

def add_noise_pil(img, variance):
    """Add Gaussian noise to PIL image"""
    img_arr = np.array(img)
    noise = np.random.normal(0, variance * 255, img_arr.shape)
    noisy = np.clip(img_arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)

def generate_random_color():
    """Generate a random RGB color"""
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def generate_contrasting_colors():
    """Generate background and text colors with good contrast"""
    bg_color = generate_random_color()
    
    # Calculate luminance for background
    bg_luminance = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]
    
    # Choose contrasting text color
    if bg_luminance > 127:
        # Dark text on light background
        text_color = (random.randint(0, 100), random.randint(0, 100), random.randint(0, 100))
    else:
        # Light text on dark background  
        text_color = (random.randint(155, 255), random.randint(155, 255), random.randint(155, 255))
    
    return bg_color, text_color

def add_random_padding(img, bg_color, min_padding=10, max_padding=100):
    """Add random padding to all sides of the image"""
    top = random.randint(min_padding, max_padding)
    right = random.randint(min_padding, max_padding)
    bottom = random.randint(min_padding, max_padding)
    left = random.randint(min_padding, max_padding)
    
    width, height = img.size
    new_width = width + right + left
    new_height = height + top + bottom
    
    # Create new image with padding
    padded_img = Image.new(img.mode, (new_width, new_height), bg_color)
    padded_img.paste(img, (left, top))
    
    return padded_img, (left, top, right, bottom)

def apply_augmentations_pil(img, bg="white"):
    """Apply random augmentations to the image and optionally transform bounding box"""
    augmented_img = img.copy()
    aug_info_parts = []
    
    original_width, original_height = img.size
    
    # Random rotation
    if ENABLE_ROTATION:
        rotated_img, angle = rotate_image_pil(augmented_img, bg)
        augmented_img = rotated_img
        aug_info_parts.append(f"rot{int(angle)}")

    # Random blur
    if ENABLE_BLUR and random.random() > 0.5:
        radius = random.choice(BLUR_RADIUS)
        augmented_img = augmented_img.filter(ImageFilter.GaussianBlur(radius=radius))
        aug_info_parts.append(f"blur{radius:.1f}")

    # Random noise
    if ENABLE_NOISE and random.random() > 0.3:
        noise_level = random.choice(NOISE_LEVELS)
        augmented_img = add_noise_pil(augmented_img, noise_level)
        aug_info_parts.append(f"noise{noise_level:.2f}")

    # Random contrast
    if ENABLE_CONTRAST:
        contrast = random.choice(CONTRAST_FACTORS)
        enhancer = ImageEnhance.Contrast(augmented_img)
        augmented_img = enhancer.enhance(contrast)
        aug_info_parts.append(f"cont{contrast:.1f}")

    # Join all augmentation info
    aug_info = "_".join(aug_info_parts) if aug_info_parts else "noaug"
    
    return augmented_img, aug_info



def save_image(img, rumi_name, jawi_text, font_name=None, augmentation_info=None):
    """Save image with either descriptive or sequential naming"""
    global image_counter
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if USE_DESCRIPTIVE_NAMES and font_name:
        # Descriptive naming with font and augmentation details
        base_name = f"{rumi_name}_{font_name}"
        if augmentation_info:
            filename = f"{base_name}_{augmentation_info}.jpg"
        else:
            filename = f"{base_name}.jpg"
    else:
        # Simple sequential naming
        filename = f"img_{image_counter}.jpg"
    
    filepath = os.path.join(OUTPUT_DIR, filename)
    img.save(filepath, "JPEG")
    labels.append((filename, jawi_text))
    #print(f"Saved: {filepath} (Text: {jawi_text})")
    
    # Increment the counter for the next image
    image_counter += 1

def read_entries_from_file(file_path):
    """Read Jawi text entries from a text file (one per line)"""
    entries = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                # Each line contains only Jawi text
                line = line.strip()
                if line and not line.startswith('#'):  # Skip empty lines and comments
                    jawi_text = line
                    # Use the Jawi text as the label (or you could leave Rumi blank)
                    entries.append((jawi_text, jawi_text))  # (jawi_text, label)
        
        print(f"Successfully loaded {len(entries)} Jawi entries from {file_path}")
        return entries
    
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return []
    except Exception as e:
        print(f"Error reading file: {e}")
        return []



if __name__ == "__main__":
    # Initialize paths
    PARENT_DIR = Path(__file__).parent.parent
    OUTPUT_DIR = str(PARENT_DIR / "datasets" / "dataset-tests-2" / "images" )
    FONT_DIR = str(PARENT_DIR / "fonts")
    
    # File path for Jawi text entries
    ENTRIES_FILE = str(PARENT_DIR / "data" / "jawi_random_sentences_15k.txt")
    
    # Get all fonts from the fonts directory
    all_fonts = get_all_fonts(FONT_DIR)
    print(f"Found {len(all_fonts)} fonts in {FONT_DIR}")

    if not all_fonts:
        print("Error: No fonts found in the specified FONT_DIR. Please add fonts and try again.")
        exit() # Exit if no fonts are available
    
    # Read Jawi entries from file
    entries = read_entries_from_file(ENTRIES_FILE)
    
    # If no entries were loaded, use default examples
    if not entries:
        print("Using default entries as fallback...")
        entries = [
            ("كتاب", "kitab"),
            ("مسجد", "masjid"),
            ("سلام", "salam"),
            ("ڤادڠ ڤرماءينن", "padang_permainan"),
            ("كريتا", "kereta")
        ]
    
    labels = [] # Initialize labels list
    
    # Configure augmentation toggles (can be modified as needed)
    ENABLE_AUGMENTATION = True  # Master toggle
    ENABLE_ROTATION = False # Example: True
    ENABLE_BLUR = True
    ENABLE_NOISE = True
    ENABLE_CONTRAST = True
    
    # Random color and padding configuration
    ENABLE_RANDOM_COLORS = True
    ENABLE_RANDOM_PADDING = False # This setting's integration into image generation isn't shown in the provided snippet's core path
    MIN_PADDING = 20
    MAX_PADDING = 150
    
    # Toggle for descriptive vs. sequential naming
    USE_DESCRIPTIVE_NAMES = False # As per the original main block setting
    
    num_fonts = len(all_fonts)

    # Generate images: one per entry, alternating normal/augmented, cycling fonts
    for i, (jawi, label) in enumerate(entries):
        current_font_path = all_fonts[i % num_fonts]  # Cycle through fonts
        
        # Determine if this entry should be augmented:
        # 0th entry (i=0) -> normal (i % 2 == 0)
        # 1st entry (i=1) -> augmented (i % 2 == 1)
        # ...and so on.
        should_apply_augmentation = (i % 2 == 1)
        
        #print(f"Processing entry {i+1}/{len(entries)}: '{label}', Font: '{os.path.basename(current_font_path)}', Augmented: {should_apply_augmentation and ENABLE_AUGMENTATION}")
        text_to_image_enhanced(jawi, label, current_font_path, apply_augmentation_flag=should_apply_augmentation)
    
    # Save labels
    csv_path = os.path.join(PARENT_DIR, "datasets", "dataset-tests-2", "labels.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True) # Ensure directory for labels.csv exists
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Header depends on ENABLE_RANDOM_PADDING, assuming its effect on labels is handled elsewhere or intended for future use.
        if ENABLE_RANDOM_PADDING: 
            writer.writerow(["file", "text", "bbox"])
        else:
            writer.writerow(["file", "text"])
        writer.writerows(labels)
    
    print(f"Generated {len(labels)} images with {'descriptive' if USE_DESCRIPTIVE_NAMES else 'sequential'} naming.")
    
    # Print augmentation configuration summary
    print("\nAugmentation Configuration:")
    print(f"Master toggle: {'ENABLED' if ENABLE_AUGMENTATION else 'DISABLED'}")
    if ENABLE_AUGMENTATION:
        print(f"- Rotation: {'ENABLED' if ENABLE_ROTATION else 'DISABLED'}")
        print(f"- Blur: {'ENABLED' if ENABLE_BLUR else 'DISABLED'}")
        print(f"- Noise: {'ENABLED' if ENABLE_NOISE else 'DISABLED'}")
        print(f"- Contrast: {'ENABLED' if ENABLE_CONTRAST else 'DISABLED'}")
    print(f"Random Colors: {'ENABLED' if ENABLE_RANDOM_COLORS else 'DISABLED'}")
    # ENABLE_RANDOM_PADDING might be better termed 'Localization Dataset Mode' or similar if it affects bbox
    print(f"Random Padding (for localization dataset): {'ENABLED' if ENABLE_RANDOM_PADDING else 'DISABLED'}")
    print(f"Descriptive Image Name: {'ENABLED' if USE_DESCRIPTIVE_NAMES else 'DISABLED'}")


