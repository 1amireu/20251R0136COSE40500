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
FONT_SIZE = 40
PADDING = 14
BG_COLOR = "white"
FG_COLOR = "black"
AUGMENTATIONS_PER_IMAGE = 1  # How many variants to create per original
ROTATION_ANGLES = [0, 30, 45, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
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

def text_to_image_hybrid_augmentation(jawi_text, rumi_name, fonts, is_augmented=False):
    """
    Generate images with hybrid augmentation approach.
    The original image is generated using render_with_render_time_augmentations with default parameters.
    """
    global image_counter # Assuming image_counter is used within save_image
    
    shaped = jawi_reshaper.reshape(jawi_text) # Reshape the Jawi text
    
    for font_path in fonts:
        try:
            # Correctly extract font name without extension
            font_name = os.path.basename(font_path).split('.')[0]
            
            # Generate base measurements for the text with the current font
            # This is needed by render_with_render_time_augmentations
            base_font = ImageFont.truetype(font_path, FONT_SIZE)
            dummy_img_for_measure = Image.new("RGB", (1, 1)) # Minimal image for drawing context
            draw_context = ImageDraw.Draw(dummy_img_for_measure)
            
            try: # Pillow 9.2.0+ uses textbbox
                l_bbox, t_bbox, r_bbox, b_bbox = draw_context.textbbox((0, 0), shaped, font=base_font)
            except AttributeError: # Older Pillow versions use textsize
                text_w_render, text_h_render = draw_context.textsize(shaped, font=base_font)
                l_bbox, t_bbox, r_bbox, b_bbox = 0, 0, text_w_render, text_h_render
            
            base_w, base_h = r_bbox - l_bbox, b_bbox - t_bbox # Calculate actual width and height of the text
            
            # --- Generate and save the "original" (non-augmented) image ---
            # Define default parameters for a "clean" render
            default_aug_params = {
                'render_time': {
                    'rotation': 0,                 # No rotation
                    'bg_color': (255, 255, 255),   # White background
                    'text_color': (0, 0, 0)        # Black text
                }
            }
            
            # Generate the original image using render_with_render_time_augmentations
            original_img, original_bbox_str = render_with_render_time_augmentations(
                shaped_text=shaped,
                font_path=font_path,
                base_w=base_w,
                base_h=base_h,
                aug_params=default_aug_params
            )

            save_image(
                img=original_img,
                rumi_name=rumi_name,
                jawi_text=jawi_text, # Pass the original Jawi text for label consistency
                font_name=font_name,
                bbox=original_bbox_str,
                augmentation_info=None # Or "original", "clean", etc. if you prefer
            )
            
            # --- Generate and save augmented versions ---
            if is_augmented and ENABLE_AUGMENTATION:
                for aug_idx in range(AUGMENTATIONS_PER_IMAGE):
                    # Pre-determine all augmentation parameters for this iteration
                    aug_params = generate_hybrid_augmentation_parameters()
                    
                    # Apply render-time augmentations (e.g., rotation, colors from aug_params)
                    render_time_img, bbox_str_augmented = render_with_render_time_augmentations(
                        shaped_text=shaped,
                        font_path=font_path,
                        base_w=base_w,
                        base_h=base_h,
                        aug_params=aug_params # Use the generated augmentation parameters
                    )
                    
                    # Apply post-rendering augmentations (e.g., blur, noise from aug_params)
                    final_augmented_img = apply_post_rendering_augmentations(
                        render_time_img, aug_params
                    )
                    
                    # Build augmentation info string for filename
                    aug_info_str = build_hybrid_augmentation_info(aug_params)
                    
                    save_image(
                        img=final_augmented_img,
                        rumi_name=rumi_name,
                        jawi_text=jawi_text, # Pass the original Jawi text
                        font_name=font_name,
                        bbox=bbox_str_augmented,
                        augmentation_info=f"aug{aug_idx}_{aug_info_str}"
                    )
                    
        except Exception as e:
            print(f"Error processing font {font_path} for text '{jawi_text}': {e}")

def generate_hybrid_augmentation_parameters():
    """Generate comprehensive augmentation parameters for hybrid approach"""
    params = {
        # Render-time augmentations
        'render_time': {},
        # Post-rendering augmentations  
        'post_render': {}
    }
    
    # Render-time parameters (applied during text rendering)
    if ENABLE_ROTATION:
        params['render_time']['rotation'] = random.choice(ROTATION_ANGLES) if ROTATION_ANGLES else random.uniform(0, 360)
    else:
        params['render_time']['rotation'] = 0
        
    if ENABLE_RANDOM_COLORS:
        params['render_time']['bg_color'], params['render_time']['text_color'] = generate_contrasting_colors()
    else:
        params['render_time']['bg_color'], params['render_time']['text_color'] = (255, 255, 255), (0, 0, 0)
    
    # Post-rendering parameters (applied after text is rendered)
    if ENABLE_BLUR:
        params['post_render']['blur'] = random.choice(BLUR_RADIUS) if random.random() > 0.5 else 0
    else:
        params['post_render']['blur'] = 0
        
    if ENABLE_NOISE:
        params['post_render']['noise'] = random.choice(NOISE_LEVELS) if random.random() > 0.3 else 0
    else:
        params['post_render']['noise'] = 0
        
    if ENABLE_CONTRAST:
        params['post_render']['contrast'] = random.choice(CONTRAST_FACTORS)
    else:
        params['post_render']['contrast'] = 1.0
    
    # Random padding (can be applied at either stage)
    if ENABLE_RANDOM_PADDING:
        params['padding'] = {
            'top': random.randint(MIN_PADDING, MAX_PADDING),
            'right': random.randint(MIN_PADDING, MAX_PADDING),
            'bottom': random.randint(MIN_PADDING, MAX_PADDING),
            'left': random.randint(MIN_PADDING, MAX_PADDING)
        }
    else:
        params['padding'] = None
        
    return params

def render_with_render_time_augmentations(shaped_text, font_path, base_w, base_h, aug_params):
    """
    Renders text with augmentations, calculates bounding boxes for each word,
    draws each word's bounding box on the image,
    and returns the augmented image and a space-separated bounding box string.
    Uses BICUBIC resampling for rotation.
    """
    global FONT_SIZE, PADDING # Ensure these are accessible

    try:
        font = ImageFont.truetype(font_path, FONT_SIZE)
    except IOError:
        print(f"Warning: Font not found at '{font_path}'. Using Pillow's default font.")
        font = ImageFont.load_default()

    render_params = aug_params.get('render_time', {})
    user_rotation_angle = render_params.get('rotation', 0)
    internal_rotation_angle = -user_rotation_angle

    if internal_rotation_angle != 0:
        diagonal = int(np.sqrt(base_w**2 + base_h**2)) + 6 * PADDING
        canvas_w = canvas_h = diagonal
    else:
        canvas_w = base_w + 6 * PADDING
        canvas_h = base_h + 6 * PADDING
    
    bg_color = render_params.get('bg_color', (255, 255, 255))
    text_color = render_params.get('text_color', (0, 0, 0))
    
    img = Image.new("RGB", (canvas_w, canvas_h), bg_color)
    canvas_center_x = canvas_w // 2
    canvas_center_y = canvas_h // 2

    dummy_draw = ImageDraw.Draw(Image.new("RGB", (1,1)))
    try:
        l_full, t_full, r_full, b_full = dummy_draw.textbbox((0,0), shaped_text, font=font)
    except AttributeError:
        w_full_render, h_full_render = dummy_draw.textsize(shaped_text, font=font)
        l_full, t_full, r_full, b_full = 0, 0, w_full_render, h_full_render
    
    actual_full_text_w = r_full - l_full
    actual_full_text_h = b_full - t_full

    text_block_origin_x_on_temp_canvas = canvas_center_x - actual_full_text_w // 2 - l_full
    text_block_origin_y_on_temp_canvas = canvas_center_y - actual_full_text_h // 2 - t_full

    text_block_img_temp = Image.new("RGBA", (canvas_w, canvas_h), (0,0,0,0))
    draw_temp = ImageDraw.Draw(text_block_img_temp)

    words = shaped_text.split()[::-1]
    current_x_in_block = text_block_origin_x_on_temp_canvas
    word_details_list = []

    for word in words:
        if not word.strip(): continue

        try:
            l_word, t_word, r_word, b_word = dummy_draw.textbbox((0,0), word, font=font)
        except AttributeError:
            w_word_render, h_word_render = dummy_draw.textsize(word, font=font)
            l_word, t_word, r_word, b_word = 0, 0, w_word_render, h_word_render
        
        actual_word_w = r_word - l_word
        
        word_draw_x = current_x_in_block - l_word
        word_draw_y = text_block_origin_y_on_temp_canvas - t_word

        draw_temp.text((word_draw_x, word_draw_y), word, font=font, fill=text_color)

        corners_this_word_on_temp = np.array([
            [current_x_in_block, text_block_origin_y_on_temp_canvas],
            [current_x_in_block + actual_word_w, text_block_origin_y_on_temp_canvas],
            [current_x_in_block + actual_word_w, text_block_origin_y_on_temp_canvas + actual_full_text_h], # Use full text height for word bbox consistency
            [current_x_in_block, text_block_origin_y_on_temp_canvas + actual_full_text_h],
        ])
        word_details_list.append({'word': word, 'corners_on_temp': corners_this_word_on_temp})
        
        try:
            space_width = font.getlength(' ')
        except AttributeError:
            try:
                space_width = dummy_draw.textbbox((0,0)," ",font=font)[2] - dummy_draw.textbbox((0,0)," ",font=font)[0]
            except AttributeError:
                space_width = dummy_draw.textsize(" ", font=font)[0]

        current_x_in_block += actual_word_w + space_width

    pad_offset_left, pad_offset_top = 0, 0

    if internal_rotation_angle != 0:
        # MODIFICATION: Changed from LANCZOS to BICUBIC
        resample_filter = Image.Resampling.BICUBIC if hasattr(Image, 'Resampling') else Image.BICUBIC
        try:
            rotated_text_block_img = text_block_img_temp.rotate(
                internal_rotation_angle, expand=False, center=(canvas_center_x, canvas_center_y),
                resample=resample_filter
            )
        except Exception as e: # Catch specific exceptions if possible
            print(f"Error during image rotation: {e}. Using fallback resampling.")
            # Fallback to a basic filter if even BICUBIC fails for some reason, though unlikely.
            resample_filter_fallback = Image.Resampling.NEAREST if hasattr(Image, 'Resampling') else Image.NEAREST
            rotated_text_block_img = text_block_img_temp.rotate(
                internal_rotation_angle, expand=False, center=(canvas_center_x, canvas_center_y),
                resample=resample_filter_fallback
            )
        img = Image.alpha_composite(img.convert("RGBA"), rotated_text_block_img).convert("RGB")

        cos_a = np.cos(np.radians(-internal_rotation_angle))
        sin_a = np.sin(np.radians(-internal_rotation_angle))
        rot_cx, rot_cy = canvas_center_x, canvas_center_y

        for detail in word_details_list:
            rotated_corners = []
            for (px, py) in detail['corners_on_temp']:
                x_shifted, y_shifted = px - rot_cx, py - rot_cy
                x_new = x_shifted * cos_a - y_shifted * sin_a
                y_new = x_shifted * sin_a + y_shifted * cos_a
                x_new += rot_cx
                y_new += rot_cy
                rotated_corners.append([x_new, y_new])
            detail['final_corners_unpadded'] = np.array(rotated_corners)
    else:
        img = Image.alpha_composite(img.convert("RGBA"), text_block_img_temp).convert("RGB")
        for detail in word_details_list:
            detail['final_corners_unpadded'] = detail['corners_on_temp']

    if aug_params.get('padding'):
        padding_config = aug_params['padding']
        if padding_config:
            img, (pad_offset_left, pad_offset_top) = apply_padding_with_bbox_tracking(
                img, padding_config, bg_color
            )
            for detail in word_details_list:
                detail['final_corners_padded'] = detail['final_corners_unpadded'] + np.array([pad_offset_left, pad_offset_top])
        else:
            for detail in word_details_list:
                detail['final_corners_padded'] = detail['final_corners_unpadded']
    else:
        for detail in word_details_list:
            detail['final_corners_padded'] = detail['final_corners_unpadded']

    final_img_draw = ImageDraw.Draw(img)
    final_word_bbox_strings_for_output = []

    for detail in word_details_list:
        corners = detail['final_corners_padded']
        x1, y1 = corners[0]
        x2, y2 = corners[1]
        x3, y3 = corners[2]
        x4, y4 = corners[3]

        poly_points_to_draw = [
            (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))),
            (int(round(x3)), int(round(y3))), (int(round(x4)), int(round(y4))),
            (int(round(x1)), int(round(y1)))
        ]
        bbox_color = "magenta"
        #final_img_draw.line(poly_points_to_draw, fill=bbox_color, width=1)

        word_bbox_str_for_file = (
            f"{int(round(x1))},{int(round(y1))},"
            f"{int(round(x2))},{int(round(y2))},"
            f"{int(round(x3))},{int(round(y3))},"
            f"{int(round(x4))},{int(round(y4))},"
            f"{detail['word']}"
        )
        final_word_bbox_strings_for_output.append(word_bbox_str_for_file)

    final_bbox_output_string = '\n'.join(final_word_bbox_strings_for_output)
    return img, final_bbox_output_string

def apply_post_rendering_augmentations(img, aug_params):
    """Apply post-rendering augmentations efficiently"""
    augmented_img = img.copy()
    post_params = aug_params['post_render']
    
    # Apply blur (fast post-processing)
    if post_params['blur'] > 0:
        augmented_img = augmented_img.filter(ImageFilter.GaussianBlur(radius=post_params['blur']))
    
    # Apply noise (efficient numpy operation)
    if post_params['noise'] > 0:
        augmented_img = add_noise_pil(augmented_img, post_params['noise'])
    
    # Apply contrast adjustment (fast PIL operation)
    if post_params['contrast'] != 1.0:
        enhancer = ImageEnhance.Contrast(augmented_img)
        augmented_img = enhancer.enhance(post_params['contrast'])
    
    return augmented_img

def apply_padding_with_bbox_tracking(img, padding_params, bg_color):
    """Apply padding and return the padded image"""
    top = padding_params['top']
    right = padding_params['right'] 
    bottom = padding_params['bottom']
    left = padding_params['left']
    
    width, height = img.size
    new_width = width + right + left
    new_height = height + top + bottom
    
    # Create new image with padding
    padded_img = Image.new(img.mode, (new_width, new_height), bg_color)
    padded_img.paste(img, (left, top))
    
    return padded_img, (left, top)

def build_hybrid_augmentation_info(params):
    """Build comprehensive augmentation info string"""
    info_parts = []
    
    # Render-time augmentations
    if params['render_time']['rotation'] != 0:
        info_parts.append(f"rot{int(params['render_time']['rotation'])}")
    
    # Post-rendering augmentations
    if params['post_render']['blur'] > 0:
        info_parts.append(f"blur{params['post_render']['blur']:.1f}")
    if params['post_render']['noise'] > 0:
        info_parts.append(f"noise{params['post_render']['noise']:.2f}")
    if params['post_render']['contrast'] != 1.0:
        info_parts.append(f"cont{params['post_render']['contrast']:.1f}")
    
    # Padding info
    if params['padding']:
        info_parts.append("padded")
    
    return "_".join(info_parts) if info_parts else "clean"
            
# ─── AUGMENTATION HELPERS ───────────────────────────────────────────────
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


def save_image(img, rumi_name, jawi_text, bbox=None, font_name=None, augmentation_info=None):
    """
    Saves image and bounding box information.
    For multi-word bboxes, the order of words in the label file is reversed (last word first).
    """
    global image_counter

    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(LABEL_DIR, exist_ok=True)

    if USE_DESCRIPTIVE_NAMES and font_name:
        base_name = f"{rumi_name}_{font_name}"
        if augmentation_info:
            image_filename_stem = f"{base_name}_{augmentation_info}"
        else:
            image_filename_stem = base_name
        image_filename = f"{image_filename_stem}.jpg"
    else:
        image_filename_stem = f"img_{image_counter}"
        image_filename = f"{image_filename_stem}.jpg"

    label_filename = f"gt_img_{image_counter}.txt"
    image_filepath = os.path.join(IMAGE_DIR, image_filename)
    label_filepath = os.path.join(LABEL_DIR, label_filename)

    # Save the image first
    img.save(image_filepath, "JPEG")

    # Calculate bounding box if not already provided
    if bbox is None:
        # If calculate_text_bbox is called, it needs to produce the space-separated
        # word-bbox format if that's the desired input for reversal.
        bbox = calculate_text_bbox(img, jawi_text, font_name)

    # --- Prepare content for the label file (reverse order if multi-word) ---
    content_for_label_file = ""
    if bbox and isinstance(bbox, str): # Check if bbox is a non-empty string
        # Split the bbox string into individual "coordinate,word" units
        word_bbox_units = [unit for unit in bbox.split('\n') if unit.strip()] # Filter out empty strings from multiple spaces

        if len(word_bbox_units) > 0: # Proceed only if there are units to process
            # Reverse the list of units
            reversed_word_bbox_units = word_bbox_units[::-1] # Slicing creates a reversed copy
            
            # Join them back with spaces
            content_for_label_file = '\n'.join(reversed_word_bbox_units)
        else: # bbox might have been all spaces or empty
            content_for_label_file = bbox # Keep it as is
    elif bbox is not None: # If bbox is not a string (e.g. some other object) or None
        content_for_label_file = str(bbox) # Default to string conversion
    # If bbox was None and calculate_text_bbox returned None/empty, content_for_label_file remains ""

    # --- Write the (potentially reversed) content to the individual label file ---
    try:
        with open(label_filepath, "w", encoding="utf-8") as f:
            f.write(content_for_label_file)
    except Exception as e:
        print(f"Error writing label file {label_filepath}: {e}")
    
    # Append to the master labels list (storing what was actually written to the file)
    labels.append((image_filename, jawi_text, content_for_label_file))
    image_counter += 1

    return image_filepath, label_filepath

def calculate_text_bbox(img, text, font):
    """Calculate the bounding box coordinates of the text in the image"""
    # Get text size
    dummy = Image.new("RGB", (1, 1))
    d = ImageDraw.Draw(dummy)
    try:
        l, t, r, b = d.textbbox((0, 0), text, font=font)
    except AttributeError:
        print("Need newer version of Pillow to run this code")
        return None

    text_w, text_h = r - l, b - t

    # Calculate bounding box coordinates
    x1 = 3 * PADDING  # Assuming text starts at 3*PADDING
    y1 = 3 * PADDING  # Assuming text starts at 3*PADDING
    x2 = x1 + text_w
    y2 = y1
    x3 = x1 + text_w
    y3 = y1 + text_h
    x4 = x1
    y4 = y1 + text_h

    return f"{x1},{y1},{x2},{y2},{x3},{y3},{x4},{y4},{text}"

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


# ─── MAIN EXECUTION ────────────────────────────────────────────────────
if __name__ == "__main__":
    # Initialize paths
    PARENT_DIR = Path(__file__).parent.parent
    IMAGE_DIR = str(PARENT_DIR / "datasets" / "dataset-slides-0" / "images")
    LABEL_DIR = str(PARENT_DIR / "datasets" / "dataset-slides-0" / "labels")
    FONT_DIR = str(PARENT_DIR / "fonts")
    
    # File path for Jawi text entries
    ENTRIES_FILE = str(PARENT_DIR / "data" / "test.txt")
    
    # Get all fonts from the fonts directory
    all_fonts = get_all_fonts(FONT_DIR)
    print(f"Found {len(all_fonts)} fonts in {FONT_DIR}")
    
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
    
    labels = []
    
    # Configure augmentation toggles (can be modified as needed)
    ENABLE_AUGMENTATION = True  # Master toggle
    ENABLE_ROTATION = True
    ENABLE_BLUR = True
    ENABLE_NOISE = True
    ENABLE_CONTRAST = True
    
    # Random color and padding configuration
    ENABLE_RANDOM_COLORS = True
    ENABLE_RANDOM_PADDING = True
    MIN_PADDING = 20
    MAX_PADDING = 150
    
    # Toggle for descriptive vs. sequential naming
    USE_DESCRIPTIVE_NAMES = False
    
    # Generate original + augmented images
    for jawi, label in entries:
        text_to_image_hybrid_augmentation(jawi, label, all_fonts, is_augmented=True)
    
    print(f"Generated {len(labels)} images with {'descriptive' if USE_DESCRIPTIVE_NAMES else 'sequential'} naming")
    
    # Print augmentation configuration summary
    print("\nAugmentation Configuration:")
    print(f"Master toggle: {'ENABLED' if ENABLE_AUGMENTATION else 'DISABLED'}")
    if ENABLE_AUGMENTATION:
        print(f"- Rotation: {'ENABLED' if ENABLE_ROTATION else 'DISABLED'}")
        print(f"- Blur: {'ENABLED' if ENABLE_BLUR else 'DISABLED'}")
        print(f"- Noise: {'ENABLED' if ENABLE_NOISE else 'DISABLED'}")
        print(f"- Contrast: {'ENABLED' if ENABLE_CONTRAST else 'DISABLED'}")
        print(f"- Random Colors: {'ENABLED' if ENABLE_RANDOM_COLORS else 'DISABLED'}")
        print(f"- Localization Dataset: {'ENABLED' if ENABLE_RANDOM_PADDING else 'DISABLED'}")
        print(f"- Descriptive Image Name: {'ENABLED' if USE_DESCRIPTIVE_NAMES else 'DISABLED'}")