from PIL import Image, ImageDraw, ImageFont
import os, csv, re, arabic_reshaper
from pathlib import Path
from bidi.algorithm import get_display

# Get script's parent directory (where both scripts/ and fonts/ live)
PARENT_DIR = Path(__file__).parent.parent  # Goes up one level from script

# ─── CONFIG ───────────────────────────────────────────────────────────────
OUTPUT_DIR = str(PARENT_DIR / "images" / "test")
FONT_PATH  = str(PARENT_DIR / "fonts" / "Amiri-Bold.ttf")
FONT_SIZE  = 60
PADDING    = 20
BG_COLOR   = "white"
FG_COLOR   = "black"

# ─── STATE ────────────────────────────────────────────────────────────────
labels = []

def text_to_image(jawi_text, idx):
    """
    Renders the given Jawi text into Image{idx}.jpg
    and records (filepath, rumi_text) in labels.
    """
    # 1) shape & bidi
    shaped  = arabic_reshaper.reshape(jawi_text)
    display = get_display(shaped)
    print(get_display(jawi_text))
    print(display)

    # 2) load font
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)

    # 3) measure
    dummy = Image.new("RGB", (1, 1))
    d     = ImageDraw.Draw(dummy)
    try:
        l, t, r, b = d.textbbox((0, 0), shaped, font=font)
        w, h = r - l, b - t
    except AttributeError:
        w, h = d.textsize(shaped, font=font)

    # 4) draw
    img = Image.new("RGB", (w + 3*PADDING, h + 3*PADDING), BG_COLOR)
    d   = ImageDraw.Draw(img)
    start_x = img.width - w - PADDING
    d.text((start_x, PADDING), shaped, fill=FG_COLOR, font=font)

    # 5) save as Image{idx}.jpg
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = f"Image{idx}"
    filepath = os.path.join(OUTPUT_DIR, f"{filename}.jpg")
    filenameCsv = f"{filename}.jpg"
    img.save(filepath, "JPEG")
    print("Saved:", filepath)

    # 6) record label (file path, ground‑truth Rumi)
    labels.append((filenameCsv, jawi_text))


if __name__ == "__main__":
    # Jawi↔Rumi pairs
    entries = [
        ("كتاب",  "kitab"),
        ("مسجد",  "masjid"),
        ("سلام",  "salam"),
        ("ڤادڠ ڤرماءينن", "padang permainan"),  
        ("سام-سام! ساي ڬمبيرا داڤت ممبنتو. جك اندا ممڤوڽاءي سبارڠ سؤالن لاڬي، سيلا برتاڽ.", "kereta")
    ]

    # Generate one image per entry, named Image1.jpg, Image2.jpg, …
    for idx, (jawi, rumi) in enumerate(entries, start=1):
        text_to_image(jawi, idx)

    # Dump the labels.csv next to the images
    LABEL_PATH = "labels.csv"
    csv_path = os.path.join(PARENT_DIR, LABEL_PATH)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "jawi_text"])
        writer.writerows(labels)

    print(f"Wrote labels.csv with {len(labels)} entries → {csv_path}")
