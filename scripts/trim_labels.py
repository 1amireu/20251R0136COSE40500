import csv
from pathlib import Path

# Define paths - adjust these to match your actual file locations
PARENT_DIR = Path.cwd() 
SOURCE_CSV = PARENT_DIR / "images" / "labels.csv"  # Use exact path to your file
DEST_CSV = PARENT_DIR / "images" / "trimmed_dataset" / "trimmed_labels.csv"

# Generate the sequence
def generate_sequence(max_limit):
    indices = []
    start = 9
    while start <= max_limit:
        indices.extend([start, start + 1])
        start += 16
    return indices

# Check if source CSV exists
if not SOURCE_CSV.exists():
    print(f"Error: Source CSV file not found: {SOURCE_CSV}")
else:
    # Read original CSV and select lines based on indices
    selected_rows = []
    header = []

    with open(SOURCE_CSV, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)  # Read header
        all_rows = list(reader)
        max_index = len(all_rows)
        selected_indices = generate_sequence(max_index)
        # Adjust indices to zero-based for list access
        zero_based_indices = [i-1 for i in selected_indices if i <= max_index]

        for i in zero_based_indices:
            selected_rows.append(all_rows[i])

    # Write selected rows to new CSV
    with open(DEST_CSV, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(selected_rows)

    print(f"Created {DEST_CSV} with {len(selected_rows)} entries.")
