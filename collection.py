import pandas as pd
import os
import shutil


data_folder_path = "all-mias"
new_folder_path = "pgm_files"

os.makedirs(new_folder_path, exist_ok=True)
for filename in os.listdir(data_folder_path):
    if filename.endswith('.pgm'):
        source_path = os.path.join(data_folder_path, filename)
        destination_path = os.path.join(new_folder_path, filename)
        shutil.move(source_path, destination_path)


txt_file = 'all-mias/Info.txt'
data = []
with open(txt_file, 'r') as file:
    skip = True  
    for line in file:
        line = line.strip()
        if line.startswith("mdb"):
            skip = False
        if skip or line.startswith((' ', '#')):
            continue
        parts = line.split()
        while len(parts) < 7:
            parts.append(None)  # Fill missing values with None
        data.append(parts)

# Create a DataFrame
columns = ['REFNUM', 'BG', 'CLASS', 'SEVERITY', 'X', 'Y', 'RADIUS']
df = pd.DataFrame(data, columns=columns)

# Save to CSV
csv_file = 'data.csv'
df.to_csv(csv_file, index=False)
