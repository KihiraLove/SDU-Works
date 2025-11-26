import os
from collections import defaultdict

input_file = "rl.properties"
output_dir = "data"

groups = defaultdict(list)

with open(input_file, "r", encoding="utf-8") as f:
    for raw_line in f:
        line = raw_line.strip()
        if not line:
            continue

        if "." not in line:
            continue

        key, value = line.split(".", 1)
        key = key.strip()
        value = value.strip()

        if not key or not value:
            continue

        groups[key].append(value)

for key, values in groups.items():
    filename = f"{key}.txt".replace(" ", "").replace("\\", "")
    out_path = os.path.join(output_dir, filename)
    with open(out_path, "w", encoding="utf-8") as out_f:
        for value in values:
            out_f.write(f"{key}.{value}\n")