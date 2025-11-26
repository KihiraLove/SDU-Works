import os

data_dir = "data"
output_file = "merged.txt"

# Open (and overwrite) the output file
with open(output_file, "w", encoding="utf-8") as out_f:
    # Go through files in data directory in sorted order
    for filename in sorted(os.listdir(data_dir)):
        if not filename.endswith(".txt"):
            continue

        path = os.path.join(data_dir, filename)
        if not os.path.isfile(path):
            continue

        with open(path, "r", encoding="utf-8") as in_f:
            for line in in_f:
                # Normalize line endings
                out_f.write(line.rstrip("\n") + "\n")