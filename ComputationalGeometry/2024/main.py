from utils import *
from functions import determine_visibility

###
# author: doker24, Domonkos Kertész
# Computational geometry, fall 2024
# also available: https://github.com/KihiraLove/ComputationalGeometry
###

### Important ###
# The script calls a subprocess of your terminal to generate a .pdf,
# it required pdflatex to be available
# I only tested it on one machine, if it fails on yours, the .tex file is available to be used manually

# input file has to be in same dir as script, but filepath is also accepted
input_file_name = "input_int_small.txt"
# name of generated .tex and .pdf files
output_file_name = "lines_visual"

lines = read_lines_from_file(input_file_name)
point = Point(0, 0)
# point = read_point_from_input()

visible_segments, not_visible_segments = determine_visibility(lines, point)

for seg in visible_segments:
    print(f"Segment from ({seg.p1.x}, {seg.p1.y}) to ({seg.p2.x}, {seg.p2.y})")

latex_code = generate_latex(visible_segments, not_visible_segments, point)
save_latex(latex_code, output_file_name)
generate_pdf(output_file_name)