from functions import run_visibility_sweep
from output import create_output


"""
author: doker24, Domonkos Kertész
Computational geometry, fall 2025
also available: https://github.com/KihiraLove/ComputationalGeometry
"""


### Important ###
# This program calls a subprocess on your terminal to generate a pdf,
# it requires pdflatex to be available.
# I only tested it on one machine, if it fails on yours, .tex file will be available to be used manually

### Packages ###
# Apart from standard packages, this program uses matplotlib.pyplot
# standard packages used:
# - os
# - subprocess
# - typing
# - math
# - random

# input file should be in root, but filepath is also accepted
# input file formatting rules:
# p should be in first line as
# x y
# after first line each line should contain one segment as
# x1 y1 x2 y2
input_file_name = "input.txt"

# name of generated .tex, .pdf, .txt, .png output files
# program will generate a .pdf and .tex file using LaTeX and pdflatex
# a .txt containing text representation of segments
# a _plt.png and a _plt.pdf using matplotlib.pyplot
output_file_name = "lines_visual"

# determine visibility from p
p, visible, not_visible = run_visibility_sweep(input_file_name)

# create, render, and save output
create_output(p, visible, not_visible, output_file_name)