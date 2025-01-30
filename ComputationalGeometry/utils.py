import os
import subprocess
from classes import *

def read_lines_from_file(file_path: str) -> list[LineSegment]:
    """
    Read line segments from file
    :param file_path: name of input file
    :return: list of LineSegments
    """
    lines = []
    with open(file_path, 'r') as file:
        contents = file.readlines()
        # Strip trailing new line
        if contents[-1].endswith('\n'):
            contents[-1] = contents[-1].rstrip('\n')
        for segment in contents:
            x1, y1, x2, y2 = map(float, segment.strip().split())
            lines.append(LineSegment(Point(x1, y1), Point(x2, y2)))
    print("Lines segments read from file")
    return lines

def read_point_from_input() -> tuple:
    """
    Read viewpoint form user input
    :return: tuple of x and y coordinates
    """
    user_input = input("Enter a point (x y): ")
    x, y = map(float, user_input.strip().split())
    return x, y

def generate_latex(visible_segments: list, not_visible_segments: list, point: (float, float)) -> str:
    """
    Generate latex content
    :param visible_segments: list of visible line segments
    :param not_visible_segments: list of not visible line segments
    :param point: viewpoint
    :return: latex content in string format
    """
    latex_content = r"""
\documentclass{standalone}
\usepackage{tikz}
\begin{document}
\begin{tikzpicture}[scale=0.5]

% Draw the lines
"""

    for segment in visible_segments:
        latex_content += segment.to_tikz(True)
    for segment in not_visible_segments:
        latex_content += segment.to_tikz(False)

    latex_content += f"%draw point\n"
    latex_content += f"\\fill[red] ({point.x}, {point.y}) circle (7pt) node[below right] {{p}};\n"

    latex_content += r"""
\end{tikzpicture}
\end{document}
"""

    return latex_content

def generate_pdf(output_file_name: str) -> None:
    """
    Starts a subprocess and calls pdflatex on .tex file to generate .pdf
    :param output_file_name: .tex file name
    :return: None
    """
    log_file_name = f"{output_file_name}.log"
    aux_file_name = f"{output_file_name}.aux"
    pdf_file_name = f"{output_file_name}.pdf"
    tex_file_name = f"{output_file_name}.tex"
    # Delete old .pdf before generating
    delete_file(pdf_file_name)
    subprocess.run(["pdflatex", tex_file_name], capture_output=True)
    # Clean up latex files after generating
    delete_file(log_file_name)
    delete_file(aux_file_name)

def save_latex(latex_code: str, output_filename: str) -> None:
    """
    Saves latex content into .tex file
    :param latex_code: latex content
    :param output_filename: name of .tex file
    :return: None
    """
    tex_file_name = f"{output_filename}.tex"
    # Delete old .tex file before generating
    delete_file(tex_file_name)
    with open(tex_file_name, "w") as file:
        file.write(latex_code)
        file.close()

    print(f"LaTeX code saved to '{output_filename}'")

def delete_file(file_name: str) -> None:
    """
    Deletes file if exists
    :param file_name: path of file to delete
    :return: None
    """
    if os.path.exists(file_name):
        os.remove(file_name)