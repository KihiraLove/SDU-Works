import os
import subprocess
from typing import List, Tuple
from matplotlib import pyplot as plt
from segment import Segment


def create_output(p: Tuple[float, float], visible: List[Segment], not_visible: List[Segment], output_file_name: str) -> None:
    """
    Create and write output to files, generate pdf
    :param p: point
    :param visible: visible segments
    :param not_visible: not visible segments
    :param output_file_name: output file name
    :return: None
    """
    write_and_print_text_output(visible, not_visible, output_file_name)
    plt_plot(p, visible, not_visible, output_file_name)
    save_latex(generate_latex(p, visible, not_visible), output_file_name)
    generate_pdf(output_file_name)


def print_summary(summary: str) -> None:
    """
    Print summary to console
    :param summary: summary to print
    :return: None
    """
    print(summary)


def write_summary(summary: str, path: str) -> None:
    """
    Write summary to .txt file
    :param summary: summary to write
    :param path: output file path
    :return: None
    """
    with open(f"{path}.txt", "w", encoding="utf-8") as f:
        f.write(summary)


def write_and_print_text_output(visible: List[Segment], not_visible: List[Segment], path: str = "output") -> None:
    """
    Create summary and write it to console and .txt file
    :param visible: visible segments
    :param not_visible: not visible segments
    :param path: output file path
    :return: None
    """
    constructed_output = "VISIBLE:\n"
    for s in visible:
        constructed_output += f"{s.raw}\n"
    constructed_output += "NOT_VISIBLE:\n"
    for s in not_visible:
        constructed_output += f"{s.raw}\n"

    write_summary(constructed_output, path)
    print_summary(constructed_output)


def plt_plot(p: Tuple[float, float], visible: List[Segment], not_visible: List[Segment], path: str = "output") -> None:
    """
    Create and plot matplotlib figure, save it to .png and .pdf
    :param p: point
    :param visible: visible segments
    :param not_visible: not visible segments
    :param path: output file path
    :return: None
    """
    # Determine bounds
    xs = [p[0]]
    ys = [p[1]]
    for coll in (visible, not_visible):
        for s in coll:
            xs.extend([s.a[0], s.b[0]])
            ys.extend([s.a[1], s.b[1]])
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    dx = max(1e-6, (maxx - minx) * 0.05)
    dy = max(1e-6, (maxy - miny) * 0.05)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # draw segments
    for s in visible:
        ax.plot([s.a[0], s.b[0]], [s.a[1], s.b[1]], color="green", linewidth=2.0)
    for s in not_visible:
        ax.plot([s.a[0], s.b[0]], [s.a[1], s.b[1]], color="red", linewidth=1.5, linestyle='-')
    # draw p
    ax.plot([p[0]], [p[1]], marker="o", color="black", markersize=5)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(minx - dx, maxx + dx)
    ax.set_ylim(miny - dy, maxy + dy)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Segments visible from p (green) vs not visible (red)")

    fig.tight_layout()
    fig.savefig(f"{path}_plt.png", dpi=200)
    fig.savefig(f"{path}_plt.pdf")
    plt.close(fig)


def create_tikz(p: Tuple[float, float], visible: List[Segment], not_visible: List[Segment]) -> str:
    """
    Create Tikz representation of line segments
    :param p: point
    :param visible: visible segments
    :param not_visible: not visible segments
    :return: Tikz representation of line segments
    """
    def fsformat(v: float) -> str:
        """
        Format float to string
        :param v: float value
        :return: string representation
        """
        return f"{v:.12g}"

    # format point to Tikz
    tikz = f"\\fill ({fsformat(p[0])},{fsformat(p[1])}) circle (0.05);"

    # format segments to Tikz
    for s in visible:
        tikz += f"\\draw[very thick, green] ({fsformat(s.a[0])},{fsformat(s.a[1])}) -- ({fsformat(s.b[0])},{fsformat(s.b[1])});"
    for s in not_visible:
        tikz += f"\\draw[thick, red] ({fsformat(s.a[0])},{fsformat(s.a[1])}) -- ({fsformat(s.b[0])},{fsformat(s.b[1])});"

    return tikz


def generate_latex(p: Tuple[float, float], visible: list, not_visible: list) -> str:
    """
    Generate LaTeX content
    :param visible: list of visible line segments
    :param not_visible: list of not visible line segments
    :param p: viewpoint
    :return: LaTeX content in string format
    """
    latex_content = latex_document_begin()
    latex_content += create_tikz(p, visible, not_visible)
    latex_content += latex_document_end()
    return latex_content


def latex_document_begin() -> str:
    """
    LaTeX document beginning
    :return: string of LaTeX document beginning
    """
    return r"""
\documentclass{standalone}
\usepackage{tikz}
\begin{document}
\begin{tikzpicture}[x=1cm,y=1cm]

% Draw lines
"""


def latex_document_end() -> str:
    """
    LaTeX document end
    :return: string of LaTeX document end
    """
    return r"""
\end{tikzpicture}
\end{document}
"""


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