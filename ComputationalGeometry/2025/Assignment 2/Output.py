import os
import math
import subprocess
from PlanningResult import PlanningResult
from Geometric import Geometric
from Logger import Logger
from Configuration import Configuration

class Output:
    """
    Manages creating and saving of output
    """
    def __init__(self, logger: Logger, result: PlanningResult, config: Configuration) -> None:
        self.logger = logger
        self.config = config
        self.result = result
        self.filepath = os.path.join(self.config.output_dir, self.config.output_file_name)
        self.log_file_path = f"{self.filepath}.log"
        self.aux_file_path = f"{self.filepath}.aux"
        self.pdf_file_path = f"{self.filepath}.pdf"
        self.tex_file_path = f"{self.filepath}.tex"
        self.png_file_path = f"{self.filepath}.png"
        self.report_content = ""


    def generate_and_save(self) -> None:
        """
        Generate and save enabled output formats
        :return: None
        """
        if self.config.enable_latex:
            self.save_latex(self.create_latex_document())
            self.generate_pdf()
        if self.config.enable_matplotlib:
            self.plot_paths_matplotlib()
        if self.config.enable_detailed_report:
            self.construct_detailed_report()
            self.logger.info("Detailed report:")
            self.logger.info(self.report_content)

    def create_latex_document(self) -> str:
        """
        Generate a LaTeX document with representation of the environment and paths using Tikz

        The picture contains:
        All obstacle polygons, filled in light gray and outlined in black
        The visibility-graph shortest path, drawn as a thick blue polyline
        The grid-based approximate path, drawn as a thick red dashed polyline
        Start and goal points, drawn as small circles
        :return: Assembled LaTeX document
        :rtype: str
        """

        def latex_document_begin() -> str:
            """
            LaTeX document beginning
            :return: string of LaTeX document beginning
            :rtype: str
            """
            return ("\\documentclass{standalone}\n"
                    "\\usepackage{tikz}\n"
                    "\\begin{document}\n"
                    "\\begin{tikzpicture}[scale=1.0]\n"
                    "\n% TikZ representation of the environment and paths\n")

        def latex_document_end() -> str:
            """
            LaTeX document end
            :return: string of LaTeX document end
            :rtype: str
            """
            return ("\\end{tikzpicture}\n"
                    "\\end{document}")

        # Assemble LaTeX document
        latex_document = latex_document_begin()
        latex_document += self.generate_tikz_content()
        latex_document += latex_document_end()

        self.logger.debug("LaTeX document assembled as follows:\n" + latex_document)
        return latex_document

    def save_latex(self, latex_document: str) -> None:
        """
        Save LaTeX document into .tex file
        :param latex_document:
        :return:
        """
        # Delete old .tex file before generating new one
        self.delete_file(self.tex_file_path)
        self.write_to_file(content=latex_document, file_path=self.tex_file_path)
        self.logger.info(f"LaTeX document saved to {self.tex_file_path}")

    def generate_tikz_content(self) -> str:
        """
        Generate a TikZ representation of the environment and paths
        :return: string of TikZ content
        :rtype: str
        """
        tikz_content = ""

        # Draw obstacles
        for polygon in self.result.env.obstacles:
            tikz_content += polygon.to_tikz()

        # Draw visibility-graph path
        if self.result.vg_path is not None and len(self.result.vg_path) >= 2:
            tikz_content += self.result.vg_path_to_tikz()

        # Draw grid-based path
        if self.result.grid_path is not None and len(self.result.grid_path) >= 2:
            tikz_content += self.result.grid_path_to_tikz()

        # Draw start and goal
        tikz_content += self.result.start.to_tikz("green")
        tikz_content += self.result.goal.to_tikz("magenta")
        self.logger.info(f"TikZ content generated")

        return tikz_content

    def generate_pdf(self) -> None:
        """
        Starts a subprocess and calls pdflatex on .tex file to generate .pdf
        :return: None
        """
        # Delete old .pdf before generating
        self.delete_file(self.pdf_file_path)

        self.logger.info(f"Calling pdflatex subprocess on {self.tex_file_path} to generate pdf")
        subprocess.run(["pdflatex", self.tex_file_path], capture_output=True)

        # Clean up latex files after generating
        self.delete_file(self.log_file_path)
        self.delete_file(self.aux_file_path)

    def plot_paths_matplotlib(self) -> None:
        """
        Create a matplotlib figure showing environment and paths

        The plot contains:
        Obstacle polygons, filled lightly and outlined
        Visibility-graph shortest path as a blue polyline
        Grid-based approximate path as a red dashed polyline
        Start and goal points as markers
        :return: None
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Polygon as MplPolygon
        except ImportError:
            self.logger.warn("Matplotlib is not available. Install it with 'pip install matplotlib' to enable plotting.")
            return

        self.logger.info(f"Creating matplotlib figure and saving to: {self.png_file_path}")

        fig, ax = plt.subplots()

        # Draw obstacles.
        for poly in self.result.env.obstacles:
            pts = [(p.x, p.y) for p in poly.vertices]
            patch = MplPolygon(pts, closed=True, facecolor="lightgray", edgecolor="black")
            ax.add_patch(patch)

        # Draw visibility-graph path.
        if self.result.vg_path is not None and len(self.result.vg_path) >= 2:
            xs = [p.x for p in self.result.vg_path]
            ys = [p.y for p in self.result.vg_path]
            ax.plot(xs, ys, "-o", color="blue", linewidth=2, label="Visibility path")

        # Draw grid-based path.
        if self.result.grid_path is not None and len(self.result.grid_path) >= 2:
            xs = [p.x for p in self.result.grid_path]
            ys = [p.y for p in self.result.grid_path]
            ax.plot(xs, ys, "--o", color="red", linewidth=2, label="Grid path")

        # Draw start and goal.
        ax.plot(self.result.start.x, self.result.start.y, "o", color="green", markersize=8, label="Start")
        ax.plot(self.result.goal.x, self.result.goal.y, "o", color="magenta", markersize=8, label="Goal")

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Robot motion planning: visibility graph vs. grid-based path")
        ax.grid(True)
        ax.legend()

        fig.savefig(self.png_file_path, bbox_inches="tight", dpi=150)
        self.logger.info(f"Matplotlib figure saved to: {self.png_file_path}")
        if self.config.enable_interactive:
            self.logger.info("Showing figure interactively.")
            plt.show()
        self.logger.debug("Matplotlib plotting finished.")

    def construct_detailed_report(self) -> None:
        """
        Construct a detailed explanation of the planning results
        This function is intended to serve as a textual summary that explains
        what the algorithms did and how to interpret the numerical outputs

        The report includes:
        Basic environment statistics (number of obstacles, number of vertices)
        Visibility-graph statistics (number of vertices, number of edges)
        Shortest-path properties (length, number of segments, vertex sequence)
        Grid-based planner parameters and path properties
        A direct comparison between exact and approximate paths
        :return: None
        """
        def append_to_content(new_line: str) -> None:
            self.report_content += new_line + "\n"

        def append_empty_line() -> None:
            self.report_content += "\n"

        num_obstacles = len(self.result.env.obstacles)
        num_vertices = len(self.result.env.all_vertices())
        vg_n = len(self.result.vg_planner.vertices)
        vg_m = self.result.vg_planner.edge_count()

        geometric = Geometric()

        append_to_content("Printing detailed numerical report\n")

        # Environment summary
        append_to_content("=== Environment summary ===\n"
                          f"Number of obstacles                 : {num_obstacles}\n"
                          f"Total number of obstacle vertices   : {num_vertices}\n"
                          f"Start point                         : ({self.result.start.x:.3f}, {self.result.start.y:.3f})\n"
                          f"Goal point                          : ({self.result.goal.x:.3f}, {self.result.goal.y:.3f})\n"
        )

        # Visibility graph summary.
        append_to_content("=== Visibility graph planner (exact shortest path) ===\n"
                          "The visibility graph has one vertex for each obstacle vertex plus\n"
                          "the start and goal points. Two vertices are connected if the\n"
                          "segment between them does not intersect any obstacle.\n"
                         f"Number of visibility graph vertices : {vg_n}\n"
                         f"Number of visibility graph edges    : {vg_m}"
        )
        append_empty_line()

        if self.result.vg_path is None:
            append_to_content("No path was found in the visibility graph (graph is disconnected).")
        else:
            vg_len = geometric.polyline_length(self.result.vg_path)
            append_to_content(f"Shortest path length                : {vg_len:.6f}\n"
                              f"Number of path segments             : {max(0, len(self.result.vg_path) - 1)}\n"
                               "Path vertex sequence (in order):"
            )
            for idx, p in enumerate(self.result.vg_path):
                append_to_content(f"  {idx:2d}: ({p.x:.6f}, {p.y:.6f})")
            append_empty_line()

        # Grid-based planner summary.
        append_to_content("=== Grid-based planner (uniform grid / quadtree) ===\n"
                          "The plane is discretized into a uniform grid. Each cell is marked as\n"
                          "free or blocked based on its center. A BFS search is performed from\n"
                          "the start cell to the goal cell, restricted to free cells.\n"
                         f"Grid size                            : {self.result.grid_planner.grid_size} x {self.result.grid_planner.grid_size}\n"
                         f"Bounding square left/bottom          : ({self.result.grid_planner.left:.3f}, {self.result.grid_planner.bottom:.3f})\n"
                         f"Cell size                            : {self.result.grid_planner.cell_size:.6f}"
        )

        if self.result.grid_path is None:
            append_to_content("No grid-based path was found. Reasons may include:\n"
                              "The start or goal cell is inside an obstacle at this resolution.\n"
                              "The grid resolution is too coarse to preserve connectivity."
            )
        else:
            grid_len = geometric.polyline_length(self.result.grid_path)
            append_to_content(f"Grid-based path length (approximate) : {grid_len:.6f}\n"
                              f"Number of grid steps                 : {max(0, len(self.result.grid_path) - 1)}\n"
                              "Path cell-center sequence (in order):"
            )
            for idx, p in enumerate(self.result.grid_path):
                append_to_content(f"  {idx:2d}: ({p.x:.6f}, {p.y:.6f})")
        append_empty_line()

        # Comparison
        append_to_content("=== Comparison of exact vs. approximate path ===")
        if self.result.vg_path is None or self.result.grid_path is None:
            append_to_content("A direct numerical comparison is not possible because at least one of the planners did not find a path."
            )
        else:
            vg_len = geometric.polyline_length(self.result.vg_path)
            grid_len = geometric.polyline_length(self.result.grid_path)
            ratio = grid_len / vg_len if vg_len > 0 else math.inf
            append_to_content(f"Exact (visibility graph) path length : {vg_len:.6f}\n"
                             f"Approx (grid) path length            : {grid_len:.6f}\n"
                             f"Approx / exact length ratio          : {ratio:.6f}\n"
                              "The ratio quantifies how much longer the grid-based path is\n"
                              "compared to the true shortest path. Values closer to 1 indicate\n"
                              "a high-quality approximation; larger values suggest that the grid\n"
                              "resolution or methodology may need refinement."
            )
        append_empty_line()

    def write_to_file(self, content: str, file_path: str) -> None:
        """
        Write content to a file
        :param content: content to write
        :type content: str
        :param file_path: path of file
        :type file_path: str
        :return:
        """
        with open(file_path, "w", encoding="utf8") as file:
            file.write(content)
            file.close()
            self.logger.info(f"Content written to {file_path}")

    def delete_file(self, file_path: str) -> None:
        """
        Deletes file if exists
        :param file_path: path of file to delete
        :type file_path: str
        :return: None
        """
        if os.path.exists(file_path):
            os.remove(file_path)
            self.logger.info(f"{file_path} deleted.")