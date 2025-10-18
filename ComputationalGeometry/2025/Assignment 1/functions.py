from typing import List, Tuple, Dict
from segment import Segment
from status_treap import StatusTreap
from distance_oracle import DistanceOracle
from math_functions import wrap_angle
import random


def parse_input(path: str) -> Tuple[Tuple[float, float], List[Segment]]:
    """
    Parse the input file into p and S
    :param path: input file path
    :return: p and S, as a point and a list of segments
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip() != ""]
    if not lines:
        raise ValueError("input.txt is empty")

    # first line is p
    parts = lines[0].split()
    if len(parts) < 2:
        raise ValueError("First line must have px py")
    px, py = float(parts[0]), float(parts[1])

    # segments from second line to end
    segments: List[Segment] = []
    sid = 0
    for ln in lines[1:]:
        toks = ln.split()
        if len(toks) < 4:
            continue
        x1, y1, x2, y2 = map(float, toks[:4])
        # Store raw with normalized spacing for output consistency:
        raw_line = f"{x1:.12g} {y1:.12g} {x2:.12g} {y2:.12g}"
        segments.append(Segment(sid, (x1, y1), (x2, y2), raw_line, px, py))
        sid += 1

    return (px, py), segments


def build_events(segments: List[Segment]) -> Dict[float, Dict[str, List[Segment]]]:
    """
    Event queue Q keyed by angle theta in [0, 2π)
    At each event angle, we will first remove all segments whose interval ends, then insert all segments whose interval starts
    For degenerate (start==end), put them as start only
    :param segments: S
    :return: event queue
    """
    Q: Dict[float, Dict[str, List[Segment]]] = {}
    def add_event(theta: float, kind: str, s: Segment) -> None:
        """
        Add an event to the event queue
        :param theta: event angle
        :param kind: start or end
        :param s: segment
        :return: None
        """
        theta = wrap_angle(theta)
        bucket = Q.setdefault(theta, {"start": [], "end": []})
        bucket[kind].append(s)

    for s in segments:
        if s.degenerate:
            # single-angle visibility check
            add_event(s.start, "start", s)
        else:
            add_event(s.end, "end", s)
            add_event(s.start, "start", s)
    return Q


def initial_active(theta: float, segments: List[Segment]) -> List[Segment]:
    """
    Initially active segments
    :param theta: event angle
    :param segments: S
    :return: initially active segments
    """
    return [s for s in segments if s.active_at(theta)]

def run_visibility_sweep(input_path: str = "input.txt") -> Tuple[Tuple[float, float], List[Segment], List[Segment]]:
    """
    Check visibility of segments from p
    :param input_path: path of input file
    :return: p, visible segments, not visible segments
    """

    # deterministic treap priorities
    random.seed(0)

    (px, py), segments = parse_input(input_path)
    events = build_events(segments)
    thetas = sorted(events.keys())

    # Distance oracle and status treap:
    oracle = DistanceOracle(px, py)
    status = StatusTreap(oracle)

    visible_ids = set()

    # Start at θ = 0+ε to initialize status for the first wedge.
    theta = 0.0 + 1e-9
    oracle.set_theta(theta)
    for s in initial_active(theta, segments):
        status.insert(s)

    # Mark the nearest in the initial wedge [0, first_event)
    first_min = status.min_segment()
    if first_min is not None:
        visible_ids.add(first_min.id)

    # Process all event angles in [0, 2π)
    prev_theta = theta
    for th in thetas:
        # Step 1: set θ = th - ε to remove segments that end here (currently active)
        oracle.set_theta(wrap_angle(th - 1e-12))
        # Remove all 'end' segments at this angle
        for s in events[th]["end"]:
            status.remove(s)

        # Step 2: set θ = th + ε to insert segments that start here
        oracle.set_theta(wrap_angle(th + 1e-12))
        for s in events[th]["start"]:
            if s.degenerate:
                # Special case: segment active only at this angle.
                # Decide visibility by comparing its distance at th+ε
                # to the current front segment after performing removals.
                # If it is strictly the nearest, mark visible.
                d_s = oracle.ray_segment_distance(s)
                front = status.min_segment()
                d_front = oracle.ray_segment_distance(front) if front else float("inf")
                if d_s < d_front - 1e-12:
                    visible_ids.add(s.id)
                # Do not keep degenerate segments in status.
            else:
                status.insert(s)

        # Step 3: After updates, the nearest over (th, next_th) is the current min.
        front = status.min_segment()
        if front is not None:
            visible_ids.add(front.id)

        prev_theta = th

    # After the last event, the wedge (last_event, 2π) also has the current min
    if status.min_segment() is not None:
        # idempotent
        visible_ids.add(status.min_segment().id)

    visible = [s for s in segments if s.id in visible_ids]
    not_visible = [s for s in segments if s.id not in visible_ids]
    return (px, py), visible, not_visible

