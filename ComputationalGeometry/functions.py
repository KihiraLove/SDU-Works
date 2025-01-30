from classes import *

def create_events(segments: list, p: Point) -> list[Event]:
    """
    Create events for the start and end of line segments
    :param segments: line segments
    :param p: viewpoint
    :return: list of events
    """
    events = []
    for segment in segments:
        angle1, angle2 = segment.angle(p)
        epsilon = 1e-9
        events.append(Event(angle1 - epsilon, segment, True))
        events.append(Event(angle2 + epsilon, segment, False))
    return events

def sort_events(events: list[Event]) -> list[Event]:
    """
    Sort events by angle, prefer starting events
    :param events: list of events
    :return: sorted list of events
    """
    return sorted(events, key=lambda e: (e.angle, e.is_start))

def is_segment_hidden(new_segment: LineSegment, active_segments: list[LineSegment], p: Point) -> bool:
    """
    Check if the new segment is hidden by any of the active segments
    :param new_segment: segment to check
    :param active_segments: currently active segments
    :param p: viewpoint
    :return:
    """
    new_segment_range = new_segment.angle(p)
    for active_segment in active_segments:
        active_segment_range = active_segment.angle(p)
        # Check if angular ranges overlap
        if new_segment_range[0] <= active_segment_range[1] and new_segment_range[1] >= active_segment_range[0]:
            # If they overlap, compare distances
            if active_segment.distance(p) < new_segment.distance(p):
                return True
    return False

def determine_visibility(segments: list[LineSegment], p: Point) -> tuple[list[LineSegment], list[LineSegment]]:
    """
    Determine which segments are visible
    :param segments: list of line segments
    :param p: viewpoint
    :return: list of visible line segments, list of not visible line segments
    """
    events = create_events(segments, p)
    sorted_events = sort_events(events)

    active_segments = []
    visible_segments = []
    not_visible_segments = []

    for event in sorted_events:
        if event.is_start:
            # Check if the new segment is hidden by any active segments
            if not is_segment_hidden(event.segment, active_segments, p):
                visible_segments.append(event.segment)
            else:
                not_visible_segments.append(event.segment)
            active_segments.append(event.segment)
        else:
            # Remove the segment from the active set
            active_segments.remove(event.segment)

    return visible_segments, not_visible_segments