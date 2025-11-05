import pickle
import numpy as np
import math
from dataclasses import dataclass
from typing import Any, Self
from data_manager import IS_STREET_INDEX
from .data_structures import Point, StreetBorder




class FoundStreets:

    def __init__(self):
        self.streets: list[Street] = []
        self.disruptions: dict[Street, list[Point]] = {} # list of problematic points by their strets
        self.crossroads: list[Crossroad]

type Grid = np.ndarray[(Any, Any, Any), Any]

def is_border_point(grid_part: Grid, point: Point):
    y, x = point
    neighborhood = grid_part[y - 1 : y + 2, x - 1 : x + 2, IS_STREET_INDEX]
    return neighborhood.sum() < neighborhood.size

BFSGridNode = tuple[Point, Point] # origin and actual point

def _queue_neighbors_if_necessary(node: BFSGridNode, grid_part, queue: list[BFSGridNode], visited):

    _, point = node
    y, x = point

    checked_coords = [
        (y-1, x),
        (y+1, x),
        (y, x-1),
        (y, x+1)
    ]
    for n_y, n_x in checked_coords:
        # queued are points which are:
        # * normal street points, if they have not been already visited
        # * border points (always), since they may be reached from different
        # borders what means, that these borders need to be merged
        if grid_part[y, x, IS_STREET_INDEX] == 1 and (n_y, n_x) not in visited:
            queue.append((point, (n_y, n_x)))
        elif is_border_point(grid_part, (n_y, n_x)):
            queue.append((point, (n_y, n_x)))


def discover_street_borders(grid_part: Grid, handle: Point):
    queue: list[BFSGridNode] = [(None, handle)] # (origin, point to visit)
    visited: list[Point] = []
    borders: list[StreetBorder] = []
    borders_of_points: dict[Point, StreetBorder] = {}
    
    while queue:
        node = queue.pop(0)
        origin, point = node

        # check, if borders need to be merged
        if point not in visited:
            # handle typical case
            visited.append(point)
            if is_border_point(grid_part, point):
                if origin in borders_of_points.keys():
                    border = borders_of_points[origin]
                    border.appendChild(origin, point)
                    borders_of_points[point] = border
                else: # new border part
                    new_border = StreetBorder(origin)
                    borders.append(new_border)
                    borders_of_points[point] = new_border
        else: 
            current_border = borders_of_points[point]
            removed_border = borders_of_points[origin]
            if current_border != removed_border:
                removed_border.appendChild(origin, point) # insert merging point
                current_border.merge(removed_border, point, inplace=True)
                # rebind points to new border after merging
                for p in borders_of_points.keys():
                    if borders_of_points[p] == removed_border:
                        borders_of_points[p] = current_border
                # remove deleted
                borders.remove(removed_border)
            

        _queue_neighbors_if_necessary(node, grid_part, queue, visited)
    return borders, borders_of_points, visited
    

def find_first_non_checked_street(grid: Grid, are_checked: np.ndarray[(Any, Any), bool]) -> tuple[bool, tuple[int, int]]:

    for (y, x), value in np.ndenumerate(are_checked):
        if value == 0:
            are_checked[y, x] = 1
            if grid[y, x, IS_STREET_INDEX] == True:
                return True, (x, y), value
    return False, None

def identify_borders(grid_part: Grid):
    are_checked = np.zeros(grid_part, dtype=bool)
    is_found, (row, col) = find_first_non_checked_street(grid_part)
    borders: list[StreetBorder] = []
    borders_of_points: dict[Point, StreetBorder] = {}
    while is_found:
        new_borders, new_borders_of_points, visited = discover_street_borders(grid_part, (row, col))
        borders += new_borders
        borders_of_points.update(new_borders_of_points)
        # mark all visited fields 
        for checked_y, checked_x in visited:
            are_checked[checked_y, checked_x] = 1
        # find next
        is_found, (row, col) = find_first_non_checked_street(grid_part)
    return borders, borders_of_points

@dataclass
class Street:
    id: int
    points: list[Point]

@dataclass
class Crossroad:
    streets: list[int] # ids

def _find_opposite_border(grid_part: Grid, borders: list[StreetBorder], borders_of_points: dict[Point, StreetBorder], point: Point):
    pass

step = 5

def identify_streets(grid_part: Grid, borders: list[StreetBorder], borders_of_points: dict[Point, StreetBorder]):
    for border in borders:
        points = border.to_list()
        for point in points[::step]:
            #TODO
            # _find_opposite_border()
            pass

overlap = 1

def divide_into_parts(grid: Grid, part_size: int = 200):
    width, height = grid.shape
    x_parts_number = math.ceil(width / part_size)
    y_parts_number = math.ceil(height / part_size)

    results = []
    for row in range(x_parts_number):
        # lower/upper in the sense of indices values
        # parts overlap by 1 row
        y_lower = max(0, row * part_size - overlap)
        y_upper = min(0, (row + 1) * part_size + overlap)
        for col in range(y_parts_number):
            # parts overlap by 1 column
            x_lower = max(0, col * part_size - overlap)
            x_upper = min(width, (col + 1) * part_size + overlap)
            results.push(identify_streets(grid[y_lower:y_upper, x_lower:x_upper]))
    # and so on...