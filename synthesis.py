import json
import pprint

file = './arc-agi_training_challenges.json'
with open (file, 'r') as f:
    data = json.load(f)
    

problem_keys = list(data.keys())

# p1 = data['007bbfb7']
print(problem_keys[10])
p1 = data[problem_keys[10]]

solved = [
    10
]

def copy_grid(grid, empty=False):
    new_grid = []
    for row in grid:
        if empty:
            new_grid.append([0 for _ in row])
        else:
            new_grid.append([el for el in row])
    return new_grid

def scale_grid(grid, n=3):
    """scales the entire grid"""
    n_rows = len(grid)
    n_cols = len(grid[0])
    new_grid = [[0] * (n_cols * n) for _ in range(n_rows * n)]

    for i in range(n_rows):
        for j in range(n_cols):
            val = grid[i][j]
            # paint
            s_i_start = i * n
            s_j_start = j * n
            for s_i in range(s_i_start, s_i_start + n):
                for s_j in range(s_j_start, s_j_start + n):
                    new_grid[s_i][s_j] = grid[i][j]

    return new_grid

def scale_shape_in_grid():
    """Keep the same dim for grid, scales the shape only"""
    pass

"""
ALTERNATIVE
paint shape
bool_vector of shape and paint shape into grid
1 1 1 
0 0 0
0 1 1
"""

def unishape(grid, split_rows, split_cols):
    grid = copy_grid(grid)
    n_rows = len(grid)
    n_cols = len(grid[0])
    assert n_rows % split_rows == 0
    assert n_cols % split_cols == 0

    shapes = []

    for i in range(n_rows):
        for j in range(n_cols):
            shape_num = i // split_rows

    # strategy 1: create a grid of shapes
    """
    that way you can match a bool vector of the shape to the grid
    bool_vector of shape and paint shape into grid
    1 1 1 
    0 0 0
    0 1 1
    """

    # s2: iterate by shape coords
    for i in range(0, n_rows, split_rows):
        for j in range(0, n_cols, split_cols):
            shape = []
            # build shape
            for subi in range(i, i+split_rows):
                for subj in range(j, j+split_cols):
                    shape.append((subi, subj))

            shapes.append(shape)

    return shapes

def create_bool_mask_grid(grid):
    grid = copy_grid(grid)
    n_rows = len(grid)
    n_cols = len(grid[0])
    for i in range(n_rows):
        for j in range(n_cols):
            if grid[i][j] != 0: 
                grid[i][j] = 1
    return grid

def apply_paint_mask_to_grid(grid, paint_mask, wall_color=None):
    grid = copy_grid(grid)
    n_rows = len(grid)
    n_cols = len(grid[0])
    for i in range(n_rows):
        for j in range(n_cols):
            if wall_color is not None and grid[i][j] == wall_color:
                continue
            grid[i][j] = paint_mask[i][j]
    return grid

def duplicate_into_bool_grid(bool_grid, pattern):
    pat_n_rows = len(pattern)
    pat_n_cols = len(pattern[0])
    """
    1 1 1
    0 0 0
    0 1 1
    """
    bool_n_rows = len(bool_grid)
    bool_n_cols = len(bool_grid[0])
    """
    1 1 1
    0 0 0
    0 1 1
    """
    result_n_rows = pat_n_rows * bool_n_rows
    result_n_cols = pat_n_cols * bool_n_cols
    result_grid = [[0] * result_n_cols for _ in range(result_n_rows)]

    for i in range(0, result_n_rows, pat_n_rows):
        for j in range(0, result_n_cols, pat_n_cols):
            bool_grid_row = i // pat_n_rows
            bool_grid_col = j // pat_n_cols
            bool_grid_val = bool_grid[bool_grid_row][bool_grid_col]
            if bool_grid_val == 0:
                continue

            for paint_i in range(i, i + pat_n_rows):
                for paint_j in range(j, j + pat_n_cols):
                    modi = paint_i % pat_n_rows
                    modj = paint_j % pat_n_cols
                    pat_val = pattern[modi][modj]
                    result_grid[paint_i][paint_j] = pat_val

    return result_grid

def find_grid_in_grid(grid1, grid2):
    """
    TODO: improvements
    - exhaust so theres no overlaps
    - what if the colors changed? handle color changes
    """
    g1_n_rows = len(grid1)
    g1_n_cols = len(grid1[0])
    g2_n_rows = len(grid2)
    g2_n_cols = len(grid2[0])

    def traverse(i, j):
        for g1_i in range(g1_n_rows):
            for g1_j in range(g1_n_cols):
                adj_i = i + g1_i
                adj_j = j + g1_j
                g1_val = grid1[g1_i][g1_j]
                g2_val = grid2[adj_i][adj_j]
                if g1_val != g2_val:
                    return False
        return True

    starting_pos = []

    for i in range(g2_n_rows - g1_n_rows + 1):
        for j in range(g2_n_cols - g1_n_cols + 1):
            if grid2[i][j] == grid1[0][0]:
                if traverse(i, j):
                    starting_pos.append((i, j))
    return starting_pos

def find_grid_in_grid_partial(grid1, grid2):
    """
    TODO: improvements
    - only handles one color
    - exhaust so theres no overlaps
    - what if the colors changed? handle color changes
    """
    g1_n_rows = len(grid1)
    g1_n_cols = len(grid1[0])
    g2_n_rows = len(grid2)
    g2_n_cols = len(grid2[0])

    color = None
    def traverse(i, j):
        nonlocal color
        added_to_g1 = set()
        for g1_i in range(g1_n_rows):
            for g1_j in range(g1_n_cols):
                adj_i = i + g1_i
                adj_j = j + g1_j
                # print(f"adjij: {adj_i}-{adj_j}")
                g1_val = grid1[g1_i][g1_j]
                g2_val = grid2[adj_i][adj_j]
                if g1_val == 0: 
                    if g2_val != 0:
                        color = g2_val
                        added_to_g1.add((adj_i, adj_j))
                else:
                    # here g1_val != 0
                    if g2_val == 0:
                        # lost a color
                        return False
                    else:
                        # here g1_val != 0 and g2_val != 0
                        pass
        return added_to_g1

    starting_pos = []
    added_to_g1 = set()
    for i in range(g2_n_rows - g1_n_rows + 1):
        for j in range(g2_n_cols - g1_n_cols + 1):
            if grid2[i][j] == grid1[0][0]:
                from_traverse = traverse(i, j)
                if from_traverse:
                    added_to_g1.update(from_traverse)
                    starting_pos.append((i, j))

    return starting_pos, list(added_to_g1), color

def find_grid_in_grid_partial_shapes(grid1, grid2):
    """
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0]
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 1, 1, 0, 4, 0, 0, 0, 0, 0]
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ---
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0]
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0]
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0]
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0]
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0]
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    [2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 4]
    [2, 2, 0, 2, 2, 2, 0, 2, 2, 2, 0, 1, 1, 1, 0, 4, 4, 4, 0, 4, 4]
    [2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 4]
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    """
    # partial shapes = []
    # first find shapes in grid1
    # for each shape in grid1, find it in grid2
    #   if entire shape is found -> pass
    #   if partial shape is found -> add to partial shapes
    # return partial_shapes
    pass

def get_non_zeroes(grid):
    count = 0
    n_rows = len(grid)
    n_cols = len(grid[0])
    for i in range(n_rows):
        for j in range(n_cols):
            if grid[i][j] != 0:
                count += 1
    return count

def find_8d_shapes(grid):
    shapes = []
    visited = set()
    def traverse(i, j):
        if (i, j) in visited:
            return
        visited.add((i, j))
        shapes[-1].append((i, j))

        # go in 8 directiosn
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            newi, newj = i + di, j + dj
            if 0 <= newi < len(grid) and 0 <= newj < len(grid[0]) and grid[newi][newj] != 0:
                traverse(newi, newj)

    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] != 0:
                shapes.append([])
                traverse(i, j)
    return shapes

# def check_void_fill(point, grid):
#     target_color = None
#     def traverse(i, j):
#         # if the point is adjacent to the border, return False
#         if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]):
#             return False
#     traverse(point[0], point[1])
#     return grid

def create_void_fill_mask(points, grid):
    mask = copy_grid(grid, empty=True)
    visited = set()
    def traverse(i, j):
        if (i, j) in visited:
            return
        if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]):
            return

        if grid[i][j] == 0:
            visited.add((i, j))
            mask[i][j] = 1
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                traverse(i + di, j + dj)

    for point in points:
        traverse(point[0], point[1])
    return mask

def color_grid(grid, mask, color):
    grid = copy_grid(grid)
    n_rows = len(grid)
    n_cols = len(grid[0])
    for i in range(n_rows):
        for j in range(n_cols):
            if mask[i][j] == 1:
                grid[i][j] = color
    return grid

def find_walls(grid):
    """
    walls should ONLY be in vertical and horizontal lines
    wrote solving: 09629e4f
    """
    """Can there be more than 1 wall color?"""
    n_rows = len(grid)
    n_cols = len(grid[0])
    # is there any line that fully extends vertically or horizontally?
    walls = {
        "horizontal": {},
        "vertical": {},
    }
    dominant_wall_color = None

    cannot_be_wall = set()

    for i in range(n_rows):
        j_counter = {}
        for j in range(n_cols):
            grid_val = grid[i][j]
            j_counter[grid_val] = j_counter.get(grid_val, 0) + 1
        
        for key in j_counter:
            if j_counter[key] == n_cols and key in cannot_be_wall:
                print("🍅 there is a horizontal wall", i, "of color", key)
                walls["horizontal"][i] = key
                dominant_wall_color = key
            else:
                cannot_be_wall.add(key)

        # for j in range(n_cols):
        #     if j == 0:
        #         wall_element = grid[i][j]
        #     else:
        #         if grid[i][j] != wall_element:
        #             is_wall = False
        #             break
        # if is_wall:
        #     print("🍅 there is a horizontal wall", i, "of color", wall_element)
        #     walls["horizontal"][i] = wall_element
        #     dominant_wall_color = wall_element

    for j in range(n_cols):
        i_counter = {}
        for i in range(n_rows):
            grid_val = grid[i][j]
            i_counter[grid_val] = i_counter.get(grid_val, 0) + 1

        for key in i_counter:
            if i_counter[key] == n_rows and key in cannot_be_wall:
                print("🍅 there is a vertical wall", j, "of color", key)
                walls["vertical"][j] = key
                dominant_wall_color = key
            else:
                cannot_be_wall.add(key)

        #     if i == 0:
        #         wall_element = grid[i][j]
        #     else:
        #         if grid[i][j] != wall_element:
        #             is_wall = False
        #             break
        # if is_wall:
        #     print("🍅 there is a vertical wall", j, "of color", wall_element)
        #     walls['vertical'][j] = wall_element
        #     dominant_wall_color = wall_element

    # print(f"dominant_wall_color={dominant_wall_color}")
    return walls, dominant_wall_color

def get_wall_color_mask(grid, wall_color):
    """
    wrote solving: 09629e4f
    """
    mask = copy_grid(grid, empty=True)
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == wall_color:
                mask[i][j] = wall_color
    return mask

def find_walled_group_positions(grid, wall_color):
    """
    wrote solving: 09629e4f
    """
    grid = copy_grid(grid)

    n_rows = len(grid)
    n_cols = len(grid[0])

    wall_group_positions = []

    group_set = None

    def traverse(i, j):
        if (i,j) in group_set:
            return
        if i < 0 or i >= n_rows or j < 0 or j >= n_cols:
            return
        if grid[i][j] == wall_color:
            return

        group_set.add((i,j))
        grid[i][j] = wall_color
        traverse(i + 1, j)
        traverse(i - 1, j)
        traverse(i, j + 1)
        traverse(i, j - 1)
    
    for i in range(n_rows):
        for j in range(n_cols):
            if grid[i][j] != wall_color:
                group_set = set()
                traverse(i, j)
                new_group = sorted(list(group_set))
                wall_group_positions.append(new_group)
    return wall_group_positions


def count_values(values, all_nums=True):
    """
    wrote solving: 09629e4f
    """
    if all_nums:
        value_counts = {i: 0 for i in range(10)}
    else:
        value_counts = {}

    for value in values:
        assert value in value_counts
        value_counts[value] += 1

    return value_counts

from collections import Counter
def get_least_numerous_index(values) -> int:
    """
    wrote solving: 09629e4f
    return the index of the value that is most numerous
    """
    # create a counter
    # get the min value
    # if there are more than one, return -1
    # else return index
    value_counts = Counter(values)
    min_value = min(value_counts, key=value_counts.get)

    min_value_found_count = 0
    for value, count in value_counts.items():
        if value == min_value:
            min_value_found_count += 1

    if min_value_found_count > 1:
        return -1
    else:
        return values.index(min_value)


def find_walled_group_values(wall_group_positions, input_grid, output_grid):
    """
    wrote solving: 09629e4f
    """
    result = []

    for i, group_pos in enumerate(wall_group_positions):
        input_group_values = []
        output_group_values = []
        for row, col in group_pos:
            input_group_values.append(input_grid[row][col])
            output_group_values.append(output_grid[row][col])

        input_value_counts = count_values(input_group_values)
        output_value_counts = count_values(output_group_values)

        group_result = {
            "input_group_values": input_group_values,
            "input_group_values_count_non_zero": sum([1 for value in input_group_values if value != 0]),
            "output_group_values": output_group_values,
            # "output_group_values_count_non_zero": sum([1 for value in output_group_values if value != 0]),
            "input_value_counts": input_value_counts,
            "output_value_counts": output_value_counts,
            "input_most_numerous_value": max(input_value_counts, key=input_value_counts.get),
            "input_least_numerous_value": min(input_value_counts, key=input_value_counts.get),
            "output_most_numerous_value": max(output_value_counts, key=output_value_counts.get),
            "output_least_numerous_value": min(output_value_counts, key=output_value_counts.get),
            "size": len(input_group_values),
            "size_non_zero": len([value for value in input_group_values if value != 0]),
        }
        # most numerous value non zero
        input_value_counts_non_zero = {value: count for value, count in input_value_counts.items() if value != 0}
        if input_value_counts_non_zero:
            group_result["input_most_numerous_value_non_zero"] = max(input_value_counts_non_zero, key=input_value_counts_non_zero.get)
            group_result["input_least_numerous_value_non_zero"] = min(input_value_counts_non_zero, key=input_value_counts_non_zero.get)
        else:
            group_result["input_most_numerous_value_non_zero"] = None
            group_result["input_least_numerous_value_non_zero"] = None

        output_value_counts_non_zero = {value: count for value, count in output_value_counts.items() if value != 0}
        if output_value_counts_non_zero:
            group_result["output_most_numerous_value_non_zero"] = max(output_value_counts_non_zero, key=output_value_counts_non_zero.get)
            group_result["output_least_numerous_value_non_zero"] = min(output_value_counts_non_zero, key=output_value_counts_non_zero.get)
        else:
            group_result["output_most_numerous_value_non_zero"] = None
            group_result["output_least_numerous_value_non_zero"] = None

        result.append(group_result)

    non_zero_values_count_list = [group_result["input_group_values_count_non_zero"] for group_result in result]
    least_non_zero_values_index = get_least_numerous_index(non_zero_values_count_list)
    has_least_numerous_value = least_non_zero_values_index != -1
    least_non_zero_group = result[least_non_zero_values_index]['input_group_values'] if has_least_numerous_value else None
    # TODO search over all patterns
    return {
        "walled_groups": result,
        "aggegates": {
            "input_has_least_numerous_value": has_least_numerous_value,
            "input_least_non_zero_group_values": least_non_zero_group,
            "output_most_non_zero_values": None,
            "output_most_non_zero_values_value_mask": None,
            "output_least_non_zero_values_value_mask": None,
        }
    }

"""
wall_groups: [
    [
        (0,0),
        (0,1),
        (0,2),
        (1,0),
        (1,1),
        (1,2),
    ],
    [
        (2,0),
        (2,1),
        (2,2),
        (3,0),
        (3,1),
        (3,2),
    ]
]
"""

def color_groups_as_values(input_grid, wall_group_positions, values):
    """
    "input grid" sets the right dimensions
    input_grid = [
        [0, 7, 7],
        [5, 5, 5],
        [0, 2, 1],
    ]
    "color mask" is something like [6,9]
    "groups" are like [
        [(0,0), (0,1), (0,2)], # the top row
        [(2,0), (2,1), (2,2)], # the bottom row
    ]
    mask = [
        [6, 6, 6],
        [5, 5, 5],
        [9, 9, 9],
    ]
    """
    if values is None:
        return None

    result_mask = copy_grid(input_grid, empty=True)

    # print("input_grid")
    # for row in input_grid:
    #     print(row)
    # print("wall_group_positions", wall_group_positions)
    # print("values", values)
    for i, group in enumerate(wall_group_positions):
        color_for_group = values[i]
        for row, col in group:
            result_mask[row][col] = color_for_group

    return result_mask

import copy

def examine_examples(examples):
    print("⭐️ examine examples")
    results = []
    for i, train_ex in enumerate(examples):
        print(f"📝 example {i+1}")

        train_ex_input = copy.deepcopy(train_ex['input'])
        train_ex_output = copy.deepcopy(train_ex['output'])
        for row in train_ex_input:
            print(row)
        print("---")
        for row in train_ex_output:
            print(row)
        assert train_ex_input == train_ex['input']

        input_non_zeroes = get_non_zeroes(train_ex_input)
        output_non_zeroes = get_non_zeroes(train_ex_output)

        # boolean_mask
        # treat the input as colorless boolean 1,0
        bool_mask_grid =  create_bool_mask_grid(train_ex_input)


        # dims
        input_n_rows = len(train_ex_input)
        input_n_cols = len(train_ex_input[0])
        output_n_rows = len(train_ex_output)
        output_n_cols = len(train_ex_output[0])
        dims_equal = input_n_rows == output_n_rows and input_n_cols == output_n_cols
        dims_rows_multiple = output_n_rows / input_n_rows
        dims_cols_multiple = output_n_cols / input_n_rows
        dims_rows_divides_evenly = output_n_rows % input_n_rows == 0
        dims_cols_divides_evenly = output_n_cols % input_n_cols == 0

        # grouping
        # wall separation
        walls, wall_color = find_walls(train_ex_input)
        wall_group_positions = find_walled_group_positions(train_ex_input, wall_color)
        wall_group_values = find_walled_group_values(wall_group_positions, train_ex_input, train_ex_output)
        # TODO more preds based on other values
        input_least_non_zero_group = wall_group_values['aggegates']['input_least_non_zero_group_values']
        pred_groups_colored_to_least_non_zero_group = color_groups_as_values(train_ex_input, wall_group_positions, input_least_non_zero_group)
        pred_input_to_least_non_zero_group_mask = apply_paint_mask_to_grid(train_ex_input, pred_groups_colored_to_least_non_zero_group, wall_color)

        input_in_output = find_grid_in_grid(train_ex_input, train_ex_output)
        input_in_output_partial, input_in_output_partial_added, void_mask_color = find_grid_in_grid_partial(train_ex_input, train_ex_output)

        # TODO what is this even?
        x = find_grid_in_grid_partial_shapes(train_ex_input, train_ex_output)

        # use first one for now
        void_fill_mask = create_void_fill_mask(input_in_output_partial_added, train_ex_input)
        pred_void_fill_color = color_grid(train_ex_input, void_fill_mask, void_mask_color)

        # sandbox
        # its not scale, its duplication, at a specific place
        scale_3x = scale_grid(train_ex_input, n=3)
        scale_3x_unishape_3_3 = unishape(scale_3x, 3, 3)

        bool_grid_dupe = duplicate_into_bool_grid(bool_mask_grid, train_ex_input)
        assert train_ex_input == train_ex['input']
        assert train_ex_output == train_ex['output']

        features_obj = { 
            # "train_ex_input": train_ex_input,
            # "train_ex_output": train_ex_output,
            #
            # "bool_mask_grid": bool_mask_grid,
            #
            "dims_input": f"{input_n_rows}x{input_n_cols}",
            "dims_input_n_rows": input_n_rows,
            "dims_input_n_cols": input_n_cols,
            "dims_output": f"{output_n_rows}x{output_n_cols}",
            "dims_output_n_rows": output_n_rows,
            "dims_output_n_cols": output_n_cols,
            "dims_equal": dims_equal,
            "dims_multiplier_rows": dims_rows_multiple,
            "dims_multiplier_cols": dims_cols_multiple,
            "dims_multiplier_divides_evenly_rows": dims_rows_divides_evenly,
            "dims_multiplier_divides_evenly_cols": dims_cols_divides_evenly,
            #
            "walls": walls,
            # "wall_group_positions": wall_group_positions,
            # "wall_group_values": wall_group_values,
            # "pred_groups_colored_to_least_non_zero_group": pred_groups_colored_to_least_non_zero_group,
            #
            "input_in_output": input_in_output,
            "input_in_output_partial": input_in_output_partial,
            "input_in_output_partial_added": input_in_output_partial_added,
            "void_fill_color": void_mask_color,
            #
            # "void_fill_mask": void_fill_mask,
            # "pred_void_fill_color": pred_void_fill_color,
            #
            "input_non_zeroes": input_non_zeroes,
            "output_non_zeroes": output_non_zeroes,
            "input_output_multiplier_non_zeroes": output_non_zeroes / input_non_zeroes,
            #
            # "scale_3x": scale_3x,
            # "duplicate_into_bool_grid": bool_grid_dupe,
        }
        # print("answer check")
        # if train_ex_output == scale_3x:
        #     print("  ✅ answer is scale3x")
        # else:
        #     print("  answer not scale3x")

        # if train_ex_output == bool_grid_dupe:
        #     print("  ✅ answer is bool grid dupe")
        # else:
        #     print("  answer not bool grid dupe")

        # assert train_ex_input == train_ex['input']

        print("🧠 LOGIC")

        print("dims equal?")
        example_solved = False
        pred = None

        if dims_equal:
            print("> dims equal")

            print("input in output?")
            if input_in_output:
                print("> input in output")
            elif input_in_output_partial:
                print("> input in output partial")
                print(f"added: {input_in_output_partial_added}")

                print("predict pattern: void fill")
                if pred_void_fill_color == train_ex_output:
                    print("✅ SOLVED:pred void fill color is correct")
                    print(f"pred: {pred_void_fill_color}")
                    # TODO there are other ways to complete a pattern
                    example_solved = True
                    pred = "pred void fill color"
                else:
                    print("❌ pred void fill color is incorrect")
            else:
                # print("TODO check if dims of a shape is equal to the output")
                print("input not in output")

                # walls?
                print("are walls separating groups?")
                print("wall count", len(walls['horizontal']) + len(walls['vertical']))
                for wall in walls['horizontal']:
                    print(f"horizontal wall={wall}")
                for wall in walls['vertical']:
                    print(f"vertical wall={wall}")

                # print("pred_input_to_least_non_zero_group_mask")
                # for row in pred_input_to_least_non_zero_group_mask:
                #     print(row)

                # print("train_ex_output")
                # for row in train_ex_output:
                #     print(row)

                # if pred_groups_colored_to_least_non_zero_group == train_ex_output:
                #     print("✅ pred groups colored to least non zero group is correct")
                # else:
                #     print("❌ pred groups colored to least non zero group is incorrect")
                
                if pred_input_to_least_non_zero_group_mask == train_ex_output:
                    print("✅ SOLVED: pred input to least non zero group mask is correct")
                    example_solved = True
                    pred = "pred input to least non zero group mask"
                else:
                    print("❌ pred input to least non zero group mask is incorrect")
        else:
            print("dims not equal")

            print("predict: bool grid dupe")
            if train_ex_output == bool_grid_dupe:
                print("✅ SOLVED:pred bool grid dupe is correct")
                example_solved = True
                pred = "pred bool grid dupe"
            else:
                print("❌ pred bool grid dupe is incorrect")

        print("🍄 features")
        pprint.pprint(features_obj)

        # feature_vector = [features_obj[key] for key in sorted(list(features_obj.keys()))]
        results.append((example_solved, pred))

    all_same_method = len(set([result[1] for result in results])) == 1
    return results, all_same_method

# examples_evaluated, all_same_method = examine_examples(p1['train'])
examples_evaluated, all_same_method = examine_examples(p1['train'][:2])
print("🎯 all_same_method", all_same_method)
print("🎯 examples_evaluated", all([result[0] for result in examples_evaluated]))

"""
generation ideas
📝 example 1
[0, 7, 7]
[7, 7, 7]
[0, 7, 7]
---
📝 example 2
[4, 0, 4]
[0, 0, 0]
[0, 4, 0]

output
duplication with diff bool masks -> bool mask made up of the shape
[
 [1,1,1]
]
[
 [1,1,1],
 [1,0,1],
 [1,1,1]
]
[
 [0,1,1],
 [1,1,1],
 [0,1,1]
]

duplication with rotation masks
[
 [1,90,180,270],
 [1,90,180,270],
 [1,90,180,270],
]
alt:find shapes, rotation mask
get patterns
[
  [1,1,1,1] <- says there is a repeating pattern four times
]

HUMAN BRAINSTORM
[0, 7, 7]
[7, 7, 7]
[0, 7, 7]

[0, 7, 7] * 2
[7, 7, 7]
[0, 7, 7]

pattern match horizontally, vertically
change colors
movement
"""