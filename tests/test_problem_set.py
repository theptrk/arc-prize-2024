import pytest
from ProblemSet import Grid, Example, Problem, ProblemSet, Shape, Transform, Agent, check_input_output, create_solution_candidates, get_universal_solution

def sorted_equal(item_list1, item_list2):
    if isinstance(item_list1[0], list):
        return sorted([sorted(sublist) for sublist in item_list1]) == sorted([sorted(sublist) for sublist in item_list2])
    else:
        return sorted(item_list1) == sorted(item_list2)

def test_grid_init():
    grid = Grid([[1, 0], [0, 0]])
    assert grid.grid == [[1, 0], [0, 0]]

def test_grid_repr():
    grid = Grid([[1, 0], [0, 0]])
    assert repr(grid) == "Grid(2x2)"

def test_find_shapes_any_color():
    grid = Grid([[1, 0], [0, 0]])
    assert grid.find_shapes_any_color() == [[(0, 0)]]

    grid = Grid([
        [1, 0],
        [0, 0],
        [0, 1]
    ])
    assert grid.find_shapes_any_color() == [[(0, 0)], [(2, 1)]]

    line =[
        [0,0,0],
        [1,1,0],
        [0,0,0]
    ]
    grid = Grid(line)
    assert grid.find_shapes_any_color() == [[(1, 0), (1, 1)]]

    triangle_single_color =[
        [0,0,0],
        [0,1,0],
        [1,1,1]
    ]
    grid = Grid(triangle_single_color)
    assert grid.find_shapes_any_color() == [sorted([(1, 1), (2, 0), (2, 1), (2, 2)])]

    triangle_multi_4d =[
        [0,0,5],
        [0,1,0],
        [1,1,1]
    ]
    grid = Grid(triangle_multi_4d)
    assert grid.find_shapes_any_color(n_directions=4) == [[(0, 2)], [(1, 1), (2, 0), (2, 1), (2, 2)]]
    assert grid.shapes_any_color_4d == [[(0, 2)], [(1, 1), (2, 0), (2, 1), (2, 2)]]

    triangle_multi_8d =[
        [0,0,5],
        [0,1,0],
        [1,1,1]
    ]
    grid = Grid(triangle_multi_8d)
    assert sorted_equal(grid.find_shapes_any_color(), [[(0, 2), (1, 1), (2, 0), (2, 1), (2, 2)]])
    assert grid.shapes_any_color_8d == [[(0, 2), (1, 1), (2, 0), (2, 1), (2, 2)]]

def test_find_wall_among_shapes():
    pass

def test_find_all_possible_movements():
    grid = Grid([[0,0,0],
                 [0,1,0]])
    shape = Shape([(1, 1)])
    assert grid.find_all_possible_movements(shape) == [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0], [0, 1]]

    grid = Grid([[1,0,0],
                 [1,1,0]])
    shape = Shape([(0, 0), (1, 0), (1, 1)])
    assert grid.find_all_possible_movements(shape) == [[-1, -1], [-1, 0], [-1, 1], [-1, 2], [0, -1], [0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]

def test_move_shape():
    # single shape, single move
    grid = Grid([[1,0,0],
                 [1,1,0]])
    shape1 = grid.shapes_any_color_8d[0]
    assert grid.move_shape(shape1, -1, -1) == Grid([[1,0,0],
                                                    [0,0,0]])

    assert grid.move_shape(shape1, 1, 1) == Grid([[0,0,0],
                                                  [0,1,0]])

    # two shapes, multiple moves
    grid = Grid([[1,0,1],
                 [1,1,0]])
    shape1 = grid.shapes_any_color_8d[0]
    assert grid.move_shape(shape1, 1, 0) == Grid([[0,0,0],
                                                  [1,0,1]])

    first_move_shape1 = grid.move_shape(shape1, 0, -1)
    assert first_move_shape1 == Grid([[0,1,0],
                                      [1,0,0]])

    shape2 = first_move_shape1.shapes_any_color_8d[0]
    second_move_shape2 = first_move_shape1.move_shape(shape2, 0, -1)
    assert second_move_shape2 == Grid([[1,0,0],
                                       [0,0,0]])

def test_check_solution():
    input_grid = Grid([[1,0,0],
                       [1,1,0]])
    raw_transforms = [{"name": "move_all_shapes", "args": (1, 0)}]
    output_grid = Grid([[0,0,0],
                        [1,0,0]])

    assert check_input_output(input_grid, raw_transforms, output_grid)


"""
Test cases

Problem 1:
# moves 1 row down
input_grid = Grid([[1,1,0],
                   [0,0,0]])

output_grid = Grid([[0,0,0],
                    [1,1,0]])

Problem 2:
# color=1 moves 1 row down
# color=2 moves 1 row down and 1 column right
input_grid = Grid([[1,1,0],
                   [0,0,0]])

output_grid = Grid([[0,0,0],
                    [1,1,0]])

input_grid = Grid([[2,2,0],
                   [0,0,0]])

output_grid = Grid([[0,0,0],
                    [0,2,2]])

Problem 3:
# shapes consist of multiple colors move down by unique color count
input_grid = Grid([[2,1,0],
                   [0,0,0],
                   [0,0,0]])

output_grid = Grid([[0,0,0],
                    [0,0,0],
                    [2,1,0]])

# shapes consist of multiple colors move down by unique color count
input_grid = Grid([[1,1,0],
                   [0,0,0],
                   [0,0,0]])

output_grid = Grid([[0,0,0],
                    [1,1,0],
                    [0,0,0]])
"""


"""
Input grid -> transform -> compare(output grid)
"""
# @pytest.fixture
# def test_example():
#     input_grid = Grid([[1, 0], [0, 0]])
#     return Example(input_grid, output_grid=None)



# """
# What do we want to test?

# Transform apply works
# - example: give a shape (line) it moves it down 1 row
# - example: give a shape (rectangle)it moves it down 1 row

# The agent creates a list of transforms
# """
# def test_transform_apply(test_example):
#     transform = Transform()
#     grid = test_example.input
#     assert transform.apply(grid) == Grid([[0, 0], [0, 0]])


@pytest.fixture
def example():
    input_grid = Grid([[1, 2], [3, 4]])
    output_grid = Grid([[5, 6], [7, 8]])
    return Example(input_grid, output_grid)

def test_example_init(example):
    assert isinstance(example.input, Grid)
    assert isinstance(example.output, Grid)

def test_example_repr(example):
    assert repr(example) == "Example Grid(2x2)"

@pytest.fixture
def problem():
    train_examples = [Example(Grid([[1, 2]]), Grid([[3, 4]]))]
    test_examples = [Example(Grid([[5, 6]]), None)]
    return Problem(train_examples, test_examples, "test_id")

def test_problem_init(problem):
    assert len(problem.train) == 1
    assert len(problem.test) == 1
    assert problem.id == "test_id"

def test_problem_repr(problem):
    assert repr(problem) == "Problem test_id (test len=(1) train len=1)"

@pytest.fixture
def problem_set():
    raw_data = {
        "problem1": {
            "train": [{"input": [[1, 2]], "output": [[3, 4]]}],
            "test": [{"input": [[5, 6]]}]
        }
    }
    return ProblemSet(raw_data)

def test_problem_set_init(problem_set):
    assert len(problem_set.data) == 1
    assert "problem1" in problem_set.data

def test_problem_set_get(problem_set):
    problem = problem_set.get("problem1")
    assert isinstance(problem, Problem)
    assert problem.id == "problem1"

def test_problem_set_repr(problem_set):
    assert repr(problem_set) == "ProblemSet(1 problems)"



def test_create_solution_candidates_for_movement():
    input_grid = Grid([[1, 1, 0, 1],
                       [1, 0, 0, 0],
                       [0, 0, 0, 0]])
    # move 1 row down
    output_grid = Grid([[0, 0, 0, 0],
                        [1, 1, 0, 1],
                        [1, 0, 0, 0]])

    candidates = create_solution_candidates(input_grid, output_grid)
    target_candidates = [[{"name": "move_all_shapes", "args": [1, 0]}]]
    assert sorted_equal(candidates, target_candidates)

    input_grid = Grid([[0, 0, 0, 0],
                       [1, 0, 0, 1],
                       [0, 0, 0, 0]])
    # move 1 row up
    output_grid = Grid([[1, 0, 0, 1],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]])

    candidates = create_solution_candidates(input_grid, output_grid)
    target_candidates = [[{"name": "move_all_shapes", "args": [-1, 0]}]]
    assert sorted_equal(candidates, target_candidates)

# TODO: test for nuance movement
def xtest_create_solution_candidates_for_nuance_movement():
    input_grid = Grid([[1, 1, 0, 0],
                       [1, 0, 1, 0],
                       [0, 0, 0, 0]])

    output_grid = Grid([[0, 0, 0, 0],
                        [1, 1, 0, 0],
                        [1, 0, 1, 0]])
    candidates = create_solution_candidates(input_grid, output_grid)
    RED_COLOR = 1
    BLUE_COLOR = 2
    target_candidates = [
        [
            [
                {"name": "move_shape", "args": (RED_COLOR, 1, 0)},
                {"name": "move_shape", "args": (BLUE_COLOR, 1, 1)}
            ]
        ]
    ]
    assert sorted_equal(candidates, target_candidates)

# def test_get_universal_solution():
#     train_examples = [Example(Grid([[1, 0]]), Grid([[0, 1]]))]
#     test_examples = [Example(Grid([[0, 1]]), None)]
#     problem1 = Problem(train_examples, test_examples, "test_id")
#     assert get_universal_solution(problem1) == [{"name": "move_all_shapes", "args": (1, 0)}]