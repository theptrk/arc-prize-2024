from typing import List, Dict, Tuple


class Shape:
    def __init__(self, shape: List[Tuple[int, int]]):
        self.positions = shape


class ColoredShape:
    def __init__(self, shape: List[Tuple[int, int]]):
        self.shape = shape
        self.positions = []
        self.color_count = 0
        self.color_set = set()
        for x, y, color in shape:
            self.positions.append((x, y))
            if color not in self.color_set:
                self.color_count += 1
                self.color_set.add(color)

    def convert_to_grid(self) -> 'Grid':
        pass

    def get_positions(self) -> List[Tuple[int, int]]:
        return self.positions

    def is_multi_color(self) -> bool:
        return self.color_count > 1

    def get_colors(self) -> List[int]:
        return list(self.color_set)

    def move_xy(self, dx: int, dy: int) -> 'Shape':
        return Shape([(x + dx, y + dy) for x, y in self.positions])


class Grid:
    def __init__(self, grid: List[List[int]]):
        self.grid = grid
        self.n_rows = len(grid)
        self.n_cols = len(grid[0])
        self.shapes_any_color_4d = self.find_shapes_any_color(n_directions=4)
        self.shapes_any_color_8d = self.find_shapes_any_color(n_directions=8)

    def __repr__(self):
        return f"Grid({self.n_rows}x{self.n_cols})"

    def display(self):
        print('\n'.join(' '.join(str(cell) for cell in row) for row in self.grid))

    def copy_raw(self):
        result = []
        for row in self.grid:
            result.append(row.copy())
        return result

    def xfind_shapes_all_colors(self) -> List[Shape]:
        def is_valid(x, y):
            return 0 <= x < self.n_rows and 0 <= y < self.n_cols and self.grid[x][y] != 0 and not visited[x][y]

        def dfs(x, y, shape):
            stack = [(x, y)]
            while stack:
                cx, cy = stack.pop()
                color = self.grid[cx][cy]
                info = (cx, cy, color)
                shape.append(info)
                for dx, dy in directions:
                    nx, ny = cx + dx, cy + dy
                    if is_valid(nx, ny):
                        visited[nx][ny] = True
                        stack.append((nx, ny))

        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        visited = [[False] * self.n_cols for _ in range(self.n_rows)]
        shapes = []

        for i in range(self.n_rows):
            for j in range(self.n_cols):
                if self.grid[i][j] != 0 and not visited[i][j]:
                    visited[i][j] = True
                    shape = []
                    dfs(i, j, shape)
                    shapes.append(Shape(sorted(shape)))
        return sorted(shapes)

    def find_shapes_any_color(self, n_directions=8) -> List[List[Tuple[int, int]]]:
        def is_valid(x, y):
            return 0 <= x < self.n_rows and 0 <= y < self.n_cols and self.grid[x][y] != 0 and not visited[x][y]

        def dfs(x, y, shape):
            stack = [(x, y)]
            while stack:
                cx, cy = stack.pop()
                shape.append((cx, cy))
                for dx, dy in directions:
                    nx, ny = cx + dx, cy + dy
                    if is_valid(nx, ny):
                        visited[nx][ny] = True
                        stack.append((nx, ny))


        if n_directions == 4:
            directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        elif n_directions == 8:
            directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        visited = [[False] * self.n_cols for _ in range(self.n_rows)]
        shapes = []

        for i in range(self.n_rows):
            for j in range(self.n_cols):
                if self.grid[i][j] != 0 and not visited[i][j]:
                    visited[i][j] = True
                    shape = []
                    dfs(i, j, shape)
                    shapes.append(sorted(shape))
        return sorted(shapes)

    def find_all_possible_movements(self, shape: Shape) -> List[Tuple[int, int]]:
        print("fn find_all_possible_movements")
        if isinstance(shape, list):
            shape = Shape(shape)

        # Calculate the dimensions of the shape
        min_row = min(x for x, _ in shape.positions)
        max_row = max(x for x, _ in shape.positions)
        min_col = min(y for _, y in shape.positions)
        max_col = max(y for _, y in shape.positions)

        # print(f"shape: {shape.positions}")
        # print(f"min_row: {min_row}, max_row: {max_row}, min_col: {min_col}, max_col: {max_col}")

        # grid_with_shape = paint_1d_to_2d_grid(shape, color=5, grid=original_grid)
        # printrows(grid_with_shape)

        # grid_with_shape = paint_1d_to_2d_grid(shape, color=5, grid=original_grid)
        # printrows(grid_with_shape)
        # Note the y is the y axis and movement denotes width
        # Note the x is the x axis and movement denotes height
        shape_width = max_col - min_col + 1
        shape_height = max_row - min_row + 1
        # print(f"shape width: {shape_width}, shape height: {shape_height}")

        # Calculate the dimensions of the original grid
        original_grid = self.grid
        grid_height = len(original_grid)
        grid_width = len(original_grid[0])

        # Determine the range of possible movements, including negative directions
        possible_movements = []
        # negative limit is relative to where the last s point can move left without falling off the grid
        # example
        # the formula is

        # Note x axis movement depends on the rightmost column position
        # 0 0 5 5   <- max_col=index 3 so it can move 3 to the left along x axis
        # 5 5 5 0   <- max_col=index 2 so it can move 2 to the left along x axis (-2 movement)
        # formula is -(max_col)
        col_negative_move_limit = -(max_col)
        # 0 0 5 5   <- min_col=index 2, grid_width=4 so it can move 1 right without falling off
        # 5 5 5 0   <- min_col=index 0, grid_width=4 so it can move 4 right without falling off (4 movement)
        # formula (grid_width-1) - min_col
        col_positive_move_limit = (grid_width-1) - min_col
        # print(f"col_negative_move_limit: {col_negative_move_limit}, col_positive_move_limit: {col_positive_move_limit}")

        row_negative_move_limit = -(shape_height)
        # Note y axis movement depends on the bottommost row position
        # 0
        # 5 <- row index 1 min_row=1 max_row=1
        # 0
        # 0
        # grid_height = 4
        # y_axis_negative = max_row = 1 and can move up -(max_row) which is -1
        # y_axis_positive = (grid_height - 1) - min_row which is (4 - 1)=3-1 = 2
        row_negative_move_limit = -(max_row)
        row_positive_move_limit = (grid_height - 1) - min_row
        # print(f"row_negative_move_limit: {row_negative_move_limit}, row_positive_move_limit: {row_positive_move_limit}")
        for d_row in range(row_negative_move_limit, row_positive_move_limit+1):  # Adjusted to allow upward movement
            for d_col in range(col_negative_move_limit, col_positive_move_limit+1):

                # we need to check if the shape is still showing on the grid
                # because an oblong shape can move to the extremes of left, top and no longer be visible
                # how to check: apply the movement to the shape and check if any of the shape's positions are out of bounds or on a 0
                # print(f"checking visibiilty for {d_row}, {d_col}")
                if self.is_any_part_visible(shape, d_row, d_col):
                    possible_movements.append([d_row, d_col])

        return possible_movements

    def is_any_part_visible(self, shape: Shape, d_row: int, d_col: int) -> bool:
        for x, y in shape.positions:
            new_x, new_y = x + d_row, y + d_col
            if 0 <= new_x < self.n_rows and 0 <= new_y < self.n_cols:
                return True
        return False

    # this should be a "Transform"
    def move_shape(self, raw_shape: List[List[int]], d_row: int, d_col: int) -> List[List[int]]:
        grid_copy = self.copy_raw()
        position_to_color = {}
        for x, y in raw_shape:
            position_to_color[(x, y)] = grid_copy[x][y]
            grid_copy[x][y] = 0

        for x, y in raw_shape:
            new_x, new_y = x + d_row, y + d_col
            if 0 <= new_x < self.n_rows and 0 <= new_y < self.n_cols:
                grid_copy[new_x][new_y] = position_to_color[(x, y)]
        return Grid(grid_copy)

    def __eq__(self, other):
        return self.grid == other.grid


class Example:
    def __init__(self, input_grid: Grid, output_grid: Grid = None):
        self.input = input_grid
        self.output = output_grid

    def check_same_dimensions(self) -> bool:
        pass

    def find_shapes(self) -> List[Shape]:
        pass

    def isolate_shapes_in_grid(self) -> List[Grid]:
        pass

    def check_input_in_output(self) -> bool:
        pass

    def check_solution(self) -> bool:
        pass

    def __repr__(self):
        return f"Example {self.input}"
        # return f"Example(input={self.input}, output={self.output})"

    def display(self):
        input_rows = self.input.grid
        input_rows = self.input.grid
        output_rows = self.output.grid if self.output else None

        # Calculate the maximum width needed for input and output rows
        max_input_width = max(len(' '.join(str(cell) for cell in row)) for row in input_rows)
        max_output_width = max(len(' '.join(str(cell) for cell in row)) for row in output_rows) if output_rows else 0

        print("input:".ljust(max_input_width + 2), "output:" if self.output else "no output:")
        for i in range(len(input_rows)):
            input_row = ' '.join(str(cell) for cell in input_rows[i])
            output_row = ' '.join(str(cell) for cell in output_rows[i]) if output_rows else ''
            print(input_row.ljust(max_input_width + 2), output_row)


class Problem:
    def __init__(self, train: List[Example], test: List[Example], id: str):
        self.test = test
        self.train = train
        self.id = id

    def __repr__(self):
        return f"Problem {self.id} (test len=({len(self.test)}) train len={len(self.train)})"

    def display(self):
        print("Train examples")
        for example in self.train:
            example.display()

        print("Test examples")
        for example in self.test:
            example.display()

    def solve(self) -> List[Grid]:
        """
        Solve the problem
        """
        pass


class ProblemSet:
    def __init__(self, raw_data):
        self.data = self._process_raw_data(raw_data)
        self.raw_data = raw_data

    def _process_raw_data(self, raw_data):
        data = {}
        for problem_id, raw_problem_data in raw_data.items():
            raw_train_data = raw_problem_data['train']
            raw_test_data = raw_problem_data['test']
            train_examples = [Example(Grid(raw_train_data[i]['input']), Grid(raw_train_data[i]['output'])) for i in range(len(raw_train_data))]
            test_examples = [Example(Grid(raw_test_data[i]['input']), output_grid=None) for i in range(len(raw_test_data))]
            data[problem_id] = Problem(train_examples, test_examples, problem_id)
        return data

    def get(self, problem_id):
        return self.data[problem_id]


    def __repr__(self):
        return f"ProblemSet({len(self.data)} problems)"


class Transform:
    def __init__(self):
        self.moves = [
            {
                "name": "move_shape",
                "args": (1, 0)
            }
        ]

    def apply(self, input: Grid) -> Grid:
        result = input.copy_raw()
        for move in self.moves:
            if move["name"] == "move":
                result = self.move(result, move["args"])
        return Grid(result)

    def move(self, input: List[List[int]], args: Tuple[int, int]) -> List[List[int]]:
        pass


"""
true function
- blue moves (1,0)
- red moves (1,1)

problem:
    test: [
        "input"
    ]
    train: [
        "input"
        "output"
    ]

read train
learn that blue moves (1,0)
learn that red moves (1,1)

how do learn? try a bunch until it checks out
exhasutive search.
1. given single shape, find movements to produce output
2. given all colors, shapes by color, find movements to produce output

create transformations = [
    move_shape_by_color ("blue", (1,0))
    move_shape_by_color ("red", (1,1))
]

check solution (input, transformations, output)

predict(problem.test.input, transformations)

PARAMETERS:
-
-
-
"""


def predict_list(input_grids: List[Grid], transforms: List[Dict]) -> List[Grid]:
    return [predict(input_grid, transforms) for input_grid in input_grids]

def predict(input_grid: Grid, raw_transforms: List[Dict]) -> Grid:
    print(f"predicting {raw_transforms}")
    result_grid = Grid(input_grid.copy_raw())

    for raw_transform in raw_transforms:
        name = raw_transform["name"]
        args = raw_transform["args"]
        match name:
            case "move_all_shapes":
                for shape in result_grid.shapes_any_color_8d:
                    row_delta = args[0]
                    col_delta = args[1]
                    result_grid = result_grid.move_shape(shape, row_delta, col_delta)
            case "move_shape":
                for shape in result_grid.shapes_any_color_8d:
                    color = args[0]
                    if shape.color_set == set([color]):
                        row_delta = args[1]
                        col_delta = args[2]
                        result_grid = result_grid.move_shape(shape, row_delta, col_delta)
            case "move_shape_type":
                # TODO: Implement moving a specific shape type
                pass
            case _:
                print(f"Unknown transform: {name}")

    return result_grid

def apply_transforms(input_grid: Grid, transforms: List[Dict]) -> Grid:
    pass

def apply_transform(input_grid: Grid, transform: Dict) -> Grid:
    pass

def check_input_output(input_grid: Grid, raw_transforms, output_grid: Grid) -> bool:
    result_grid = predict(input_grid, raw_transforms)
    return result_grid == output_grid

def check_all_input_output(input_grids: List[Grid], output_grids: List[Grid], raw_transforms: List[Dict]):
    all_valid = True
    for input_grid, output_grid in zip(input_grids, output_grids):
        if not check_input_output(input_grid, raw_transforms, output_grid):
            all_valid = False
    return all_valid

# here we get shapes, check them against the mid point of all other shapes
def xcheck_improved(input_grid: Grid, output_grid: Grid, raw_transforms: List[Dict]) -> bool:
    # mid point distance between input and output shapes
    input_shapes = input_grid.find_shapes_any_color()
    output_shapes = output_grid.find_shapes_any_color()
    # get average distance between shapes

    # perform the transformation

    # get mid point distance between input and output shapes
    # get the average distance between shapes

    # improved if the new distance is less than the old distance
    # return improved


def exhaustive_search(input_grid: Grid, output_grid: Grid) -> List[List[Dict]]:
    print("fn exhaustive_search")
    strategies = []
    strategies_set = set()
    # does moving all shapes down by 1 row work?
    # strategies.append([{"name": "move_all_shapes", "args": (1, 0)}])
    # does moving all shapes down by 1 row and right by 1 column work?
    # strategies.append([{"name": "move_all_shapes", "args": (1, 1)}])

    # check all movements

    # FIXME: note this is checking that the movement of some shape
    # becomes the movement of ALL shapes because it uses "move_all_shapes"
    # To add nuance:
    # - this should be the movement of this shape color, type, etc.
    for shape in input_grid.find_shapes_any_color():
        possible_movements = input_grid.find_all_possible_movements(shape)
        for movement in possible_movements:
            # strategies.append([{"name": "move_all_shapes", "args": movement}])
            # first check if this is a "perfect" move
            # print(f"checking movement: {movement}")
            import json
            single_move = [{"name": "move_all_shapes", "args": movement}]
            json_string = json.dumps(single_move)
            if check_input_output(input_grid, single_move, output_grid):
                strategies_set.add(json_string)
                print(f"⭐️ found perfect move: {movement}")
                strategies.append(single_move)

    print(list(strategies_set))
    deduplicated_strategies = [json.loads(strat) for strat in list(strategies_set)]
    print("deduplicated_strategies", deduplicated_strategies)
    return deduplicated_strategies
    # print(f"strategies: {strategies}")
    # return strategies

def create_solution_candidates(input_grid: Grid, output_grid: Grid):
    # solution_candidates = []

    # for candidate in exhaustive_search(input_grid, output_grid):
    #     if check_input_output(input_grid, candidate, output_grid):
    #         solution_candidates.append(candidate)
    # return solution_candidates
    return exhaustive_search(input_grid, output_grid)

def resolve_solution_candidates(problem_to_solution_candidates: Dict[int, List[List[Dict]]]):
    pass

def create_problem_space_solution_candidates(input_grids: List[Grid], output_grids: List[Grid]):
    pass

def get_universal_solution(problem: Problem):
    print("fn get_universal_solution")
    problem_to_solution_candidates = {}
    for i, example in enumerate(problem.train):
        print(f"problem.train[{i}]")
        print(example)
        solution_candidates = create_solution_candidates(example.input, example.output)
        print(f"solution_candidates: {solution_candidates}")
        problem_to_solution_candidates[i] = solution_candidates

    problem_to_solution_candidates['all'] = create_problem_space_solution_candidates(problem.train, problem.test)

    universal_solution = resolve_solution_candidates(problem_to_solution_candidates)

    if check_all_input_output(problem.test, universal_solution):
        return universal_solution
    else:
        return None

def solve(problem: Problem):
    universal_solution = get_universal_solution(problem)
    if universal_solution is None:
        raise Exception("No solution found")

    return predict_list(problem.test, universal_solution)

class Agent:
    def __init__(self, problem: Problem):
        self.problem = problem

    def fit(self) -> List[Transform]:
        pass

    def predict(self, transforms: List[Transform], grid: Grid) -> Grid:
        pass