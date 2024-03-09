from _helpers import Node, Stack, Queue, PriorityQueue

class DFS_Algorithm:
    def __init__(self, start_pos, goal_pos, grid_dim):
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.grid_dim = grid_dim
        self.stack = Stack()
        self.stack.push(Node(pos=start_pos, parent=None))

    def get_successors(self, x, y):
        return [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]

    def is_valid_cell(self, pos):
        return 0 <= pos[0] <= self.grid_dim[0] and 0 <= pos[1] <= self.grid_dim[1]

    def backtrack_solution(self, curr_node):
        return self._backtrack(curr_node)

    def _backtrack(self, curr_node):
        return [] if curr_node.parent is None else self._backtrack(curr_node.parent) + [curr_node.position()]

    def update(self, grid):
        curr_state = self.stack.pop()
        x, y = curr_state.position()
        done = False
        solution_path = []

        for step in self.get_successors(x, y):
            if self.is_valid_cell(step) and grid[step[0], step[1]] in [1, 3]:  # 1: empty cell has not been explored yet, 3: goal cell
                self.stack.push(Node(pos=step, parent=curr_state))

                if step == self.goal_pos:
                    done = True
                    solution_path = self.backtrack_solution(curr_state)
                    break

                grid[x, y] = 4  # visited

        return solution_path, done, grid

class BFS_Algorithm:
    def __init__(self, start_pos, goal_pos, grid_dim):
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.grid_dim = grid_dim
        self.queue = Queue()
        self.queue.push(Node(pos=start_pos, parent=None))

    def get_successors(self, node):
        x, y = node
        return [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]

    def is_valid_cell(self, pos):
        return 0 <= pos[0] <= self.grid_dim[0] and 0 <= pos[1] <= self.grid_dim[1]

    def bfs(self, grid):
        while not self.queue.empty():
            current = self.queue.get()
            x, y = current.position()

            if (x, y) == self.goal_pos:
                path = self.backtrack_solution(current)
                return path, True, grid

            for neighbor in self.get_successors((x, y)):
                if self.is_valid_cell(neighbor) and grid[neighbor[0], neighbor[1]] != 2:  # Check for obstacles
                    self.queue.push(Node(pos=neighbor, parent=current))
                    grid[neighbor[0], neighbor[1]] = 4  # Mark as visited

        return [], False, grid

    def backtrack_solution(self, current):
        path = [current.position()]
        while current.parent is not None:
            current = current.parent
            path.append(current.position())
        return path[::-1]

    def update(self, grid):
        path, found, grid = self.bfs(grid)
        return path, found, grid

class IDS_Algorithm:
    def __init__(self, start_pos, goal_pos, grid_dim):
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.grid_dim = grid_dim
        self.max_depth = 10

    def dfs(self, node, depth, grid):
        x, y = node
        if depth == 0:
            return False, []
        
        if node == self.goal_pos:
            return True, [node]

        for neighbor in self.get_successors(node):
            if not self.is_valid_cell(neighbor) or grid[neighbor[0], neighbor[1]] == 2:  # Check for obstacles
                continue
            
            grid[neighbor[0], neighbor[1]] = 4  # Mark as visited
            
            found, path = self.dfs(neighbor, depth - 1, grid)
            if found:
                return True, [node] + path
        
        return False, []

    def get_successors(self, node):
        x, y = node
        return [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]

    def is_valid_cell(self, pos):
        return 0 <= pos[0] <= self.grid_dim[0] and 0 <= pos[1] <= self.grid_dim[1]

    def update(self, grid):
        for depth in range(self.max_depth):
            found, path = self.dfs(self.start_pos, depth, grid)
            if found:
                return path, True, grid

        return [], False, grid

class A_Star_Algorithm:
    def __init__(self, start_pos, goal_pos, grid_dim):
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.grid_dim = grid_dim
        self.open_set = PriorityQueue()
        self.open_set.push((0, start_pos))
        self.came_from = {}
        self.g_score = {start_pos: 0}
        self.f_score = {start_pos: heuristic(start_pos, goal_pos)}

    def get_successors(self, node):
        x, y = node
        return [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]

    def is_valid_cell(self, pos):
        return 0 <= pos[0] <= self.grid_dim[0] and 0 <= pos[1] <= self.grid_dim[1]

    def reconstruct_path(self, current):
        path = [current]
        while current in self.came_from:
            current = self.came_from[current]
            path.append(current)
        return path[::-1]

    def update(self, grid):
        while not self.open_set.empty():
            current_cost, current_node = self.open_set.get()

            if current_node == self.goal_pos:
                path = self.reconstruct_path(current_node)
                return path, True, grid

            for neighbor in self.get_successors(current_node):
                if not self.is_valid_cell(neighbor) or grid[neighbor[0], neighbor[1]] == 2:  # Check for obstacles
                    continue

                tentative_g_score = self.g_score[current_node] + 1  # Assuming each step costs 1
                if neighbor not in self.g_score or tentative_g_score < self.g_score[neighbor]:
                    self.came_from[neighbor] = current_node
                    self.g_score[neighbor] = tentative_g_score
                    self.f_score[neighbor] = tentative_g_score + heuristic(neighbor, self.goal_pos)
                    self.open_set.push((self.f_score[neighbor], neighbor))

        return [], False, grid

class A_Star_Geometric_Algorithm:
    def __init__(self, start_pos, goal_pos, grid_dim):
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.grid_dim = grid_dim
        self.open_set = PriorityQueue()
        self.open_set.push((0, start_pos))  # Initialize with start position
        self.came_from = {}
        self.g_score = {start_pos: 0}  # Cost from start along the best known path
        self.f_score = {start_pos: heuristic(start_pos, goal_pos)}  # Estimated total cost from start to goal through y

    def get_successors(self, node):
        x, y = node
        return [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]

    def is_valid_cell(self, pos):
        return 0 <= pos[0] <= self.grid_dim[0] and 0 <= pos[1] <= self.grid_dim[1]

    def reconstruct_path(self, current):
        path = [current]
        while current in self.came_from:
            current = self.came_from[current]
            path.append(current)
        return path[::-1]

    def update(self, grid):
        while not self.open_set.empty():
            _, current_node = self.open_set.get()  # Get the node with the lowest f_score value

            if current_node == self.goal_pos:
                path = self.reconstruct_path(current_node)
                return path, True, grid

            for neighbor in self.get_successors(current_node):
                if not self.is_valid_cell(neighbor) or grid[neighbor[0], neighbor[1]] == 2:  # Check for obstacles
                    continue

                tentative_g_score = self.g_score[current_node] + 1  # Cost for current node + cost of moving to the neighbor (assuming 1 for simplicity)
                if neighbor not in self.g_score or tentative_g_score < self.g_score[neighbor]:
                    self.came_from[neighbor] = current_node
                    self.g_score[neighbor] = tentative_g_score
                    self.f_score[neighbor] = tentative_g_score + heuristic(neighbor, self.goal_pos)
                    self.open_set.push((self.f_score[neighbor], neighbor))

        return [], False, grid

