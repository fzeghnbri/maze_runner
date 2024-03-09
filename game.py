import numpy as np
import pygame
import time
import csv
from Algorithm import (
    A_Star_Geometric_Algorithm,
    DFS_Algorithm,
    BFS_Algorithm,
    IDS_Algorithm,
    A_Star_Algorithm,
)
from time import sleep


def solve_maze(map_address, algorithm, start_pos=None, goal_pos=None):
    grid = np.genfromtxt(map_address, delimiter=",", dtype=int)
    num_rows, num_columns = grid.shape[0], grid.shape[1]
    empty_block_count = np.count_nonzero(grid == 1)

    # Define start & goal positions
    start_pos = (0, 0) if start_pos is None else start_pos
    goal_pos = (num_rows - 1, num_columns - 1) if goal_pos is None else goal_pos
    try:
        grid[*start_pos] = 2
        grid[*goal_pos] = 3
    except IndexError:
        print("Error: Start or goal position is out of the maze!")
        return None
    grid_dim = (num_rows - 1, num_columns - 1)

    black, white, green, red, grey, blue, magenta = (
        (0, 0, 0),
        (255, 255, 255),
        (50, 205, 50),
        (255, 99, 71),
        (211, 211, 211),
        (0, 94, 255),
        (255, 0, 255),
    )
    idx_to_color = [black, white, green, red, blue, magenta]

    window_size = [min(800, num_columns * 15), min(800, num_rows * 15)]
    margin = 1
    height = (window_size[1] - margin * (num_rows - 1)) / num_rows
    width = (window_size[0] - margin * (num_columns - 1)) / num_columns

    pygame.init()

    screen = pygame.display.set_mode(window_size)

    pygame.display.set_caption(f"{algorithm} Pathfinder. Solving: {map_address}")

    done = False
    run = False
    close = False

    clock = pygame.time.Clock()

    digital_twin = None

    if algorithm == "BFS":
        digital_twin = BFS_Algorithm(
            start_pos=start_pos, goal_pos=goal_pos, grid_dim=grid_dim
        )
    elif algorithm == "DFS":
        digital_twin = DFS_Algorithm(
            start_pos=start_pos, goal_pos=goal_pos, grid_dim=grid_dim
        )
    elif algorithm == "IDS":
        digital_twin = IDS_Algorithm(
            start_pos=start_pos, goal_pos=goal_pos, grid_dim=grid_dim
        )
    elif algorithm == "A_Star":
        digital_twin = A_Star_Algorithm(
            start_pos=start_pos, goal_pos=goal_pos, grid_dim=grid_dim
        )
    elif algorithm == "ASG":
        digital_twin = A_Star_Geometric_Algorithm(
            start_pos=start_pos, goal_pos=goal_pos, grid_dim=grid_dim
        )
    else:
        return None

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

            elif event.type == pygame.KEYDOWN:
                run = True
                start_t0 = time.time()

        screen.fill(grey)

        for row in range(num_rows):
            for column in range(num_columns):
                color = idx_to_color[grid[row, column]]
                pygame.draw.rect(
                    screen,
                    color,
                    [
                        (margin + width) * column + margin,
                        (margin + height) * row + margin,
                        width,
                        height,
                    ],
                )

        clock.tick(60)
        pygame.display.flip()

        if run == True:
            solution, done, grid = digital_twin.update(grid=grid)

        if done == True:
            print(f"Total empty block numbers: {empty_block_count}")
            print(f"Explored block numbers: {np.count_nonzero(grid == 4)}")
            for pos in solution:
                grid[pos[0], pos[1]] = 5

            screen.fill(grey)

            for row in range(num_rows):
                for column in range(num_columns):
                    color = idx_to_color[grid[row, column]]
                    pygame.draw.rect(
                        screen,
                        color,
                        [
                            (margin + width) * column + margin,
                            (margin + height) * row + margin,
                            width,
                            height,
                        ],
                    )

            clock.tick(60)
            pygame.display.flip()

    print(f"Your maze solved with {algorithm} algorithm.")
    print(f"--- finished {time.time()-start_t0:.3f} s---")
    while not close:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                close = True

            elif event.type == pygame.KEYDOWN:
                close = True
    pygame.quit()
