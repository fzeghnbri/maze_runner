import csv
import random

def generate_test_case(n, m, num_obstacles):
    grid = [['1' for _ in range(m)] for _ in range(n)]

    # Add some obstacles randomly
    for _ in range(num_obstacles):
        row = random.randint(0, n - 1)
        col = random.randint(0, m - 1)
        grid[row][col] = '0'

    return grid

def save_to_csv(grid, filename='test_case.csv'):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(grid)

if __name__ == "__main__":
    # Set the size of the grid and number of obstacles
    n = 40
    m = 40
    num_obstacles = random.randint(n * m // 12,n * m // 5)

    # Generate test case
    test_case = generate_test_case(n, m, num_obstacles)

    # Save to CSV file
    save_to_csv(test_case)

    print(f'Test case saved to "test_case.csv"')
