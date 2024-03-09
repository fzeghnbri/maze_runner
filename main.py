import argparse
from game import solve_maze
from enum import Enum


class Algorithm(Enum):
    dfs = "DFS"
    bfs = "BFS"
    ids = "IDS"
    a_star = "A_Star"
    a_star_geometric = "ASG"

    def __str__(self):
        return self.value


def main():
    parser = argparse.ArgumentParser(
        prog="Maze Solver",
        description="Utilizing artificial intelligence algorithms to solve mazes",
    )
    parser.add_argument(
        "-a",
        "--algorithm",
        type=Algorithm,
        choices=list(Algorithm),
        help="Algorithm must be: [DFS, BFS, IDS, A_Star, ASG]",
    )
    parser.add_argument(
        "-m", "--map", default=0, type=int, help="0, and 1 are the available maps"
    )
    parser.add_argument(
        "-s", "--start", default=None, type=str, help="Start position of the agent"
    )
    parser.add_argument(
        "-g", "--goal", default=None, type=str, help="Goal position of the agent"
    )

    args = parser.parse_args()
    if args.algorithm:
        start = args.start.split(",") if args.start else None
        goal = args.goal.split(",") if args.goal else None
        start_pos = (int(start[0]), int(start[1])) if start else None
        goal_pos = (int(goal[0]), int(goal[1])) if goal else None

        solve_maze(
            map_address=f"mazes/maze_{args.map}.csv",
            algorithm=str(args.algorithm),
            start_pos=start_pos,
            goal_pos=goal_pos,
        )
    else:
        parser.print_help()
        exit()


if __name__ == "__main__":
    main()
