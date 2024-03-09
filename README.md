# Install requirements
To install requirement, run: `pip install -r requirements.txt`

# Run the code
Running the code needs 4 parameters:

1. -a, --algorithm
    - this parameter specifies the algorithm which is going to run.
    - `[dfs, bfs, ids, a_star, asg]` are your only options.
2. -m, --map
    - map parameter demonstrates the map which the algorithm solves
    - `[0, 1, 2, 3, 4, 5, 6, 7]` are the options of map
3. -s, --start
    - the start position of your agent, such as `1,2`
4. -g, --goal
    - the goal position of your agent, for instance `7,9`

## Example
 
```shell
python3 main.py -a a_star -m 3 -s '1,2' -g '6,4' 
```
 
