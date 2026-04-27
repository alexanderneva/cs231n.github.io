import numpy as np

def refine_grid(g,scale,num_step):
    lb = g[0]
    ub = g[-1]
    lb *= scale
    ub *= scale
    return np.linspace(lb,ub,num_step)

def cv_cube(g1,g2,num_iter):
    return ([(x,y) for x in g1 for y in g2], num_iter)

def run_SGD(num_iter):
    pass

grid_steps = 10
num_top_values = 5
num_iter = 100

def grid_search():
    results = dict.fromkeys(cv_cube(g1,g2,num_iter)[0])
    for step in range(grid_steps):
        results = run_SGD(num_iter)