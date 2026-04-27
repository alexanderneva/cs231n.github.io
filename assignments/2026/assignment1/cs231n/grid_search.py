import numpy as np

def refine_grid(g,scale,num_step):
    lb = g[0]
    ub = g[-1]
    lb *= scale
    ub *= scale
    return np.linspace(lb,ub,num_step)

def cv_cube(g1,g2,num_iter):
    return ([(x,y) for x in g1 for y in g2], num_iter)