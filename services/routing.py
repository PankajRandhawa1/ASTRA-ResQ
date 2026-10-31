"""Experimental image-grid A* route demo. Not geographic routing."""
import heapq
import math
import numpy as np

def astar_grid(mask: np.ndarray, start: tuple[int, int], goal: tuple[int, int]):
    """Find an 8-neighbour path through unmasked pixels; returns grid cells and grid cost."""
    h, w = mask.shape
    sr, sc = start; gr, gc = goal
    if not (0 <= sr < h and 0 <= sc < w and 0 <= gr < h and 0 <= gc < w): return None
    blocked = mask.astype(bool).copy(); blocked[sr, sc] = False; blocked[gr, gc] = False
    moves = [(-1,0,1.0),(1,0,1.0),(0,-1,1.0),(0,1,1.0),(-1,-1,math.sqrt(2)),(-1,1,math.sqrt(2)),(1,-1,math.sqrt(2)),(1,1,math.sqrt(2))]
    def heuristic(r,c):
        dx,dy=abs(r-gr),abs(c-gc); return max(dx,dy)+(math.sqrt(2)-1)*min(dx,dy)
    queue=[(heuristic(sr,sc),0.0,sr,sc)]; came_from={}; costs={(sr,sc):0.0}; closed=set()
    while queue:
        _,cost,r,c=heapq.heappop(queue); node=(r,c)
        if node in closed: continue
        if node==(gr,gc):
            path=[node]
            while node in came_from: node=came_from[node]; path.append(node)
            path.reverse(); return path,cost
        closed.add(node)
        for dr,dc,step in moves:
            nr,nc=r+dr,c+dc
            if not (0<=nr<h and 0<=nc<w) or blocked[nr,nc]: continue
            if dr and dc and (blocked[r+dr,c] or blocked[r,c+dc]): continue
            new=cost+step
            if new<costs.get((nr,nc),float('inf')):
                costs[(nr,nc)]=new; came_from[(nr,nc)]=node; heapq.heappush(queue,(new+heuristic(nr,nc),new,nr,nc))
    return None
