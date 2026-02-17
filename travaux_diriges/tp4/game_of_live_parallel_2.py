"""
Game of Life - Parallel Version (1 UI Process, N Compute Processes)
Domain Decomposition with Ghost Cells (1D Halo Exchange)
"""
import pygame as pg
import numpy as np
import time
import sys
from mpi4py import MPI

dico_patterns = {
    'blinker' : ((5,5),[(2,1),(2,2),(2,3)]),
    'toad'    : ((6,6),[(2,2),(2,3),(2,4),(3,3),(3,4),(3,5)]),
    "acorn"   : ((100,100), [(51,52),(52,54),(53,51),(53,52),(53,55),(53,56),(53,57)]),
    "beacon"  : ((6,6), [(1,3),(1,4),(2,3),(2,4),(3,1),(3,2),(4,1),(4,2)]),
    "boat"    : ((5,5),[(1,1),(1,2),(2,1),(2,3),(3,2)]),
    "glider"  : ((100,90),[(1,1),(2,2),(2,3),(3,1),(3,2)]),
    "glider_gun": ((400,400),[(51,76),(52,74),(52,76),(53,64),(53,65),(53,72),(53,73),(53,86),(53,87),(54,63),(54,67),(54,72),(54,73),(54,86),(54,87),(55,52),(55,53),(55,62),(55,68),(55,72),(55,73),(56,52),(56,53),(56,62),(56,66),(56,68),(56,69),(56,74),(56,76),(57,62),(57,68),(57,76),(58,63),(58,67),(59,64),(59,65)]),
    "space_ship": ((25,25),[(11,13),(11,14),(12,11),(12,12),(12,14),(12,15),(13,11),(13,12),(13,13),(13,14),(14,12),(14,13)]),
    "die_hard": ((100,100), [(51,57),(52,51),(52,52),(53,52),(53,56),(53,57),(53,58)]),
    "pulsar"  : ((17,17),[(2,4),(2,5),(2,6),(7,4),(7,5),(7,6),(9,4),(9,5),(9,6),(14,4),(14,5),(14,6),(2,10),(2,11),(2,12),(7,10),(7,11),(7,12),(9,10),(9,11),(9,12),(14,10),(14,11),(14,12),(4,2),(5,2),(6,2),(4,7),(5,7),(6,7),(4,9),(5,9),(6,9),(4,14),(5,14),(6,14),(10,2),(11,2),(12,2),(10,7),(11,7),(12,7),(10,9),(11,9),(12,9),(10,14),(11,14),(12,14)]),
    "floraison": ((40,40), [(19,18),(19,19),(19,20),(20,17),(20,19),(20,21),(21,18),(21,19),(21,20)]),
    "block_switch_engine" : ((400,400), [(201,202),(201,203),(202,202),(202,203),(211,203),(212,204),(212,202),(214,204),(214,201),(215,201),(215,202),(216,201)]),
    "u"       : ((200,200), [(101,101),(102,102),(103,102),(103,101),(104,103),(105,103),(105,102),(105,101),(105,105),(103,105),(102,105),(101,105),(101,104)]),
    "flat"    : ((200,400), [(80,200),(81,200),(82,200),(83,200),(84,200),(85,200),(86,200),(87,200), (89,200),(90,200),(91,200),(92,200),(93,200),(97,200),(98,200),(99,200),(106,200),(107,200),(108,200),(109,200),(110,200),(111,200),(112,200),(114,200),(115,200),(116,200),(117,200),(118,200)])
}

class LocalGrille:
    """ Handles the local chunk of the grid for a specific compute node """
    def __init__(self, local_dim, initial_cells):
        self.dimensions = local_dim
        self.cells = initial_cells

    def compute_next_iteration(self, ghost_top, ghost_bot):
        """ Computes next gen using ghost cells for top and bottom boundaries """
        ny, nx = self.dimensions
        next_cells = np.empty(self.dimensions, dtype=np.uint8)
        
        # Stack ghost cells to create a working grid of size (ny+2, nx)
        work_grid = np.vstack([ghost_top, self.cells, ghost_bot])
        
        # Offset by 1 because work_grid has a top ghost row at index 0
        for i in range(ny):
            wi = i + 1 
            for j in range(nx):
                j_left = (j - 1 + nx) % nx  # Left/Right boundaries remain periodic (modulo)
                j_right = (j + 1) % nx
                
                # Sum 8 neighbors using the padded work_grid
                s = (work_grid[wi-1, j_left] + work_grid[wi-1, j] + work_grid[wi-1, j_right] +
                     work_grid[wi, j_left]   +                      work_grid[wi, j_right] +
                     work_grid[wi+1, j_left] + work_grid[wi+1, j] + work_grid[wi+1, j_right])
                
                if self.cells[i, j] == 1:
                    next_cells[i, j] = 1 if (s == 2 or s == 3) else 0
                else:
                    next_cells[i, j] = 1 if s == 3 else 0
                    
        self.cells = next_cells

class App:
    """ Handles UI. Only used by Rank 0 """
    def __init__(self, geometry, dimensions, color_life=pg.Color("black"), color_dead=pg.Color("white")):
        self.dimensions = dimensions
        self.size_x = geometry[1] // dimensions[1]
        self.size_y = geometry[0] // dimensions[0]
        self.width = dimensions[1] * self.size_x
        self.height = dimensions[0] * self.size_y
        self.screen = pg.display.set_mode((self.width, self.height))
        self.col_life = color_life
        self.col_dead = color_dead

    def compute_rectangle(self, i: int, j: int):
        return (self.size_x * j, self.height - self.size_y * (i + 1), self.size_x, self.size_y)

    def draw(self, full_cells):
        self.screen.fill(self.col_dead)
        for i in range(self.dimensions[0]):
            for j in range(self.dimensions[1]):
                if full_cells[i, j] == 1:
                    self.screen.fill(self.col_life, self.compute_rectangle(i, j))
        pg.display.update()

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if size < 3:
        if rank == 0:
            print("Please run with at least 3 processes: mpiexec -n 3 python ...")
        return

    # Configuration 
    choice = 'glider' if len(sys.argv) <= 1 else sys.argv[1]
    resx = 800 if len(sys.argv) <= 2 else int(sys.argv[2])
    resy = 800 if len(sys.argv) <= 3 else int(sys.argv[3])
    
    dim, pattern = dico_patterns.get(choice, dico_patterns['glider'])
    global_ny, global_nx = dim
    
    num_workers = size - 1
    # Note: Assuming global_ny is perfectly divisible by num_workers for simplicity
    local_ny = global_ny // num_workers 

    if rank == 0:
        # --- UI PROCESS (Rank 0) ---
        pg.init()
        appli = App((resx, resy), dim)
        
        # Initialize full grid and distribute chunks to workers
        full_cells = np.zeros(dim, dtype=np.uint8)
        indices_i = [v[0] for v in pattern]
        indices_j = [v[1] for v in pattern]
        full_cells[indices_i, indices_j] = 1
        
        for w in range(1, size):
            start_row = (w - 1) * local_ny
            end_row = start_row + local_ny
            chunk = full_cells[start_row:end_row, :]
            comm.send(chunk, dest=w, tag=10)

        mustContinue = True
        while mustContinue:
            t1 = time.time()
            
            # Broadcast state to workers
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    mustContinue = False
            for w in range(1, size):
                comm.send(mustContinue, dest=w, tag=99)
            
            if not mustContinue: break

            # Gather computed chunks from workers
            for w in range(1, size):
                chunk = comm.recv(source=w, tag=20)
                start_row = (w - 1) * local_ny
                full_cells[start_row:start_row + local_ny, :] = chunk

            t2 = time.time()
            appli.draw(full_cells)
            t3 = time.time()
            
            print(f"Gather time: {t2-t1:2.2e} s, Draw time: {t3-t2:2.2e} s\r", end='')

        pg.quit()

    else:
        # --- COMPUTE PROCESSES (Ranks 1 to N) ---
        # 1. Receive initial local grid from Rank 0
        local_cells = comm.recv(source=0, tag=10)
        local_grid = LocalGrille((local_ny, global_nx), local_cells)
        
        # 2. Determine neighbors for torus topology (periodic boundaries)
        worker_id = rank - 1
        top_neighbor = rank - 1 if worker_id > 0 else num_workers
        bot_neighbor = rank + 1 if worker_id < num_workers - 1 else 1

        while True:
            # Check if UI closed
            mustContinue = comm.recv(source=0, tag=99)
            if not mustContinue: break

            # --- HALO EXCHANGE (Ghost Cells) ---
            ghost_top = np.empty(global_nx, dtype=np.uint8)
            ghost_bot = np.empty(global_nx, dtype=np.uint8)
            
            # Send top row up, receive bottom ghost from down
            comm.Sendrecv(local_grid.cells[0, :], dest=top_neighbor, 
                          recvbuf=ghost_bot, source=bot_neighbor)
            
            # Send bottom row down, receive top ghost from up
            comm.Sendrecv(local_grid.cells[-1, :], dest=bot_neighbor, 
                          recvbuf=ghost_top, source=top_neighbor)

            # Compute next step using the received ghost cells
            local_grid.compute_next_iteration(ghost_top, ghost_bot)
            
            # Send computed chunk back to Rank 0 for drawing
            comm.send(local_grid.cells, dest=0, tag=20)

if __name__ == '__main__':
    main()