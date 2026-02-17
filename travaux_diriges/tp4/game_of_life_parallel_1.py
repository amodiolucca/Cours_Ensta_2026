# jeu_de_la_vie_v1.py
from mpi4py import MPI
import pygame as pg
import numpy as np
import time
import sys

# --- Copie des dictionnaires et configuration (identique au code original) ---
dico_patterns = {
    'glider': ((100,90),[(1,1),(2,2),(2,3),(3,1),(3,2)]),
    'pulsar': ((17,17),[(2,4),(2,5),(2,6),(7,4),(7,5),(7,6),(9,4),(9,5),(9,6),(14,4),(14,5),(14,6),(2,10),(2,11),(2,12),(7,10),(7,11),(7,12),(9,10),(9,11),(9,12),(14,10),(14,11),(14,12),(4,2),(5,2),(6,2),(4,7),(5,7),(6,7),(4,9),(5,9),(6,9),(4,14),(5,14),(6,14),(10,2),(11,2),(12,2),(10,7),(11,7),(12,7),(10,9),(11,9),(12,9),(10,14),(11,14),(12,14)]),
    # ... ajoutez les autres patterns ici si nécessaire
}

class Grille:
    # ... (Même code que votre classe Grille originale) ...
    def __init__(self, dim, init_pattern=None):
        self.dimensions = dim
        if init_pattern is not None:
            self.cells = np.zeros(self.dimensions, dtype=np.uint8)
            indices_i = [v[0] for v in init_pattern]
            indices_j = [v[1] for v in init_pattern]
            self.cells[indices_i,indices_j] = 1
        else:
            self.cells = np.random.randint(2, size=dim, dtype=np.uint8)

    def compute_next_iteration(self):
        # ... (Logique originale inchangée pour V1) ...
        ny, nx = self.dimensions
        next_cells = np.empty(self.dimensions, dtype=np.uint8)
        
        # Simple loop implementation (as provided)
        for i in range(ny):
            i_above = (i+ny-1)%ny
            i_below = (i+1)%ny
            for j in range(nx):
                j_left = (j-1+nx)%nx
                j_right= (j+1)%nx
                # Sum neighbors
                s = (self.cells[i_above, j_left] + self.cells[i_above, j] + self.cells[i_above, j_right] +
                     self.cells[i, j_left] + self.cells[i, j_right] +
                     self.cells[i_below, j_left] + self.cells[i_below, j] + self.cells[i_below, j_right])
                
                if self.cells[i,j] == 1:
                    next_cells[i,j] = 1 if s == 2 or s == 3 else 0
                else:
                    next_cells[i,j] = 1 if s == 3 else 0
        self.cells = next_cells
        return self.cells # Return full grid for MPI transport

class App:
    # ... (Même logique d'affichage que votre code original) ...
    def __init__(self, geometry, dim, color_life=pg.Color("black"), color_dead=pg.Color("white")):
        self.dim = dim
        self.col_life = color_life
        self.col_dead = color_dead
        self.size_x = geometry[1]//dim[1]
        self.size_y = geometry[0]//dim[0]
        self.width = dim[1] * self.size_x
        self.height= dim[0] * self.size_y
        self.screen = pg.display.set_mode((self.width,self.height))

    def draw(self, cells):
        # Fill background
        self.screen.fill(self.col_dead)
        # Draw live cells
        rows, cols = cells.shape
        for i in range(rows):
            for j in range(cols):
                if cells[i,j] == 1:
                    rect = (self.size_x*j, self.height - self.size_y*(i + 1), self.size_x, self.size_y)
                    self.screen.fill(self.col_life, rect)
        pg.display.update()

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # Configuration
    choice = 'glider'
    dim, pattern = dico_patterns[choice]
    resx, resy = 800, 800

    if rank == 0:
        # --- UI Process ---
        pg.init()
        app = App((resx, resy), dim)
        running = True
        
        while running:
            # 1. Check pygame events
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
            
            # 2. Tell compute node status
            comm.send(running, dest=1, tag=1)
            if not running: break

            # 3. Receive grid from compute node
            grid_cells = comm.recv(source=1, tag=2)
            
            # 4. Draw
            app.draw(grid_cells)
            
        pg.quit()

    elif rank == 1:
        # --- Compute Process ---
        grid = Grille(dim, pattern)
        
        while True:
            # 1. Check if we should continue
            should_run = comm.recv(source=0, tag=1)
            if not should_run: break
            
            # 2. Compute
            grid.compute_next_iteration()
            
            # 3. Send data to UI
            comm.send(grid.cells, dest=0, tag=2)

if __name__ == '__main__':
    main()