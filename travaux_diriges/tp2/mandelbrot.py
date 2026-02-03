# mandelbrot_mpi.py
import numpy as np
from dataclasses import dataclass
from PIL import Image
from math import log
from time import time
import matplotlib.cm
from mpi4py import MPI

@dataclass
class MandelbrotSet:
    max_iterations: int
    escape_radius:  float = 2.0

    def __contains__(self, c: complex) -> bool:
        return self.stability(c) == 1

    def convergence(self, c: complex, smooth=False, clamp=True) -> float:
        value = self.count_iterations(c, smooth)/self.max_iterations
        return max(0.0, min(value, 1.0)) if clamp else value

    def count_iterations(self, c: complex,  smooth=False) -> int | float:
        z:    complex
        iter: int

        # Optimisations géométriques (Cardioïde/Bulbe)
        if c.real*c.real+c.imag*c.imag < 0.0625:
            return self.max_iterations
        if (c.real+1)*(c.real+1)+c.imag*c.imag < 0.0625:
            return self.max_iterations
        if (c.real > -0.75) and (c.real < 0.5):
            ct = c.real-0.25 + 1.j * c.imag
            ctnrm2 = abs(ct)
            if ctnrm2 < 0.5*(1-ct.real/max(ctnrm2, 1.E-14)):
                return self.max_iterations
        
        # Itération principale
        z = 0
        for iter in range(self.max_iterations):
            z = z*z + c
            if abs(z) > self.escape_radius:
                if smooth:
                    return iter + 1 - log(log(abs(z)))/log(2)
                return iter
        return self.max_iterations

# --- Configuration MPI ---
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Paramètres
mandelbrot_set = MandelbrotSet(max_iterations=50, escape_radius=10)
width, height = 1024, 1024

scaleX = 3./width
scaleY = 2.25/height

# --- 1. Partition par blocs (Block Partitioning) ---
# Division de la hauteur (lignes) par le nombre de processus
rows_per_process = height // size

# Définition des bornes pour ce processus
start_y = rank * rows_per_process
end_y = (rank + 1) * rows_per_process

# Allocation locale
local_convergence = np.empty((rows_per_process, width), dtype=np.double)

# Début du chrono
comm.Barrier()
deb = time()

# --- Calcul Local ---
for y_local in range(rows_per_process):
    y_global = start_y + y_local
    for x in range(width):
        c = complex(-2. + scaleX*x, -1.125 + scaleY * y_global)
        local_convergence[y_local, x] = mandelbrot_set.convergence(c, smooth=True)

# Fin du chrono
comm.Barrier()
fin = time()

# --- Rassemblement des données (Gather) ---
final_convergence = None
if rank == 0:
    final_convergence = np.empty((height, width), dtype=np.double)

# MPI Gather : Concaténation des blocs locaux sur le rang 0
comm.Gather(local_convergence, final_convergence, root=0)

# --- Sortie et sauvegarde (Maître uniquement) ---
if rank == 0:
    print(f"Nombre de processus : {size}")
    print(f"Temps du calcul de l'ensemble de Mandelbrot : {fin-deb}")
    
    # Constitution de l'image
    deb_img = time()
    image = Image.fromarray(np.uint8(matplotlib.cm.plasma(final_convergence)*255))
    fin_img = time()
    print(f"Temps de constitution de l'image : {fin_img-deb_img}")
    
    image.save("mandelbrot_mpi.png")