# mandelbrot_cyclic.py
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
    def convergence(self, c: complex, smooth=False, clamp=True) -> float:
        value = self.count_iterations(c, smooth)/self.max_iterations
        return max(0.0, min(value, 1.0)) if clamp else value
    def count_iterations(self, c: complex,  smooth=False) -> int | float:
        if c.real*c.real+c.imag*c.imag < 0.0625: return self.max_iterations
        if (c.real+1)*(c.real+1)+c.imag*c.imag < 0.0625: return self.max_iterations
        if (c.real > -0.75) and (c.real < 0.5):
            ct = c.real-0.25 + 1.j * c.imag
            ctnrm2 = abs(ct)
            if ctnrm2 < 0.5*(1-ct.real/max(ctnrm2, 1.E-14)): return self.max_iterations
        z = 0
        for iter in range(self.max_iterations):
            z = z*z + c
            if abs(z) > self.escape_radius:
                if smooth: return iter + 1 - log(log(abs(z)))/log(2)
                return iter
        return self.max_iterations

# --- Configuration MPI ---
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

mandelbrot_set = MandelbrotSet(max_iterations=50, escape_radius=10)
width, height = 1024, 1024
scaleX = 3./width
scaleY = 2.25/height

# --- Stratégie Cyclique ---
# Répartition des lignes avec un pas de 'size'
my_rows = list(range(rank, height, size))
local_height = len(my_rows)

# Allocation locale
local_convergence = np.empty((local_height, width), dtype=np.double)

comm.Barrier()
deb = time()

# Calcul
for i, y_global in enumerate(my_rows):
    for x in range(width):
        c = complex(-2. + scaleX*x, -1.125 + scaleY * y_global)
        local_convergence[i, x] = mandelbrot_set.convergence(c, smooth=True)

comm.Barrier()
fin = time()

# --- Rassemblement des données ---
recv_buffer = None
if rank == 0:
    recv_buffer = np.empty((height, width), dtype=np.double)

# Gather concatène les blocs (l'ordre sera : tout P0, tout P1...)
comm.Gather(local_convergence, recv_buffer, root=0)

if rank == 0:
    print(f"Cyclique - Processus : {size} | Temps : {fin-deb:.4f} s")
    
    # Reconstruction de l'image
    final_image = np.empty((height, width), dtype=np.double)
    
    # Hypothèse : division exacte (height % size == 0)
    rows_per_proc = height // size 
    
    for p in range(size):
        # Extraction du bloc venant du processus 'p'
        chunk = recv_buffer[p*rows_per_proc : (p+1)*rows_per_proc, :]
        # Placement aux positions correctes (slicing avec pas)
        final_image[p::size, :] = chunk

    image = Image.fromarray(np.uint8(matplotlib.cm.plasma(final_image)*255))
    image.save("mandelbrot_cyclic.png")