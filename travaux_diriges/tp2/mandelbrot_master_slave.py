# mandelbrot_master_slave.py
import numpy as np
from mpi4py import MPI
from dataclasses import dataclass
from PIL import Image
from math import log
from time import time
import matplotlib.cm

# --- CLASSE VECTORISÉE ---
class MandelbrotSet:
    def __init__(self, max_iterations : int, escape_radius : float = 2. ):
        self.max_iterations = max_iterations
        self.escape_radius  = escape_radius

    def convergence(self, c: np.ndarray, smooth=False, clamp=True) -> np.ndarray:
        value = self.count_iterations(c, smooth)/self.max_iterations
        return np.maximum(0.0, np.minimum(value, 1.0)) if clamp else value

    def count_iterations(self, c: np.ndarray,  smooth=False) -> np.ndarray:
        iter = self.max_iterations * np.ones(c.shape, dtype=np.double)
        mask = (np.abs(c) >= 0.25) | (np.abs(c+1.) >= 0.25)
        
        z = np.zeros(c.shape, dtype=np.complex128)
        for it in range(self.max_iterations):
            z[mask] = z[mask]*z[mask] + c[mask]
            has_diverged = np.abs(z) > self.escape_radius
            if has_diverged.size > 0:
                iter[has_diverged] = np.minimum(iter[has_diverged], it)
                mask = mask & ~has_diverged
            if np.any(mask) == False : break
        
        has_diverged = np.abs(z) > 2
        if smooth:
            iter[has_diverged] += 1 - np.log(np.log(np.abs(z[has_diverged])))/log(2)
        return iter

# --- CONFIGURATION MPI ---
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

TAG_WORK = 11
TAG_DATA = 12
TAG_STOP = 13

width, height = 1024, 1024
max_iter = 50 
mandelbrot_set = MandelbrotSet(max_iterations=max_iter, escape_radius=2.)
scaleX = 3./width
scaleY = 2.25/height

# ==========================================
# MAÎTRE (MASTER) - RANK 0
# ==========================================
if rank == 0:
    print(f"Maître-Esclave démarré avec {size} processus.")
    final_convergence = np.empty((height, width), dtype=np.double)
    
    next_row = 0
    total_rows = height
    active_workers = 0
    
    deb = time()

    # 1. Distribution initiale
    for worker_rank in range(1, size):
        if next_row < total_rows:
            comm.send(next_row, dest=worker_rank, tag=TAG_WORK)
            next_row += 1
            active_workers += 1
        else:
            comm.send(None, dest=worker_rank, tag=TAG_STOP)

    # 2. Boucle dynamique
    while active_workers > 0:
        status = MPI.Status()
        # Réception (index, données)
        data = comm.recv(source=MPI.ANY_SOURCE, tag=TAG_DATA, status=status)
        source_rank = status.Get_source()
        
        row_idx, row_data = data
        final_convergence[row_idx, :] = row_data

        if next_row < total_rows:
            comm.send(next_row, dest=source_rank, tag=TAG_WORK)
            next_row += 1
        else:
            comm.send(None, dest=source_rank, tag=TAG_STOP)
            active_workers -= 1

    fin = time()
    print(f"Maître-Esclave Temps Total : {fin-deb:.4f} s")
    
    image = Image.fromarray(np.uint8(matplotlib.cm.plasma(final_convergence.T)*255))
    image.save("mandelbrot_master_slave.png")

# ==========================================
# ESCLAVE (SLAVE) - RANK > 0
# ==========================================
else:
    while True:
        status = MPI.Status()
        y = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()

        if tag == TAG_STOP:
            break

        if tag == TAG_WORK:
            # Calcul vectorisé de la ligne complète
            real_vals = np.linspace(-2.0, -2.0 + scaleX * width, width, endpoint=False)
            imag_val = -1.125 + scaleY * y
            c_row = real_vals + 1j * imag_val
            
            result_row = mandelbrot_set.convergence(c_row, smooth=True)
            
            comm.send((y, result_row), dest=0, tag=TAG_DATA)