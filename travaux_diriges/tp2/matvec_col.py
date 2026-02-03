# matvec_col.py
from mpi4py import MPI
import numpy as np
from time import time

# Configuration MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Dimension du problème
dim = 4800

# 1. Calcul de N_loc (Colonnes par processus)
if dim % size != 0:
    if rank == 0: print("Erreur : La dimension doit être divisible par le nombre de processus.")
    comm.Abort()

N_loc = dim // size

# Intervalle de colonnes pour ce processus
start_col = rank * N_loc
end_col   = (rank + 1) * N_loc

# 2. Initialisation locale de la matrice A (Seulement N_loc colonnes)
# A_local est de taille (dim, N_loc)
A_local = np.empty((dim, N_loc), dtype=np.float64)
for j_local in range(N_loc):
    j_global = start_col + j_local
    for i in range(dim):
        A_local[i, j_local] = (i + j_global) % dim + 1

# 3. Initialisation locale du vecteur u (Seulement la partie correspondant aux colonnes)
u_local = np.array([j_global + 1.0 for j_global in range(start_col, end_col)])

comm.Barrier()
start_time = time()

# 4. Calcul du produit local
# (dim x N_loc) * (N_loc) -> (dim)
v_partial = np.dot(A_local, u_local)

# 5. Réduction (Allreduce) pour sommer les vecteurs partiels
v_final = np.zeros(dim, dtype=np.float64)
comm.Allreduce(v_partial, v_final, op=MPI.SUM)

end_time = time()

if rank == 0:
    print(f"--- Décomposition par COLONNES ---")
    print(f"Dimension : {dim}, Processus : {size}")
    print(f"Temps : {end_time - start_time:.6f} s")
    # Vérification simple
    print(f"v (5 premiers éléments) : {v_final[:5]}")