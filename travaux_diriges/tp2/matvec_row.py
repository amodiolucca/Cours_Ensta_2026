# matvec_row.py
from mpi4py import MPI
import numpy as np
from time import time

# Configuration MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Dimension du problème
dim = 4800

# 1. Calcul de N_loc (Lignes par processus)
if dim % size != 0:
    if rank == 0: print("Erreur : La dimension doit être divisible par le nombre de processus.")
    comm.Abort()

N_loc = dim // size

# Intervalle de lignes pour ce processus
start_row = rank * N_loc
end_row   = (rank + 1) * N_loc

# 2. Initialisation locale de la matrice A (Seulement N_loc lignes)
# A_local est de taille (N_loc, dim)
A_local = np.empty((N_loc, dim), dtype=np.float64)
for i_local in range(N_loc):
    i_global = start_row + i_local
    for j in range(dim):
        A_local[i_local, j] = (i_global + j) % dim + 1

# 3. Initialisation du vecteur u (Complet)
# Dans la décomposition par lignes, chaque processus a besoin du vecteur u entier
u = np.array([i + 1.0 for i in range(dim)])

comm.Barrier()
start_time = time()

# 4. Calcul du produit local
# (N_loc x dim) * (dim) -> (N_loc)
v_local = np.dot(A_local, u)

# 5. Rassemblement (Allgather) pour reconstituer le vecteur complet
# Chaque processus aura le vecteur v complet à la fin
v_final = np.empty(dim, dtype=np.float64)
comm.Allgather(v_local, v_final)

end_time = time()

if rank == 0:
    print(f"--- Décomposition par LIGNES ---")
    print(f"Dimension : {dim}, Processus : {size}")
    print(f"Temps : {end_time - start_time:.6f} s")
    print(f"v (5 premiers éléments) : {v_final[:5]}")