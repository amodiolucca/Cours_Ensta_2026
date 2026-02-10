from mpi4py import MPI
import numpy as np

def parallel_bucket_sort():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Configuration
    N = 20           # Nombre total d'éléments
    MAX_VAL = 100    # Valeur maximale (exclusive)

    #Le processus 0 génère un tableau de nombres arbitraires
    if rank == 0:
        data = np.random.randint(0, MAX_VAL, N)
        print(f"[Processus {rank}] Données originales : {data}")
        chunks = np.array_split(data, size)
    else:
        chunks = None

    #Dispatch aux autres processus (Scatter)
    local_data = comm.scatter(chunks, root=0)
    print(f"[Processus {rank}] Chunk initial reçu : {len(local_data)} éléments")

    #Tri en parallèle

    #Répartition dans les buckets
    buckets = [[] for _ in range(size)]
    
    for number in local_data:
        # Calcul de l'index du bucket
        bucket_index = int((number / MAX_VAL) * size)
        if bucket_index == size: 
            bucket_index -= 1
        buckets[bucket_index].append(number)

    #Échange des données (All-to-All)
    received_buckets = comm.alltoall(buckets)
    
    # Tri local
    local_final_list = []
    for bucket in received_buckets:
        local_final_list.extend(bucket)
    
    local_final_list.sort()
    print(f"[Processus {rank}] Liste locale triée : {local_final_list}")

    #Rassemblement sur le processus 0 (Gather)
    gathered_data = comm.gather(local_final_list, root=0)

    if rank == 0:
        # Aplatir la liste (flatten)
        final_sorted_array = [item for sublist in gathered_data for item in sublist]
        
        print("-" * 50)
        print("RÉSULTAT FINAL SUR LE PROCESSUS 0 :")
        print(final_sorted_array)
        
        # Vérification
        if all(final_sorted_array[i] <= final_sorted_array[i+1] for i in range(len(final_sorted_array)-1)):
            print("Succès : Le tableau est trié !")
        else:
            print("Erreur : Le tableau n'est PAS trié.")

if __name__ == "__main__":
    parallel_bucket_sort()