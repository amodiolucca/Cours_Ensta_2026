import math

def bucket_sort(array, num_buckets=10):
    if len(array) == 0:
        return array

    min_value = min(array)
    max_value = max(array)
    
    # Évite la division par zéro si tous les éléments sont identiques
    if min_value == max_value:
        return array

    # Plage totale des données (Range)
    total_range = (max_value - min_value)
    
    buckets = [[] for _ in range(num_buckets)]

    for value in array:
        # Formule : Normalisation (valeur - min) / étendue -> transforme en 0.0-1.0
        # Puis multiplication par num_buckets pour trouver l'index
        position = int(((value - min_value) / total_range) * num_buckets)
        
        # Cas limite : Si la valeur est exactement max_value, l'index déborde.
        # On la place dans le dernier bucket.
        if position == num_buckets:
            position -= 1
            
        buckets[position].append(value)
    
    # Tri individuel de chaque bucket et concaténation
    sorted_array = []
    for bucket in buckets:
        bucket.sort()
        sorted_array.extend(bucket) 
    
    return sorted_array


# Test avec des nombres flottants
lista = [0.897, 0.565, 0.656, 0.1234, 0.665, 0.3434]
print("--- Test 1 (Floats) ---")
print(f"Original : {lista}")
print(f"Trié     : {bucket_sort(lista)}")

print("\n")

# Test avec des nombres entiers
lista_inteiros = [10, 50, 5, 20, 99, 4] 
print("--- Test 2 (Entiers) ---")
print(f"Original : {lista_inteiros}")
print(f"Trié     : {bucket_sort(lista_inteiros)}")