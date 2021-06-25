import numpy as np
from spike_swarm_sim.register import evo_operator_registry

@evo_operator_registry(name='random_mating')
def random_mating(parents, fitness=None):
    np.random.shuffle(parents)
    return parents

def nearest_fitness(parents, fitness):
    parents = [v for v, _ in sorted(zip(parents, fitness), key=lambda x: x[1])]
    return parents

def nearest_genotype(parents, fitness=None):
    np.random.shuffle(parents)
    genotype = parents.pop(0)
    parents_ordered = [genotype.copy()]
    while len(parents):
        nearest = np.argmin([np.linalg.norm(genotype - v) for j, v in enumerate(parents)])
        genotype = parents.pop(nearest)
        parents_ordered.append(genotype.copy())
    return parents_ordered

def speciation(parents, fitness, n_species=2):
    parents_ordered = []
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_species)
    kmeans.fit(parents)
    species = kmeans.predict(parents) # the index of the cluster (specie) of each genotype.
    for sp in range(kmeans.cluster_centers_.shape[0]):
        species_pop = [parents[i] for i, vv in enumerate(species == sp) if vv]
        np.random.shuffle(species_pop)
        parents_ordered.extend(species_pop)
    return parents_ordered

