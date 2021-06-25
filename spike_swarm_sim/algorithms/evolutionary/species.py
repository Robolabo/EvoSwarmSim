import logging
import numpy as np

class Species:
    def __init__(self, id, generation, compatib_thresh=3, c1=1, c2=1, c3=2.):
        self.id = id
        self.compatib_thresh = compatib_thresh
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.num_genotypes = 0
        self.representative = {}
        self.mean_fitness = {'raw' : 0, 'adjusted' : 0}
        self.max_fitness = {'raw' : 0, 'adjusted' : 0}
        self.min_fitness = {'raw' : 0, 'adjusted' : 0}
        self.fitness_sum = {'raw' : 0, 'adjusted' : 0}
        self.creation_generation = generation
        self.history = {key : [] for key in ['num_genotypes', 'mean_fitness',
                                        'max_fitness', 'min_fitness', 'sum_fitness']}
        self.last_improvement = 0

    def compatibility(self, genotype):
        """ Computes the compatibility distance of the genotype to the species 
        as defined in the NEAT paper. It returns both the distance and whether the 
        genotype is compatible to the species or not.
        ============================================================================
        - Args:
            genotype [dict] :
        - Returns:
            is_compatible [bool] : whether the genotype is compatible or not.
            distance [float] : compatibility distance of the genotype to the species 
                    representative.
        ============================================================================
        """
        if self.representative is None or len(self.representative) == 0:
            return (False, 1000.)
        repr_innovations = set([g['innovation'] for g in self.representative['connections'].values()])
        genotype_innovations = set([g['innovation'] for g in genotype['connections'].values()])
        # Do not care about disjoint and excess. For the moment we use same weights.
        diff_genes = genotype_innovations - repr_innovations
        common_genes = genotype_innovations.intersection(repr_innovations)
        weights_repr = np.array([g['weight'] for g in self.representative['connections'].values()
                        if g['innovation'] in common_genes])
        weights_genotype = np.array([g['weight'] for g in genotype['connections'].values()
                        if g['innovation'] in common_genes])
        assert len(weights_repr) == len(weights_genotype)
        # W_dist = np.abs(weights_repr.mean() - weights_genotype.mean()) #!CHECK
        W_dist = np.linalg.norm(weights_repr - weights_genotype) / np.sqrt(len(weights_genotype))
        # dist = 2 * self.c1 * (len(diff_genes) / max(len(weights_repr), len(weights_genotype))) \
        #         + self.c3 * W_dist
        dist = 2 * self.c1 * len(diff_genes) / max(len(repr_innovations), len(genotype_innovations)) + self.c3 * W_dist
        
        return dist < self.compatib_thresh, dist

    def update_stats(self, fitness_scores):
        self.history['num_genotypes'].append(self.num_genotypes)
        self.history['mean_fitness'].append(np.mean(fitness_scores))
        self.history['max_fitness'].append(max(fitness_scores))
        self.history['min_fitness'].append(min(fitness_scores))
        self.history['sum_fitness'].append(sum(fitness_scores))

        adj_fitness_scores = fitness_scores.copy() / self.num_genotypes
        self.mean_fitness.update({
            'raw' : np.mean(fitness_scores),
            'adjusted': np.mean(adj_fitness_scores)
        })
        self.max_fitness.update({
            'raw' : max(fitness_scores),
            'adjusted': max(adj_fitness_scores)
        })
        self.min_fitness.update({
            'raw' : min(fitness_scores),
            'adjusted': min(adj_fitness_scores)
        })
        self.fitness_sum.update({
            'raw' : sum(fitness_scores),
            'adjusted': sum(adj_fitness_scores)
        })

    @property
    def is_extinct(self):
        return self.last_improvement >= 15
        
