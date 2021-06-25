import copy
import numpy as np
from spike_swarm_sim.register import evo_operator_registry



@evo_operator_registry(name='truncation_selection')
def truncation_selection(population, fitness, n_sel):
    """ Truncation selection operator of GA. The n_sel best individuals, 
    according to the fitness scores, are selected as parents.
    =========================================================================
    - Args:
        population [list]: list of genotypes from which parents are selected.
        fitness [list]: evaluated fitness scores of the genotypes in population.
        n_sel [int]: number of genotypes to be selected.
    - Returns:
        selected [list]: list of selected genotypes.
        fitness_sel [list]: list of the fitness score corresponding to the 
            selected genotypes (respecting the same order)
    =========================================================================
    """
    f_order = np.argsort(fitness.copy())[::-1]
    selected = [copy.deepcopy(population[i]) for i in f_order[:n_sel]]
    fitness_sel = [fitness[i] for i in f_order[:n_sel]]
    return selected, fitness_sel

@evo_operator_registry(name='roulette_selection')
def roulette_selection(population, fitness, n_sel):
    """ Roulette selection operator of GA. It assigns a fitness proportionale 
    probability to each genotype and samples n_sel parents from a categorical 
    dist. with the computed probabilities.
    =========================================================================
    - Args:
        population [list]: list of genotypes from which parents are selected.
        fitness [list]: evaluated fitness scores of the genotypes in population.
        n_sel [int]: number of genotypes to be selected.
    - Returns:
        selected [list]: list of selected genotypes.
        fitness_sel [list]: list of the fitness score corresponding to the 
            selected genotypes (respecting the same order)
    =========================================================================
    """
    probs = [f / np.sum(fitness) for f in fitness]
    sel_idxs = np.random.choice(len(population), p=probs, replace=True, size=n_sel)
    selected = [population[i] for i in sel_idxs]
    fitness_sel = [fitness[i] for i in sel_idxs]
    return selected, fitness_sel

@evo_operator_registry(name='lin_rank_selection')
def lin_rank_selection(population, fitness, n_sel, sp=1.5):
    """ Linear rank selection operator of GA. Samples stochastically n_sel 
    genotypes using ranks as probabilities in order to mitigate genetic drift.
    Ranks are computed linearly, so that the prob. decreases linearly with the 
    fitness order of the individuals (fittest is the most probable).
    =========================================================================
    - Args:
        population [list]: list of genotypes from which parents are selected.
        fitness [list]: evaluated fitness scores of the genotypes in population.
        n_sel [int]: number of genotypes to be selected.
        sp [float]: defines the slope and intercept of the ranking computation.
    - Returns:
        selected [list]: list of selected genotypes.
        fitness_sel [list]: list of the fitness score corresponding to the 
            selected genotypes (respecting the same order)
    =========================================================================
    """
    fitness_order = np.argsort(fitness)
    population = [population[ii] for ii in fitness_order]
    # * First individual is the worst
    probs = [2 - sp + 2 * (sp - 1) * i / (len(population) - 1) for i in range(len(population))]
    probs[0] += 1 - sum(probs)

    sel_idxs = np.random.choice(len(population), replace=True, p=probs, size=n_sel)
    selected = [population[i] for i in sel_idxs]
    fitness_sel = [fitness[i] for i in sel_idxs]
    return selected, fitness_sel

@evo_operator_registry(name='nonlin_rank_selection')
def nonlin_rank_selection(population, fitness, n_sel, p_best=.1):
    """ Non-linear rank selection operator of GA. Samples stochastically n_sel 
    genotypes using ranks as probabilities in order to mitigate genetic drift.
    Ranks are computed non-linearly, so that the prob. decreases exponentially with the 
    fitness order of the individuals (fittest is the most probable).
    =========================================================================
    - Args:
        population [list]: list of genotypes from which parents are selected.
        fitness [list]: evaluated fitness scores of the genotypes in population.
        n_sel [int]: number of genotypes to be selected.
        p_best [float]: prob. of the fittest individual (prob of rank 0). 
    - Returns:
        selected [list]: list of selected genotypes.
        fitness_sel [list]: list of the fitness score corresponding to the 
            selected genotypes (respecting the same order)
    =========================================================================
    """
    fitness_order = np.argsort(fitness)[::-1]
    population = [population[ii] for ii in fitness_order]
    probs = [p_best * (1 - p_best) ** i for i in range(len(population))]
    probs[0] += 1 - sum(probs)
    sel_idxs = np.random.choice(len(population), p=probs, replace=True, size=n_sel)
    selected = [population[i] for i in sel_idxs]
    fitness_sel = [fitness[i] for i in sel_idxs]
    # selected = [v for v, _ in sorted(zip(selected, fitness_sel), key=lambda x: x[1])]
    return selected, fitness_sel


@evo_operator_registry(name='tournament_selection')
def tournament_selection(population, fitness, n_sel, tournament_size=3):
    """ Tournament selection operator of GA. Firstly, n_sel tournaments are 
    arraged, with tournament contestants elected randomly (with same prob., 
    regardless of their fitness). For each tournament, of sizes tournament_size,
    a winner is selected deterministically as the fittest genotype in the group.
    =========================================================================
    - Args:
        population [list]: list of genotypes from which parents are selected.
        fitness [list]: evaluated fitness scores of the genotypes in population.
        n_sel [int]: number of genotypes to be selected.
        tournament_size [int]: size of each tournament.
    - Returns:
        selected [list]: list of selected genotypes.
        selected_fitness [list]: list of the fitness score corresponding to the 
            selected genotypes (respecting the same order)
    =========================================================================
    """
    selected = []
    selected_fitness = []
    for _ in range(n_sel):
        competitors_idx = np.random.choice(len(population), replace=True, size=tournament_size)
        competitors = [population[idx] for idx in competitors_idx]
        competitors_fitness = [fitness[idx] for idx in competitors_idx]
        selected.append(competitors[np.argmax(competitors_fitness)])
        selected_fitness.append(np.max(competitors_fitness))
    # selected = [v for v, _ in sorted(zip(selected, selected_fitness), key=lambda x: x[1])]
    return selected, selected_fitness


@evo_operator_registry(name='stochastic_universal_selection')
def stochastic_universal_selection(population, fitness, n_sel):
    """ Stochastic Universal Selection operator.
    #TODO Comment and test this selection operator (may have errors).
    =========================================================================
    - Args:
        population [list]: list of genotypes from which parents are selected.
        fitness [list]: evaluated fitness scores of the genotypes in population.
        n_sel [int]: number of genotypes to be selected.
    - Returns:
        selected [list]: list of selected genotypes.
        selected_fitness [list]: list of the fitness score corresponding to the 
            selected genotypes (respecting the same order)
    =========================================================================
    """
    P = sum(fitness) / n_sel
    start = np.random.random()*P
    pointers = [start + i*P for i in range(n_sel)] 
    selected = []
    selected_fitness = []
    for pt in pointers:
        i = 0
        while sum(fitness[:i] < pt): 
            i += 1 
        selected.append(population[i])
        selected_fitness.append(fitness[i])
    return selected, selected_fitness