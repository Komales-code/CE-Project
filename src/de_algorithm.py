import numpy as np
from src.model import evaluate_svr

def differential_evolution(
    X_train, X_test, y_train, y_test,
    pop_size=30, generations=50,
    F=0.8, CR=0.9
):
    bounds = [
        (0.1, 100),     # C
        (0.01, 1),      # epsilon
        (0.0001, 1)     # gamma
    ]

    population = np.array([
        [np.random.uniform(low, high) for low, high in bounds]
        for _ in range(pop_size)
    ])

    fitness = np.array([
        evaluate_svr(ind, X_train, X_test, y_train, y_test)
        for ind in population
    ])

    history = []

    for gen in range(generations):
        for i in range(pop_size):
            idxs = [idx for idx in range(pop_size) if idx != i]
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]

            mutant = a + F * (b - c)
            mutant = np.clip(mutant,
                              [b[0] for b in bounds],
                              [b[1] for b in bounds])

            crossover = np.random.rand(len(bounds)) < CR
            trial = np.where(crossover, mutant, population[i])

            trial_fitness = evaluate_svr(
                trial, X_train, X_test, y_train, y_test
            )

            if trial_fitness < fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness

        history.append(fitness.min())

    best_idx = np.argmin(fitness)
    return population[best_idx], fitness[best_idx], history

