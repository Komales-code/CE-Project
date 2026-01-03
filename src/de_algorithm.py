import numpy as np
from src.model import evaluate_svr

fitness_cache = {}

def cached_fitness(params, X_train, X_test, y_train, y_test):
    key = tuple(np.round(params, 4))
    if key not in fitness_cache:
        fitness_cache[key] = evaluate_svr(
            params, X_train, X_test, y_train, y_test
        )
    return fitness_cache[key]

def differential_evolution(
    X_train, X_test, y_train, y_test,
    pop_size=15,
    generations=20,
    F=0.8,
    CR=0.9,
    progress_callback=None
):
    bounds = [
        (0.1, 100),
        (0.01, 1),
        (0.0001, 1)
    ]

    population = np.array([
        [np.random.uniform(low, high) for low, high in bounds]
        for _ in range(pop_size)
    ])

    fitness = np.array([
        cached_fitness(ind, X_train, X_test, y_train, y_test)
        for ind in population
    ])

    history = []

    for gen in range(generations):
        for i in range(pop_size):
            idxs = [idx for idx in range(pop_size) if idx != i]
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]

            mutant = a + F * (b - c)
            mutant = np.clip(
                mutant,
                [b[0] for b in bounds],
                [b[1] for b in bounds]
            )

            cross = np.random.rand(len(bounds)) < CR
            trial = np.where(cross, mutant, population[i])

            trial_fitness = cached_fitness(
                trial, X_train, X_test, y_train, y_test
            )

            if trial_fitness < fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness

        history.append(fitness.min())

        if progress_callback:
            progress_callback(gen + 1)

    best_idx = np.argmin(fitness)
    return population[best_idx], fitness[best_idx], history
