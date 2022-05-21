"""Samplers for Single Path One-Shot Search Space."""

import numpy as np
from copy import deepcopy
from collections import deque


class RAND():
    """Random choice"""
    def __init__(self, num_choice, layers):
        self.num_choice = num_choice
        self.child_len = layers
        self.history = []
        
    def record(self, child, value):
        self.history.append({"child":child, "value":value})
    
    def suggest(self):
        return list(np.random.randint(self.num_choice, size=self.child_len))
    
    def final_best(self):
        best_child = min(self.history, key=lambda i:i["value"])
        return best_child['child'], best_child['value']


class REA():
    """Regularized Evolution Algorithm"""
    def __init__(self, num_choice, layers, population_size=20, better=min):
        self.num_choice = num_choice
        self.population_size = population_size
        self.child_len = layers
        self.better = better
        self.population = deque()
        self.history = []
        # init population
        self.init_pop = np.random.randint(
            self.num_choice, size=(self.population_size, self.child_len)
        )

    def _get_mutated_parent(self):
        parent = self.better(self.population, key=lambda i:i["value"])  # default: min(error)
        return self._mutate(parent['child'])

    def _mutate(self, parent):
        parent = deepcopy(parent)
        idx = np.random.randint(0, len(parent))
        prev_value, new_value = parent[idx], parent[idx]
        while new_value == prev_value:
            new_value = np.random.randint(self.num_choice)
        parent[idx] = new_value
        return parent

    def record(self, child, value):
        self.history.append({"child":child, "value":value})
        self.population.append({"child":child, "value":value})
        if len(self.population) > self.population_size:
            self.population.popleft()

    def suggest(self):
        if len(self.history) < self.population_size:
            return list(self.init_pop[len(self.history)])
        else:
            return self._get_mutated_parent()
    
    def final_best(self):
        best_child = self.better(self.history, key=lambda i:i["value"])
        return best_child['child'], best_child['value']
