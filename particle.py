from parameters import params
import numpy as np

class HistoryEntry:
    def __init__(self, index, gen, pos, inertia, cognitive, social, best_idx, is_best, best_pos):
        self.gen = gen
        self.pos = pos
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.best_idx = best_idx
        self.index = index
        self.best_pos = best_pos
        self.is_best = is_best

class Particle:
    def __init__(self, pos, vel, pso, index):
        self.pos            = pos
        self.vel            = vel
        self.index          = index
        self.fitness        = self.fitness_update()
        self.personal_best  = pos
        self.best_fitness   = self.fitness
        self.best_pos       = np.array([])
        self.pso            = pso
        self.history        = np.array([[]], dtype=HistoryEntry)
        entry = HistoryEntry(self.pso.generation, self.index, self.pos, np.array([]), np.array([]), np.array([]), self.pso.global_best_index, self.pso.global_best_index == self.index, self.pso.global_best_pos)
        self.history = np.append(self.history, entry)
    
    def update(self):
        inertia = params.w*self.vel
        cognitive = params.c1 * (self.personal_best - self.pos)
        social = params.c2 * (self.pso.global_best_pos - self.pos)
        new_vel = inertia + cognitive + social
        self.pos = self.pos + new_vel
        self.vel = new_vel
        new_fitness = self.fitness_update()
        if new_fitness < self.best_fitness:
            self.personal_best = self.pos
            self.best_fitness = new_fitness
        self.fitness = new_fitness
        entry = HistoryEntry(self.pso.generation, self.index, self.pos, inertia, cognitive, social, self.pso.global_best_index, self.pso.global_best_index == self.index, self.pso.global_best_pos)
        self.history = np.append(self.history, entry)

    def fitness_update(self):
        return self.circle_fitness()
        #return -np.linalg.norm(self.pos)

    def circle_fitness(self):
        fitness = 0
        for i in range(2):
            fitness += self.pos[i] * self.pos[i]
        return fitness