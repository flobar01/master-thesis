from parameters import params
import numpy as np
import random

class FitnessHistoryEntry:
    def __init__(self, global_change, local_change, fitness, generation):
        self.global_change = global_change
        self.local_change = local_change
        self.fitness = fitness
        self.gen = generation

class HistoryEntry:
    def __init__(self, index, gen, pos, inertia, cognitive, social, best_idx, is_best, best_pos, phi1, phi2, fitness):
        self.gen = gen
        self.pos = pos
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.best_idx = best_idx
        self.index = index
        self.best_pos = best_pos
        self.is_best = is_best
        self.phi1 = phi1
        self.phi2 = phi2
        self.fitness = fitness

class Particle:
    def __init__(self, pos, vel, pso, index, stationary = False):
        self.pos            = pos
        self.vel            = vel
        self.index          = index
        self.fitness        = self.fitness_update()
        self.personal_best  = pos
        self.best_fitness   = self.fitness
        self.best_pos       = np.array([])
        self.pso            = pso
        self.history        = np.array([[]], dtype=HistoryEntry)
        self.composition    = {"test": 1}
        entry = HistoryEntry(self.pso.generation, self.index, self.pos, np.array([]), np.array([]), np.array([]), self.pso.global_best_index, self.pso.global_best_index == self.index, self.pso.global_best_pos, np.array([]), np.array([]), self.fitness)
        self.history = np.append(self.history, entry)
        self.stationary = stationary
    
    def update(self):
        if not self.stationary:
            inertia = params.w*self.vel
            phi1 = np.array([random.random(), random.random()])
            phi2 = np.array([random.random(), random.random()])
            phi1 = np.array([1,1])
            phi2 = np.array([1,1])
            cognitive = params.c1 * (self.personal_best - self.pos) * phi1 
            social = params.c2 * (self.pso.global_best_pos - self.pos) * phi2 
            new_vel = inertia + cognitive + social
            self.pos = self.pos + new_vel
            self.vel = new_vel
            new_fitness = self.fitness_update()
            if new_fitness < self.best_fitness:
                global_change = False
                self.personal_best = self.pos
                self.best_fitness = new_fitness
                if new_fitness < self.pso.global_best:
                    global_change = True
                    #self.pso.global_best = new_fitness
                    #self.pso.global_best_pos = self.pos
                    #self.pso.global_best_idx = self.index
                historyEntry = FitnessHistoryEntry(global_change, True, self.pso.getCurrentMeanFitness(), self.pso.generation)
                self.pso.history = np.append(self.pso.history, historyEntry)
            self.fitness = new_fitness
            entry = HistoryEntry(self.pso.generation, self.index, self.pos, inertia, cognitive, social, self.pso.global_best_index, self.pso.global_best_index == self.index, self.pso.global_best_pos, phi1, phi2, self.fitness)
            self.history = np.append(self.history, entry)
            self.pso.overall_steps += 1
    #test function
    def fitness_update(self):
        return self.rosenbrock_fitness()

    def circle_fitness(self):
        fitness = 0
        for i in range(2):
            fitness += self.pos[i] * self.pos[i]
        return fitness

    def rosenbrock_fitness(self):
        return 100*(((self.pos[1] - (self.pos[0]**2))**2)+ (1-self.pos[0]) ** 2)