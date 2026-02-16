from particle import Particle
from parameters import params
import random
import numpy as np

class FitnessHistoryEntry:
    def __init__(self, global_change, local_change, fitness, generation, fitness_global_change):
        self.global_change = global_change
        self.local_change = local_change
        self.fitness = fitness
        self.fitness_global_change = fitness_global_change
        self.gen = generation

class PSO:
    def __init__(self):
        self.particles = []
        self.global_best = 100000000
        self.global_best_pos = np.array([1000, 1000])
        self.global_best_index = 0
        self.generation = 0
        self.global_change = False
        self.fitness_global_change = 0
        self.local_change = False
        self.overall_steps = 0
        self.history = np.array([], dtype=FitnessHistoryEntry)


    def init_population(self, to_remove):
        idx = 0
        poses = np.load("saves/start_pos.npy")
        for i in range(params.popsize):
            if i not in to_remove:
                if len(to_remove) == 0:
                    pos = np.array([random.randint(-10,10), random.randint(-10,10)])
                else:
                    pos = poses[i]
                vel = np.array([0,0])
                p = Particle(pos, vel, self, i)
                self.particles.append(p)
                if p.fitness < self.global_best:
                    self.global_best = p.fitness
                    self.global_best_pos = p.pos
                    self.global_best_index = idx
            else:
                pos = np.array([10000, 10000])
                vel = np.array([0,0])
                p = Particle(pos, vel, self, i, True)
                self.particles.append(p)
            idx += 1
        historyEntry = FitnessHistoryEntry(self.global_change, self.local_change, self.getCurrentMeanFitness(), self.generation, self.fitness_global_change)
        self.history = np.append(self.history, historyEntry)
        self.generation += 1  
        
    def update_step(self):
        for idx in range(len(self.particles)):
            p = self.particles[idx]
            p.update()
            if p.fitness < self.global_best:
                self.fitness_global_change = self.global_best / p.fitness
                self.global_best = p.fitness
                self.global_best_pos = p.pos
                self.global_best_index = idx
        historyEntry = FitnessHistoryEntry(self.global_change, self.local_change, self.getCurrentMeanFitness(), self.generation, self.fitness_global_change)
        self.fitness_global_change = 0
        self.history = np.append(self.history, historyEntry)
        self.global_change = False
        self.local_change = False
        self.generation += 1
        return self.global_best_pos

    def getCurrentMeanFitness(self):
        fitness_sum = 0
        amount = 0
        for i in self.particles:
            if not i.stationary:
                fitness_sum += i.fitness
                amount += 1
        return fitness_sum / amount
