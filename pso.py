from particle import Particle
from parameters import params
import random
import numpy as np

class FitnessHistoryEntry:
    def __init__(self, global_change, local_change, fitness, generation, global_best_ratio, fitness_ratio=0, global_improvement=0):
        self.global_change = global_change
        self.local_change = local_change
        self.fitness = fitness
        self.global_best_ratio = global_best_ratio
        self.gen = generation
        self.fitness_ratio = fitness_ratio
        self.global_improvement = global_improvement

class PSO:
    def __init__(self):
        self.particles = []
        self.global_best = 100000000
        self.global_best_pos = np.array([])
        for i in range(params.dimensions):
            self.global_best_pos = np.append(self.global_best_pos, [1000])
        self.global_best_index = 0
        self.first_global_overall_fitness = 0
        self.generation = 0
        self.global_change = False
        self.global_best_ratio = 0
        self.global_fitness = 0
        self.local_change = False
        self.overall_steps = 0
        self.history = np.array([], dtype=FitnessHistoryEntry)


    def init_population(self, to_remove):
        idx = 0
        poses = np.load("saves/start_pos.npy")
        for i in range(params.popsize):
            if i not in to_remove:
                if len(to_remove) == 0:
                    pos = np.array([])
                    for dim in range(params.dimensions):
                        rand = random.randint(-10,10)
                        pos = np.append(pos,rand)
                else:
                    pos = poses[i]
                vel = np.empty(params.dimensions)
                vel.fill(0)
                p = Particle(pos, vel, self, i)
                self.particles.append(p)
                if p.fitness < self.global_best:
                    self.global_best = p.fitness
                    self.global_best_pos = p.pos
                    self.global_best_index = idx
                    self.global_change = True
            else:
                pos = np.array([10000, 10000])
                vel = np.array([0,0])
                p = Particle(pos, vel, self, i, True)
                self.particles.append(p)
            idx += 1
        self.first_global_overall_fitness = self.getCurrentMeanFitness()
        historyEntry = FitnessHistoryEntry(self.global_change, self.local_change, self.getCurrentMeanFitness(), self.generation, self.global_best_ratio)
        self.history = np.append(self.history, historyEntry)
        self.global_fitness = self.getCurrentMeanFitness()
        self.generation += 1  
        
    def update_step(self):
        for idx in range(len(self.particles)):
            p = self.particles[idx]
            p.update()
            if p.fitness < self.global_best:
                self.global_best_ratio = self.global_best / p.fitness
                self.global_best = p.fitness
                self.global_best_pos = p.pos
                self.global_best_index = idx
        historyEntry = FitnessHistoryEntry(self.global_change, self.local_change, self.getCurrentMeanFitness(), self.generation, self.global_best_ratio, self.global_fitness/self.getCurrentMeanFitness(), self.first_global_overall_fitness/self.getCurrentMeanFitness())
        self.global_fitness = self.getCurrentMeanFitness()
        self.global_best_ratio = float(1.000000)
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
