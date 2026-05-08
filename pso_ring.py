from particle_ring import Particle
from parameters import params
import random
import math
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

class PSO_Ring:
    def __init__(self):
        self.particles = []
        self.global_best = 100000000
        self.global_best_pos = np.array([])
        self.global_best_temp_pos = np.array([])
        for i in range(params.dimensions):
            self.global_best_pos = np.append(self.global_best_pos, [1000])
            self.global_best_temp_pos = np.append(self.global_best_pos, [1000])
            
        self.global_best_index = 0
        self.global_best_temp_index = 0
        self.first_global_overall_fitness = 0
        self.generation = 0
        self.global_change = False
        self.global_best_ratio = 0
        self.global_fitness = 0
        self.local_change = False
        self.overall_steps = 0
        self.history = np.array([], dtype=FitnessHistoryEntry)

    def getlbestFitness(self, id):
        distances = {}
        for idx in range(len(self.particles)):
            if id != idx:
                distances.update({idx: self.distance(self.particles[id].pos, self.particles[idx].pos)})
        distances = dict(sorted(distances.items(), key=lambda item: item[1]))
        ids = list(distances.keys())
        best_fitness = min(self.particles[id].global_best_fitness, min(self.particles[ids[0]].global_best_fitness, self.particles[ids[1]].global_best_fitness))
        if best_fitness == self.particles[ids[0]].global_best_fitness:
            return ids[0]
        if best_fitness == self.particles[ids[1]].global_best_fitness:
            return ids[1]
        return id
        

    def distance(self, first, second):
        dist = 0
        for i in range(params.dimensions):
            dist = dist + ((first[i] - second[i])**2)
        return math.sqrt(dist)

    def init_population(self, to_remove):
        idx = 0
        made_up = True
        poses = np.load("saves/start_pos.npy")
        for i in range(params.popsize):
            if i not in to_remove:
                if len(to_remove) == 0:
                    pos = np.array([])
                    for dim in range(params.dimensions):
                        if not made_up:
                            rand = random.randint(-params.bounds, params.bounds)
                            pos = np.append(pos, rand)
                        else:
                            if i == 0:
                                pos = np.array([-5, 3])
                            else:
                                pos = np.array([-3, i/2-10])
                            '''if i == 0:
                                pos = np.append(pos,-1)
                            else:
                                pos = np.append(pos, 0)'''
                else:
                    pos = poses[i]
                vel = np.empty(params.dimensions)
                vel.fill(0)
                p = Particle(pos, vel, self, i)
                self.particles.append(p)
            else:
                pos = np.array([10000, 10000])
                vel = np.array([0,0])
                p = Particle(pos, vel, self, i, True)
                self.particles.append(p)
            idx += 1
        self.global_best_pos = self.global_best_temp_pos
        self.global_best_index = self.global_best_temp_index
        self.first_global_overall_fitness = self.getCurrentMeanFitness()
        historyEntry = FitnessHistoryEntry(self.global_change, self.local_change, self.getCurrentMeanFitness(), self.generation, self.global_best_ratio)
        self.history = np.append(self.history, historyEntry)
        self.global_fitness = self.getCurrentMeanFitness()
        self.generation += 1  
        
    def update_step(self):
        for idx in range(len(self.particles)):
            p = self.particles[idx]
            p.update()
            best_neighbor = self.particles[self.getlbestFitness(idx)]
            if best_neighbor.global_best_fitness < p.global_best_fitness:
                p.global_best_fitness_temp = best_neighbor.global_best_fitness
                p.global_best_index_temp = best_neighbor.global_best_index
                p.global_best_pos_temp = best_neighbor.global_best_pos
            
            if p.fitness < self.global_best:
                self.global_best_ratio = self.global_best / p.fitness
                self.global_best = p.fitness
                self.global_best_temp_pos = p.pos
                self.global_best_temp_index = idx

        for idx in range(len(self.particles)):
            p = self.particles[idx]
            p.global_best_fitness = p.global_best_fitness_temp
            p.global_best_index = p.global_best_index_temp
            p.global_best_pos = p.global_best_pos_temp


        self.global_best_pos = self.global_best_temp_pos
        self.global_best_index = self.global_best_temp_index
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

