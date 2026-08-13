from parameters import params
import numpy as np
import random
import math


class HistoryEntry:
    def __init__(self, index, gen, pos, inertia, cognitive, social, best_idx, is_best, best_pos, phi1, phi2, fitness, personal_best, fitness_improv=0, portion={}):
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
        self.fitness_improvement = fitness_improv
        self.personal_best = personal_best
        self.portion = portion

class Particle:
    def __init__(self, pos, vel, pso, index, stationary = False):
        self.pos            = pos
        self.vel            = vel
        self.index          = index
        self.fitness        = self.fitness_update()
        self.personal_best  = pos
        self.best_fitness   = self.fitness
        self.best_pos       = np.array([])
        self.global_best_pos = pos
        self.global_best_index = index
        self.global_best_fitness = self.fitness_update()
        self.global_best_pos_temp = pos
        self.global_best_index_temp = index
        self.global_best_fitness_temp = self.fitness_update()
        self.portion        = {index: 0 for index in range(params.popsize)}
        self.portion[self.index] = 1
        self.pso            = pso
        self.history        = np.array([[]], dtype=HistoryEntry)
        self.personal_best_portion = self.portion.copy()
        self.lbest_portion = self.portion.copy()
        self.lbest_portion_temp = self.portion.copy()
        self.vel_portion    = {index: 0 for index in range(params.popsize)}
        entry = HistoryEntry(self.index, self.pso.generation, self.pos, np.array([0,0]), np.array([0,0]), np.array([0,0]), self.pso.global_best_index, self.pso.global_best_index == self.index, self.pso.global_best_pos, np.array([]), np.array([]), self.fitness, personal_best=self.personal_best, portion=self.portion.copy())
        self.history = np.append(self.history, entry)
        self.stationary = stationary
        self.shared_best = 0
    
    def update(self):
        if not self.stationary:
            inertia = params.w*self.vel
            #phi1 = np.array([random.random(), random.random()])
            #phi2 = np.array([random.random(), random.random()])
            phi1 = np.empty(params.dimensions)
            phi1.fill(1)
            phi2 = np.empty(params.dimensions)
            phi2.fill(1)
            cognitive = params.c1 * (self.personal_best - self.pos) * phi1 
            social = params.c2 * (self.global_best_pos - self.pos) * phi2 
            new_vel = inertia + cognitive + social
            #new_vel = self.check_vel(new_vel)
            self.pos = self.pos + new_vel
            self.vel = new_vel
            new_vel_portion = {}
            for key in self.portion.keys():
                new_vel_portion[key] = (params.w * self.vel_portion[key]
                                       + params.c1 * (self.personal_best_portion[key] - self.portion[key])
                                       + params.c2 * (self.lbest_portion[key] - self.portion[key]))
            for key in self.portion.keys():
                self.portion[key] = self.portion[key] + new_vel_portion[key]
            self.vel_portion = new_vel_portion
            #self.clamp()
            #print(f"{self.global_best_pos} {self.global_best_index} {self.index}")
            self.check_fitness()
            last_fitness = self.history[self.pso.generation-1].fitness
            improvement =  np.log(abs(last_fitness) + 1e-10) - np.log(abs(self.fitness) + 1e-10)
            entry = HistoryEntry(self.index, self.pso.generation, self.pos, inertia, cognitive, social, self.global_best_index, self.global_best_index == self.index, self.pso.global_best_pos, phi1, phi2, self.fitness, self.personal_best, improvement, portion=self.portion.copy())
            self.history = np.append(self.history, entry)
            self.pso.overall_steps += 1

    def check_vel(self, vel):
        norm = np.linalg.norm(vel)
        max_v = (params.bounds*2)*0.1
        vel = vel if norm <= max_v else vel * max_v / norm
        return vel

    def clamp(self):
        for i in range(params.dimensions):
            if self.pos[i] < -params.bounds:
                self.pos[i] = -params.bounds
                self.vel = np.empty(params.dimensions)
            if self.pos[i] > params.bounds:
                self.pos[i] = params.bounds
                self.vel = np.empty(params.dimensions)
        


    def fitness_update(self):
        if params.fitness_function == "rosenbrock":
            return self.rosenbrock_fitness()
        elif params.fitness_function == "ackley":
            return self.ackley_fitness()
        elif params.fitness_function == "rastrigin":
            return self.rastrigin_fitness()
        elif params.fitness_function == "sphere":
            return self.circle_fitness()
        return self.pos[0]

    def circle_fitness(self):
        fitness = 0
        for i in range(2):
            fitness += self.pos[i] * self.pos[i]
        return fitness

    def rosenbrock_fitness(self):
        return 100*(((self.pos[1] - (self.pos[0]**2))**2)+ (1-self.pos[0]) ** 2)

    def ackley_fitness(self):
        return -20 * math.exp(-2*math.sqrt(0.5 *(self.pos[0]**2 + self.pos[1]**2))) - math.exp(0.5*(math.cos(2*math.pi*self.pos[0]) + math.cos(2*math.pi*self.pos[1]))) + math.e + 20
    
    def rastrigin_fitness(self):
        return 10*2 + (self.pos[0]**2 - 10*math.cos(2*math.pi*self.pos[0]))+ (self.pos[1]**2 - 10*math.cos(2*math.pi*self.pos[1]))

    def check_fitness(self):
        new_fitness = self.fitness_update()
        if new_fitness < self.best_fitness:
            global_change = False
            self.personal_best = self.pos
            self.best_fitness = new_fitness
            self.pso.local_change = True
            self.personal_best_portion = self.portion.copy()
            if new_fitness < self.pso.global_best:
                global_change = True
                self.pso.global_change = True
        if new_fitness < self.global_best_fitness:
            self.global_best_fitness = new_fitness
            self.global_best_index = self.index
            self.global_best_pos = self.pos
            self.lbest_portion = self.portion.copy()
    
        self.fitness = new_fitness