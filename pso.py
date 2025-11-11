from particle import Particle
from parameters import params
import random
import numpy as np

class PSO:
    def __init__(self):
        self.particles = []
        self.global_best = 10000
        self.global_best_pos = None
        self.global_best_index = 0
        self.generation = 0


    def init_population(self):
        idx = 0
        for i in range(params.popsize):
            if idx == 900:
                pos = np.array([0,0])
            else:
                pos = np.array([random.randint(1,10), random.randint(1,10)])
            vel = np.array([0,0])
            p = Particle(pos, vel, self, i)
            self.particles.append(p)
            if p.fitness < self.global_best:
                self.global_best = p.fitness
                self.global_best_pos = p.pos
                self.global_best_index = idx
            idx += 1
        
    def update_step(self):
        for idx in range(len(self.particles)):
            p = self.particles[idx]
            p.update()
            if p.fitness < self.global_best:
                self.global_best = p.fitness
                self.global_best_pos = p.pos
                self.global_best_index = idx
        self.generation += 1
    
