from particle import Particle
from parameters import params
import random
import numpy as np

class PSO:
    def __init__(self):
        self.particles = []
        self.global_best = 10000
        self.global_best_pos = np.array([1000, 1000])
        self.global_best_index = 0
        self.generation = 0
        self.overall_steps = 0


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
        
    def update_step(self):
        for idx in range(len(self.particles)):
            p = self.particles[idx]
            p.update()
            if p.fitness < self.global_best:
                self.global_best = p.fitness
                self.global_best_pos = p.pos
                self.global_best_index = idx
        self.generation += 1
        return self.global_best_pos
    
