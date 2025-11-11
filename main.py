from particle import Particle
import numpy as np
from pso import PSO
import pygame
if __name__ == '__main__':
    pso = PSO()
    debug = True
    graphical = False
    pso.init_population()
    histories = []
    occurences = {}
    for i in range(30):
        pso.update_step()
    for part in pso.particles:
        histories.append(part.history)
    for hist in histories:
        for entry in hist:
            if entry.best_idx in occurences:
                occurences[entry.best_idx] = occurences[entry.best_idx]+1
            else:
                occurences.update({entry.best_idx: 1})
    
    if debug:
        print(pso.global_best_pos)
        print(pso.global_best)
        print(occurences)
        highest_occur = max(occurences, key=occurences.get)
        best_part = pso.particles[pso.global_best_index].history
        print(pso.particles[highest_occur].history[0].pos)
        print("--------")
        for entry in best_part:
            print(entry.best_pos, entry.best_idx)
        #for entry in pso.particles[highest_occur].history:
        #print(entry.pos, entry.is_best)

    if graphical:
        pso = PSO()
        debug = False
        pso.init_population()
        pygame.init()
        screen = pygame.display.set_mode((800, 600))
        clock = pygame.time.Clock()

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            screen.fill((0,0,0))
            pso.update_step()
            for p in pso.particles:
                pygame.draw.circle(screen, (0, 255, 255), (int(p.pos[0])/10*800, int(p.pos[1])/10*600), 4)
            pygame.draw.circle(screen, (255, 0, 0), (200,200), 6)
            pygame.display.flip()
            clock.tick()
        pygame.quit()