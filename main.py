from particle import Particle
import numpy as np
from pso import PSO
from os import environ
import math
from parameters import params
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import pygame

def distance(first, second):
    return math.sqrt((first[0] - second[0])**2 + (first[1] - second[1])**2)

def run_pso(to_remove):
    pso = PSO()
    pso.init_population(to_remove)
    histories = []
    start_pos = []
    occurences = {}
    to_remove = []
    for i in range(params.popsize):
        occurences[i] = 0
    steps = 0
    stepsWithoutImprove = 0
    last_global_best = np.array([1000,1000])
    current_global_best = np.array([0,0])
    while steps < 30 and stepsWithoutImprove < 6:
        last_global_best = current_global_best
        current_global_best = pso.update_step()
        if distance(last_global_best, current_global_best) < 0.0000001:
            distance(last_global_best, current_global_best)
            stepsWithoutImprove += 1
        else:
            stepsWithoutImprove = 0
        steps += 1
    print("steps = ", steps)
    for part in pso.particles:
        histories.append(part.history)
        start_pos.append(part.history[0].pos)
    for hist in histories:
        for entry in hist:
            if entry.best_idx in occurences:
                occurences[entry.best_idx] = occurences[entry.best_idx]+1
            else:
                occurences.update({entry.best_idx: 1})
    #np.save("saves/start_pos", start_pos)
    for i in range(len(occurences)):
        if occurences[i] == 0:
            to_remove.append(i)
    if debug:
        print(f"{pso.global_best_pos} with {pso.global_best}")
        print(occurences)
        highest_occur = max(occurences, key=occurences.get)
        best_part = pso.particles[pso.global_best_index].history
        #print(pso.particles[highest_occur].history[0].pos)
        print("overall steps = ", pso.overall_steps)
        print("--------")
        return to_remove

if __name__ == '__main__':
    pso = PSO()
    INTERP_FRAMES = 20
    debug = True
    graphical = not debug
    to_remove = run_pso([])
    run_pso(to_remove)


    if graphical:
        pso = PSO()
        debug = False
        pso.init_population([])
        pygame.init()
        WIDTH = 800
        HEIGHT = 600
        frame = 0
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        clock = pygame.time.Clock()

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            t = frame / INTERP_FRAMES   # between 0 and 1
            screen.fill((0,0,0))
            pso.update_step()
            for p in pso.particles:
                x = p.prev_pos[0] * (1 - t) + p.next_pos[0] * t
                y = p.prev_pos[1] * (1 - t) + p.next_pos[1] * t
                pygame.draw.circle(screen, (0, 255, 255), (int(p.pos[0])/10*WIDTH + WIDTH/2, int(p.pos[1])/10*HEIGHT + HEIGHT/2 ), 4)
            pygame.draw.circle(screen, (255, 0, 0), (WIDTH/2,HEIGHT/2), 6)
            pygame.display.flip()
            frame += 1
            clock.tick(30)
            if frame > INTERP_FRAMES:
                frame = 0
        pygame.quit()
