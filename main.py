from particle import Particle
import numpy as np
from pso import PSO
from os import environ
import math
from parameters import params
import pandas as pd
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
environ['PYGAME_DETECT_AVX2'] = '1'

import pygame
MAX_TO_REMOVE = 3
def distance(first, second):
    return math.sqrt((first[0] - second[0])**2 + (first[1] - second[1])**2)

def start_pos_analysis(occurences, to_remove, histories, start_pos):
    for i in range(len(start_pos)):
        if i not in to_remove:
            print(f"{start_pos[i]} with ({histories[i][0].fitness})")
    print("------------")
    for i in range(len(start_pos)):
        if i in to_remove:
            print(f"part {i} {start_pos[i]} with ({histories[i][0].fitness})")

def end_pos_analysis(occurences, to_remove, histories, end_pos):
    for i in range(len(end_pos)):
        print(f"part {i} {end_pos[i]} with ({histories[i][-1].fitness})")

def run_pso(to_remove, first_run):
    pso = PSO()
    pso.init_population(to_remove)
    histories  = []
    start_pos  = []
    end_pos    = []
    occurences = {}
    to_remove  = []
    for i in range(params.popsize):
        occurences[i] = 0
    steps = 0
    stepsWithoutImprove = 0
    last_global_best = np.array([1000,1000])
    current_global_best = np.array([0,0])
    while steps < 30:# and stepsWithoutImprove < 6:
        #impact aus start-position herausfindbar
        #impact mit finaler population (part mit bester fitness der mit meisten einfluss)
        # gruppenbildung(später) 
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
        end_pos.append(part.history[-1].pos)
    for hist in histories:
        for entry in hist:
            if entry.best_idx in occurences:
                occurences[entry.best_idx] = occurences[entry.best_idx]+1
            else:
                occurences.update({entry.best_idx: 1})
    if first_run and debug:
        for i in range(len(occurences)):
            if occurences[i] == 0 and len(to_remove) < MAX_TO_REMOVE:
                to_remove.append(i)
        #start_pos_analysis(occurences, to_remove, histories, start_pos)
        np.save("saves/start_pos", start_pos)
        #end_pos_analysis(occurences, to_remove, histories, end_pos)

    if debug:
        print(f"{pso.global_best_pos} with {pso.global_best} and index {pso.global_best_index}")
        print(occurences)
        highest_occur = max(occurences, key=occurences.get)
        best_part = pso.particles[pso.global_best_index].history
        #print(pso.particles[highest_occur].history[0].pos)
        entries = []
        for i in pso.history:
            if i.global_change:
                entry = [i.gen, i.global_change, i.local_change, i.fitness, i.global_best_ratio, i.fitness_ratio, i.global_improvement]
                entries.append(entry)
            #print(f"gen {i.gen} g = {i.global_change}, l = {i.local_change}, fitness = {i.fitness}, ratio = {i.global_best_ratio}, fitness_change = {i.fitness_ratio}")
        df = pd.DataFrame(entries, columns=["Gen", "Global", "Local", "Mean Fitness", "best ratio", "Overall Ratio", "Improvement to beginning"])
        print(df)
        print("overall steps = ", pso.overall_steps)
        print("-------------------------------------------------------------------------")
    return to_remove

if __name__ == '__main__':
    pso = PSO()
    INTERP_FRAMES = 20
    debug = True
    graphical = False
    to_remove = run_pso([], True)
    run_pso(to_remove, False)


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
                pygame.draw.circle(screen, (0, 255, 255), (p.pos[0]/10*WIDTH + WIDTH/2, p.pos[1]/10*HEIGHT + HEIGHT/2 ), 4)
            pygame.draw.circle(screen, (255, 0, 0), (1/10*WIDTH + WIDTH/2, 1/10*HEIGHT + HEIGHT/2), 6)
            pygame.display.flip()
            frame += 1
            clock.tick(5)
            if frame > INTERP_FRAMES:
                frame = 0
        pygame.quit()

