from particle import Particle
import numpy as np
from pso import PSO
from os import environ
import math
from parameters import params
import pandas as pd 
import matplotlib.pyplot as plt 

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
    occurences1 = {}
    fitness_improvements = {}
    norm_fitness_improvs = {}
    all_fitness_improvs = []
    ranks = {}
    to_remove  = []
    for i in range(params.popsize):
        occurences[i] = 0
        occurences1[i] = 0
    steps = 0
    stepsWithoutImprove = 0
    last_global_best = np.array([1000,1000])
    current_global_best = np.array([0,0])
    while steps < params.iterations:#and stepsWithoutImprove < 6:
        #impact aus start-position herausfindbar
        #impact mit finaler population (part mit bester fitness der mit meisten einfluss)
        # gruppenbildung(später) 
        last_global_best = current_global_best
        current_global_best = pso.update_step()
        if distance(last_global_best, current_global_best) < 0.0000001:
            stepsWithoutImprove += 1
        else:
            stepsWithoutImprove = 0
        steps += 1
    print("steps = ", steps)
    for part in pso.particles:
        histories.append(part.history)
        start_pos.append(part.history[0].pos)
        end_pos.append(part.history[-1].pos)

    for i in range(params.iterations):
        fitness_particles = {}
        for part in pso.particles:
            if len(part.history) > 2:
                history = part.history[i]

                fitness_particles[part.index] = history.fitness
                if history.best_idx in occurences1:
                    occurences1[history.best_idx] = occurences1[history.best_idx]+1
                else:
                    occurences1.update({history.best_idx: 1})

                if history.best_idx in fitness_improvements:
                    fitness_improvements[history.best_idx] = fitness_improvements[history.best_idx] + history.fitness_improvement * 0.5
                else:
                    fitness_improvements.update({history.best_idx: history.fitness_improvement * 0.5})
                if part.index in fitness_improvements:
                    fitness_improvements[part.index] = fitness_improvements[part.index] + history.fitness_improvement * 0.5
                else:
                    fitness_improvements.update({part.index: history.fitness_improvement * 0.5})


        sorted_particles = dict(sorted(fitness_particles.items(), key=lambda item: item[1]))
        rank = 1
        for x in sorted_particles.keys():
            if x not in ranks:
                ranks.update({x: rank})
            else:
                ranks[x] += rank
            rank += 1

        total_improvement = sum(fitness_improvements.values())
        if total_improvement > 0:
            for key in fitness_improvements.keys():
                norm_fitness_improvs[key] = float(fitness_improvements[key] / total_improvement)
        all_fitness_improvs.append(norm_fitness_improvs)
        norm_fitness_improvs = {}

    sorted_ranks = dict(sorted(ranks.items(), key=lambda item: item[1]))

    if first_run and debug:
        for i in range(len(occurences)):
            if occurences[i] == 0 and len(to_remove) < MAX_TO_REMOVE:
                to_remove.append(i)
        #start_pos_analysis(occurences, to_remove, histories, start_pos)
        np.save("saves/start_pos", start_pos)
        #end_pos_analysis(occurences, to_remove, histories, end_pos)
    total_improvement = sum(fitness_improvements.values())
    for key in fitness_improvements.keys():
        norm_fitness_improvs[key] = float(fitness_improvements[key] / total_improvement)
    if debug:
        print(f"{pso.global_best_pos} with {pso.global_best} and index {pso.global_best_index}")
        #print(occurences1)
        print(norm_fitness_improvs)
        print(sorted_ranks)
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
        #print(df)
        print("overall steps = ", pso.overall_steps)
        print("-------------------------------------------------------------------------")
    return to_remove, pso

if __name__ == '__main__':
    debug = True
    graphical = True
    to_remove, pso = run_pso([], True)
    #run_pso(to_remove, False)

    if graphical:
        for i in range(params.iterations):
            x_pos = []
            y_pos = []
            for part in pso.particles:
                current = part.history[i].pos
                x_pos.append(current[0])
                y_pos.append(current[1])
            plt.clf()
            plt.scatter(x_pos, y_pos)
            plt.title(f"Generation {i}")
            plt.xlim(min(x_pos),max(x_pos))
            plt.ylim(min(y_pos),max(y_pos))
            plt.show()
            input("Press Enter for next generation...")