from particle import Particle
import numpy as np
from pso import PSO
from os import environ
import math
from parameters import params
import pandas as pd 
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors


MAX_TO_REMOVE = 3
def distance(first, second):
    dist = 0
    for i in range(params.dimensions):
        dist = dist + ((first[i] - second[i])**2)
    return math.sqrt(dist)

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

def range_conversion(dictionary):
    oldMin = min(dictionary.values())
    oldMax = max(dictionary.values())
    newMin = -1
    newMax = 1
    oldRange = (oldMax - oldMin)
    newRange = (newMax - newMin)
    for key in dictionary.keys():
        dictionary[key] = (((dictionary[key]- oldMin) * newRange) / oldRange) + newMin

def run_pso(to_remove, first_run):
    pso = PSO()
    pso.init_population(to_remove)
    histories  = []
    start_pos  = []
    end_pos    = []
    occurences = {}
    occurences1 = {}
    fitness_improvements = {}
    fitness_improvements1 = {}
    norm_fitness_improvs = {}
    all_fitness_improvs = []
    all_ranks = []
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
    for part in pso.particles:
        histories.append(part.history)
        start_pos.append(part.history[0].pos)
        end_pos.append(part.history[-1].pos)

    for i in range(params.iterations):
        fitness_particles = {}
        current_improvs = {}
        for part in pso.particles:
            if len(part.history) > 2:
                history = part.history[i]
                fitness_particles[part.index] = history.fitness
                if history.best_idx in occurences1:
                    occurences1[history.best_idx] = occurences1[history.best_idx]+1
                else:
                    occurences1.update({history.best_idx: 1})

                if history.best_idx in fitness_improvements:
                    fitness_improvements[history.best_idx] = fitness_improvements[history.best_idx] + history.fitness_improvement * params.c2
                    fitness_improvements1[history.best_idx] = (fitness_improvements1[history.best_idx] * 0.4) + history.fitness_improvement * params.c2
                else:
                    fitness_improvements.update({history.best_idx: history.fitness_improvement * params.c2})
                    fitness_improvements1.update({history.best_idx: history.fitness_improvement * params.c2})
                if part.index in fitness_improvements:
                    fitness_improvements[part.index] = (fitness_improvements[part.index]) + history.fitness_improvement * params.c1
                    fitness_improvements1[part.index] = (fitness_improvements1[part.index] * 0.4) + history.fitness_improvement * params.c1
                else:
                    fitness_improvements.update({part.index: history.fitness_improvement * params.c1})
                    fitness_improvements1.update({part.index: history.fitness_improvement * params.c1})
                
                if history.best_idx in current_improvs:
                    current_improvs[history.best_idx] = current_improvs[history.best_idx] + history.fitness_improvement * params.c2
                else:
                    current_improvs.update({history.best_idx: history.fitness_improvement*params.c2})
                if part.index in current_improvs:
                    current_improvs[part.index] = current_improvs[part.index] + history.fitness_improvement * params.c1
                else:
                    current_improvs.update({part.index: history.fitness_improvement * params.c1})
        if i > 0:
            total_improvement = sum(map(abs,current_improvs.values()))
            #range_conversion(current_improvs)
            for key in current_improvs.keys():
                current_improvs[key] = current_improvs[key] / total_improvement
        all_fitness_improvs.append(current_improvs)
        sorted_particles = dict(sorted(fitness_particles.items(), key=lambda item: item[1]))
        rank = 1
        temp_ranks = {}
        for x in sorted_particles.keys():
            if x not in ranks:
                ranks.update({x: rank})
            else:
                ranks[x] += rank
            temp_ranks.update({x: rank})
            rank += 1
        all_ranks.append(temp_ranks)
        norm_fitness_improvs = {}
        norm_fitness_improvs1 = {}

    sorted_ranks = dict(sorted(ranks.items(), key=lambda item: item[1]))

    if first_run and debug:
        for i in range(len(occurences)):
            if occurences[i] == 0 and len(to_remove) < MAX_TO_REMOVE:
                to_remove.append(i)
        #start_pos_analysis(occurences, to_remove, histories, start_pos)
        np.save("saves/start_pos", start_pos)
        #end_pos_analysis(occurences, to_remove, histories, end_pos)
    total_improvement = sum(map(abs,fitness_improvements.values()))
    total_improvement1 = sum(map(abs,fitness_improvements1.values()))
    for key in fitness_improvements.keys():
        norm_fitness_improvs[key] = float(fitness_improvements[key] / total_improvement)
        norm_fitness_improvs1[key] = float(fitness_improvements1[key] / total_improvement1)

    sorted_norm_improvs = dict(sorted(norm_fitness_improvs.items(), key=lambda item: item[1]))
    sorted_norm_improvs1 = dict(sorted(norm_fitness_improvs1.items(), key=lambda item: item[1]))
    
    if debug:
        print(f"{pso.global_best_pos} with {pso.global_best} and index {pso.global_best_index}")
        #print(occurences1)
        print(sorted_norm_improvs)
        print(sorted_norm_improvs1)
        #print(sorted_ranks)
        highest_occur = max(occurences, key=occurences.get)
        best_part = pso.particles[pso.global_best_index].history
        best_improv = list(sorted_norm_improvs)[-1]
        #print(pso.particles[highest_occur].history[0].pos)
        entries = []
        for i in pso.history:
            if i.global_change:
                entry = [i.gen, i.global_change, i.local_change, i.fitness, i.global_best_ratio, i.fitness_ratio, i.global_improvement]
                entries.append(entry)
            #print(f"gen {i.gen} g = {i.global_change}, l = {i.local_change}, fitness = {i.fitness}, ratio = {i.global_best_ratio}, fitness_change = {i.fitness_ratio}")
        df = pd.DataFrame(entries, columns=["Gen", "Global", "Local", "Mean Fitness", "best ratio", "Overall Ratio", "Improvement to beginning"])
        #print(df)
        #print("overall steps = ", pso.overall_steps)
        print("-------------------------------------------------------------------------")
    return to_remove, pso, all_fitness_improvs, all_ranks, best_improv

if __name__ == '__main__':
    debug = True
    graphical = False
    #kolineare partikel -> testen mit 1D partikel sollten sich immer wieder überholen
    #impact anpassen -> verringern mit zunehmender generation
    #bestes anfangspartikel mal rausnehmen aus evaluation
    #impact (alle anzeigen aber unwichtige grau)

    to_remove, pso, all_fitness_improvs, all_ranks, best_improvs = run_pso([], True)
    improvs_array = {}
    global_rank_history = []
    for key in all_fitness_improvs[1].keys():
        improvs_array.update({key: []})
    #run_pso(to_remove, False)
    for i in range(params.iterations):
        current_improvs = all_fitness_improvs[i]
        current_ranks = all_ranks[i]
        for key in current_improvs.keys():
            improvs_array[key].append(current_improvs[key])
        global_rank_history.append(current_ranks[pso.global_best_index])

    x = range(params.iterations)
    if not graphical:
        fig, ax = plt.subplots()
        grey_color = "#00000010"
        for key in improvs_array.keys():
            if key != best_improvs and key != pso.global_best_index:
                plt.plot(x, improvs_array[key], color=grey_color)
        ax.plot(x,improvs_array[best_improvs], label = "improvs")
        ax.plot(x, improvs_array[pso.global_best_index], label="global")
        ax.set_ylim(-0.75,0.75)
        #plt.plot(x, global_rank_history, label="global_rank")
        ax.legend()
        plt.show()

    if graphical:
        plt.ion()
        fig, ax = plt.subplots()
        norm = mcolors.Normalize(vmin=-0.1, vmax=0.4)
        for i in range(params.iterations):
            x_pos = []
            y_pos = []
            improvs = []
            ax.clear()
            for part in pso.particles:
                current = part.history[i].pos
                best_idx = part.history[i].best_idx
                x_pos.append(current[0])
                #print(current[0])
                if len(current) > 1:
                    y_pos.append(current[1])
                else:
                    y_pos.append(part.index)
                current_improvs = all_fitness_improvs[i]
                if len(current) > 1:
                    ax.text(current[0], current[1], str(part.index), fontsize=9)
                else:
                    ax.text(current[0], part.index, str(part.index), fontsize=9)
                if part.index in current_improvs:
                    improvs.append(current_improvs[part.index])
                else:
                    improvs.append(0)
            ax.scatter(x_pos, y_pos, c=improvs, cmap="RdYlGn", norm=norm)
            if len(current) > 1:
                ax.scatter(pso.global_best_pos[0], pso.global_best_pos[1], color="blue")
            ax.set_title(f"Generation {i}")
            ax.set_xlim(-10,10)
            ax.set_ylim(-10,10)
            plt.draw()
            input("Press Enter for next generation...")
        plt.ioff()
        plt.show()