from particle import Particle
import numpy as np
from pso import PSO
from pso_ring import PSO_Ring
from os import environ
import math
from parameters import params
import pandas as pd 
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
import matplotlib.animation as animation
import seaborn as sns
from evaluation import Evaluation

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

def run_pso(to_remove, first_run, using_ring):
    pso = None
    if using_ring:
        pso = PSO_Ring()
    else:
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
    all_fitness_improvs1 = []
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
    shrink_ratio = 0.8
    for i in range(params.iterations):
        fitness_particles = {}
        current_improvs = {}
        current_improvs1 = {}
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
                    fitness_improvements1[history.best_idx] = (fitness_improvements1[history.best_idx] * shrink_ratio) + history.fitness_improvement * params.c2
                else:
                    fitness_improvements.update({history.best_idx: history.fitness_improvement * params.c2})
                    fitness_improvements1.update({history.best_idx: history.fitness_improvement * params.c2})
                if part.index in fitness_improvements:
                    fitness_improvements[part.index] = (fitness_improvements[part.index]) + history.fitness_improvement * params.c1
                    fitness_improvements1[part.index] = (fitness_improvements1[part.index] * shrink_ratio) + history.fitness_improvement * params.c1
                else:
                    fitness_improvements.update({part.index: history.fitness_improvement * params.c1})
                    fitness_improvements1.update({part.index: history.fitness_improvement * params.c1})
                
                if history.best_idx in current_improvs:
                    current_improvs[history.best_idx] = current_improvs[history.best_idx] + history.fitness_improvement * params.c2
                    current_improvs1[history.best_idx] = current_improvs1[history.best_idx] * shrink_ratio + history.fitness_improvement * params.c2
                else:
                    current_improvs.update({history.best_idx: history.fitness_improvement*params.c2})
                    current_improvs1.update({history.best_idx: history.fitness_improvement*params.c2})
                if part.index in current_improvs:
                    current_improvs[part.index] = current_improvs[part.index] + history.fitness_improvement * params.c1
                    current_improvs1[part.index] = current_improvs1[part.index] * shrink_ratio + history.fitness_improvement * params.c1
                else:
                    current_improvs.update({part.index: history.fitness_improvement * params.c1})
                    current_improvs1.update({part.index: history.fitness_improvement * params.c1})
        if i > 0:
            total_improvement = sum(map(abs,current_improvs.values()))
            total_improvement1 = sum(map(abs,current_improvs1.values()))
            #range_conversion(current_improvs)
            for key in current_improvs.keys():
                if total_improvement != 0:
                    current_improvs[key] = current_improvs[key] / total_improvement
                if total_improvement1 != 0:
                    current_improvs1[key] = current_improvs1[key] / total_improvement1
        all_fitness_improvs.append(current_improvs)
        all_fitness_improvs1.append(current_improvs1)
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
        #print("-------------------------------")
        #print(sorted_norm_improvs)
        #print("---------------")
        #print("========= with decreasing =========")
        #print(sorted_norm_improvs1)
        #print(sorted_ranks)
        highest_occur = max(occurences, key=occurences.get)
        best_part = pso.particles[pso.global_best_index].history
        best_improv = list(sorted_norm_improvs)[-1]
        best_improv1 = list(sorted_norm_improvs1)[-1]
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
        #print("-------------------------------------------------------------------------")
    return to_remove, pso, all_fitness_improvs, all_ranks, best_improv, best_improv1, all_fitness_improvs1, sorted_norm_improvs, sorted_norm_improvs1

if __name__ == '__main__':
    debug = True
    graphical = not True

    to_remove, pso, all_fitness_improvs, all_ranks, best_improvs, best_improvs1, all_fitness_improvs1, sorted_norm_improvs, sorted_norm_improvs1 = run_pso([], True, False)
    improvs_array = {}
    improvs_array1 = {}
    ranks_array = {}
    global_rank_history = []
    global_fitness_history = []
    shared_best = {}
    for key in all_fitness_improvs[1].keys():
        improvs_array.update({key: []})
        improvs_array1.update({key: []})
        ranks_array.update({key: []})
    #run_pso(to_remove, False)
    for i in range(params.iterations):
        current_improvs = all_fitness_improvs[i]
        current_improvs1 = all_fitness_improvs1[i]
        current_ranks = all_ranks[i]
        global_fitness_history.append(pso.history[i].global_fitness)
        for key in current_improvs.keys():
            improvs_array[key].append(current_improvs[key])
            improvs_array1[key].append(current_improvs1[key])
            ranks_array[key].append(current_ranks[key])
        global_rank_history.append(current_ranks[pso.global_best_index])
    
        
    to_remove_ring, pso_ring, all_fitness_improvs_ring, all_ranks_ring, best_improvs_ring, best_improvs1_ring, all_fitness_improvs1_ring, sorted_norm_improvs_ring, sorted_norm_improvs1_ring = run_pso([], True, True)
    improvs_array_ring = {}
    improvs_array_ring1 = {}
    ranks_array_ring = {}
    global_rank_history_ring = []
    global_fitness_history_ring = []
    shared_best_ring = {}
    for key in all_fitness_improvs_ring[1].keys():
        improvs_array_ring.update({key: []})
        improvs_array_ring1.update({key: []})
        ranks_array_ring.update({key: []})
    #run_pso(to_remove, False)
    for i in range(params.iterations):
        current_improvs_ring = all_fitness_improvs_ring[i]
        current_improvs_ring1 = all_fitness_improvs1_ring[i]
        current_ranks_ring = all_ranks_ring[i]
        global_fitness_history_ring.append(pso_ring.history[i].global_fitness)
        for key in current_improvs_ring.keys():
            improvs_array_ring[key].append(current_improvs_ring[key])
            improvs_array_ring1[key].append(current_improvs_ring1[key])
            ranks_array_ring[key].append(current_ranks_ring[key])
        global_rank_history_ring.append(current_ranks_ring[pso_ring.global_best_index])
    
    for i in pso_ring.particles:
        shared_best_ring[i.index] = i.shared_best
    if not graphical:
        data_ranks = pd.DataFrame.from_dict(ranks_array)
        data_ranks_ring = pd.DataFrame.from_dict(ranks_array_ring)
        data_improvs = pd.DataFrame.from_dict(improvs_array)
        data_improvs1 = pd.DataFrame.from_dict(improvs_array1)
        data_improvs_ring = pd.DataFrame.from_dict(improvs_array_ring)
        data_improvs1_ring = pd.DataFrame.from_dict(improvs_array_ring1)
        standard_pal = sns.color_palette()
        sns.set_theme(style="darkgrid") 
        #plt.show()
        Evaluation.evaluate_global_fitness(global_fitness_history, global_fitness_history_ring, pso, pso_ring)
        plt.savefig(f"fitness{params.w}_{params.c1}_{params.c2}.png")
        Evaluation.evaluate_norm_improvs(sorted_norm_improvs1, sorted_norm_improvs1_ring, sorted_norm_improvs, sorted_norm_improvs_ring, pso, pso_ring)
        plt.savefig(f"improvs_{params.w}_{params.c1}_{params.c2}.png")
        Evaluation.evaluate_ranks(data_ranks, data_ranks_ring, best_improvs, best_improvs_ring, pso, pso_ring)
        plt.savefig(f"ranks_{params.w}_{params.c1}_{params.c2}.png")
        Evaluation.evaluate_improvs(data_improvs, data_improvs_ring, data_improvs1, data_improvs1_ring, best_improvs, best_improvs_ring, best_improvs1, best_improvs1_ring, pso, pso_ring)
        plt.savefig(f"normed_improvs_{params.w}_{params.c1}_{params.c2}.png")
        #plt.show()

    if graphical:
        #plt.ion()
        fig, ax = plt.subplots()
        norm = mcolors.Normalize(vmin=-0.1, vmax=0.4)
        ax.set_xlim(-10,10)
        ax.set_ylim(-10,10)
        ax.clear()
        '''for i in range(params.iterations):
            x_pos = []
            y_pos = []
            improvs = []
            plt.draw()
            update(i)
            input("Press Enter for next generation...")'''

        def update(i):
            ax.clear()
            x_pos = []
            y_pos = []
            improvs = []
            for part in pso.particles:
                current = part.history[i].pos
                best_idx = part.history[i].best_idx
                x_pos.append(current[0])
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
            ax.set_xlim(-10,10)
            ax.set_ylim(-10,10)
            ax.scatter(x_pos, y_pos, c=improvs, cmap="RdYlGn", norm=norm)
            ax.set_title(f"Generation {i}")
            if len(current) > 1:
                #ax.scatter(pso.global_best_pos[0], pso.global_best_pos[1], color="blue")
                ax.scatter(1,1, color="blue")


        ani = animation.FuncAnimation(fig, update, frames=params.iterations, interval=500, repeat=False)
        save_name = f"vis_{params.w}_{params.c1}_{params.c2}.mp4"
        ani.save(save_name, writer="ffmpeg", dpi=100)
        #plt.ioff()
        plt.show()