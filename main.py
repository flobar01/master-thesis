from particle import Particle
import numpy as np
from pso import PSO
from pso_ring import PSO_Ring
from os import environ
import os
import math
from parameters import params
import pandas as pd 
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
import matplotlib.animation as animation
import seaborn as sns
from evaluation import Evaluation
import random
import pickle

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

def flatten_improvs_dict(all_fitness_improvs):
    all_vals = []

    for generation in all_fitness_improvs:
        all_vals.extend(generation.values())

    all_vals = np.array(all_vals)

    cap = np.percentile(all_vals, 99)

    processed = []

    for generation in all_fitness_improvs:

        processed_generation = {
            k: np.sign(v) * np.log1p(abs(min(v, cap)))
            for k, v in generation.items()
        }

        processed.append(processed_generation)

    return processed

def run_pso(to_remove, first_run, using_ring, start_poses=[], run_number=0):
    pso = None
    if using_ring:
        pso = PSO_Ring()
    else:
        pso = PSO()
    pso.init_population(to_remove, start_poses)
    histories  = []
    start_pos  = []
    end_pos    = []
    occurences = {}
    occurences1 = {}
    fitness_improvements = {part.index: 0 for part in pso.particles}
    fitness_improvements1 = {part.index: 0 for part in pso.particles}

    norm_fitness_improvs = {}
    all_fitness_improvs = []
    all_fitness_improvs1 = []
    all_ranks = []
    ranks = {}
    to_remove  = []
    positions_start = []
    weight_social = 0.7
    weight_cognitive = 0.3
    for p in pso.particles:
        positions_start.append(p.pos)
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
    shrink_ratio = 0.6
    for i in range(params.iterations):
        fitness_particles = {}

        current_improvs = {part.index: 0 for part in pso.particles}
        current_improvs1 = {part.index: 0 for part in pso.particles}
        for key in fitness_improvements1.keys():
            fitness_improvements1[key] = fitness_improvements1[key] * shrink_ratio

        for part in pso.particles:
            if len(part.history) > 2:
                history = part.history[i]
                history_best = pso.particles[history.best_idx].history[i]
                fitness_particles[part.index] = history.fitness
                if history.best_idx in occurences1:
                    occurences1[history.best_idx] = occurences1[history.best_idx]+1
                else:
                    occurences1.update({history.best_idx: 1})


                fitness_improvements[history.best_idx] = fitness_improvements[history.best_idx] + (history.fitness_improvement / params.popsize) * params.c2
                fitness_improvements1[history.best_idx] = (fitness_improvements1[history.best_idx]) + (history.fitness_improvement / params.popsize) * params.c2
                fitness_improvements[part.index] = (fitness_improvements[part.index]) + history.fitness_improvement * params.c1
                fitness_improvements1[part.index] = (fitness_improvements1[part.index]) + history.fitness_improvement * params.c1
                
                current_improvs[history.best_idx] = current_improvs[history.best_idx] + (history.fitness_improvement / params.popsize) * params.c2
                current_improvs1[history.best_idx] = current_improvs1[history.best_idx] + (history.fitness_improvement / params.popsize) * params.c2
                current_improvs[part.index] = current_improvs[part.index] + history.fitness_improvement * params.c1
                current_improvs1[part.index] = current_improvs1[part.index] + history.fitness_improvement * params.c1
        
        if i > 0:
            total_improvement = sum(map(abs,current_improvs.values()))
            total_improvement1 = sum(map(abs,current_improvs1.values()))
            #range_conversion(current_improvs)
            for key in current_improvs.keys():
                if total_improvement != 0:
                    current_improvs[key] = current_improvs[key]
                if total_improvement1 != 0:
                    current_improvs1[key] = current_improvs1[key]
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
        np.save(f"saves/start_pos_{run_number}", start_pos)
        #end_pos_analysis(occurences, to_remove, histories, end_pos)
    
    processed = flatten_improvs_dict(all_fitness_improvs)
    processed1 = flatten_improvs_dict(all_fitness_improvs1)
    total_improvement = sum(map(abs,fitness_improvements.values()))
    total_improvement1 = sum(map(abs,fitness_improvements1.values()))



    for key in fitness_improvements.keys():
        norm_fitness_improvs[key] = float(fitness_improvements[key] / total_improvement)
        norm_fitness_improvs1[key] = float(fitness_improvements1[key] / total_improvement1)

    sorted_norm_improvs = dict(sorted(norm_fitness_improvs.items(), key=lambda item: item[1]))
    sorted_norm_improvs1 = dict(sorted(norm_fitness_improvs1.items(), key=lambda item: item[1]))
    
    if debug:
        print(f"{pso.global_best_pos} with {pso.global_best} and index {pso.global_best_index}")
        #print(total_portions)
        #print(occurences1)
        print("-------------------------------")
        #print(sorted_norm_improvs)
        print(pso.particles[pso.global_best_index].portion)
        #print("---------------")
        #print("========= with decreasing =========")
        #print(sorted_norm_improvs1)
        #print(sorted_ranks)
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
    highest_occur = max(occurences, key=occurences.get)
    best_part = pso.particles[pso.global_best_index].history
    best_improv = list(sorted_norm_improvs)[-1]
    best_improv1 = list(sorted_norm_improvs1)[-1]
    return to_remove, pso, processed, all_ranks, best_improv, best_improv1, processed1, sorted_norm_improvs, sorted_norm_improvs1, positions_start

if __name__ == '__main__':
    debug = True
    runs = 0
    graphical = not True
    value_sets = [[0.5, 1.4, 1.4], [0.5, 1.9, 1.9], [0.7, 1.4, 1.4], [0.7, 1.9, 1.9]]
    test_starting_pos = []
    for i in range(31):
        run = []
        for j in range(params.popsize):
            run.append(np.array([random.randint(-params.bounds, params.bounds), random.randint(-params.bounds, params.bounds)]))
        test_starting_pos.append(run)
    acc_improvs_gbest = []
    acc_improvs_lbest = []
    dec_improvs_gbest = []
    dec_improvs_lbest = []

    best_fitness_gbest = []
    best_fitness_lbest = []
    fitness_data = {
        "value_set":     [],
        "GBest": [],
        "LBest": [],
        "run":           []
    }
    '''for i in range(len(value_sets)):
        values = value_sets[i]
        params.w = values[0]
        params.c1 = values[1]
        params.c2 = values[2]
        for run in range(31):
            to_remove, pso, all_fitness_improvs, all_ranks, best_improvs, best_improvs1, all_fitness_improvs1, sorted_norm_improvs, sorted_norm_improvs1, positions_start = run_pso([], True, False, run_number=runs, start_poses=test_starting_pos[run])
            to_remove_ring, pso_ring, all_fitness_improvs_ring, all_ranks_ring, best_improvs_ring, best_improvs1_ring, all_fitness_improvs1_ring, sorted_norm_improvs_ring, sorted_norm_improvs1_ring, positions_start = run_pso([], False, True, start_poses=test_starting_pos[run])
            if i == 0:
                fitness_data["value_set"].append(f"Low Inertia \n Good Learning")
            if i == 1:
                fitness_data["value_set"].append(f"Low Inertia \n High Learning")
            if i == 2:
                fitness_data["value_set"].append(f"High Inertia \n Good Learning")
            if i == 3:
                fitness_data["value_set"].append(f"High Inertia \n High Learning")
            fitness_data["GBest"].append(pso.global_best)
            fitness_data["LBest"].append(pso_ring.global_best)
            fitness_data["run"].append(run)
    df_fitness =  pd.DataFrame(fitness_data)
    df_long = df_fitness.melt(
        id_vars=["value_set", "run"],
        value_vars=["GBest", "LBest"],
        var_name="Implementation",
        value_name="fitness"
    )
    sns.set_theme(style="ticks", palette="pastel")
    ax = sns.boxplot(x="value_set", y="fitness", hue="Implementation", data=df_long, log_scale=True)
    plt.xlabel("Parameter Set")
    plt.ylabel("Best Fitness")
    plt.tight_layout()
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.savefig(f"fitness_box.png", dpi=300, bbox_inches='tight')
    plt.show()'''
    while runs < 31:
        values = value_sets[2]
        params.w = values[0]
        params.c1 = values[1]
        params.c2 = values[2]
        start_positions = np.load(f"saves/start_pos_{runs}.npy")
        if start_positions is None:
            break
        to_remove, pso, all_fitness_improvs, all_ranks, best_improvs, best_improvs1, all_fitness_improvs1, sorted_norm_improvs, sorted_norm_improvs1, positions_start = run_pso([], True, False, run_number=runs, start_poses=start_positions)
        if pso.global_best > 1:
            continue
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
        
        
        to_remove_ring, pso_ring, all_fitness_improvs_ring, all_ranks_ring, best_improvs_ring, best_improvs1_ring, all_fitness_improvs1_ring, sorted_norm_improvs_ring, sorted_norm_improvs1_ring, positions_start = run_pso([], False, True, start_poses=positions_start)
        if pso_ring.global_best > 2:
            continue
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


        worst_starting = 0
        worst_starting_lbest = 0

        for idx in range(params.popsize):
            ranks = ranks_array[idx]
            if ranks[0] == 20:
                worst_starting = idx
                break

        for idx in range(params.popsize):
            ranks = ranks_array_ring[idx]
            if ranks[0] == 20:
                worst_starting_lbest = idx
                break

        best_fitness_gbest.append(pso.global_best)
        best_fitness_lbest.append(pso_ring.global_best)

        acc_improvs_gbest.append(sorted_norm_improvs[worst_starting])
        dec_improvs_gbest.append(sorted_norm_improvs1[worst_starting])
        
        acc_improvs_lbest.append(sorted_norm_improvs_ring[worst_starting_lbest])
        dec_improvs_lbest.append(sorted_norm_improvs1_ring[worst_starting_lbest])

    ####################################################################################################    
    ####################################################################################################    
    ####################################################################################################    
    ########################### _____           _   _                 ##################################
    ###########################|  __ \         | | (_)                ##################################
    ###########################| |__) |__  _ __| |_ _  ___  _ __  ___ ##################################
    ###########################|  ___/ _ \| '__| __| |/ _ \| '_ \/ __|##################################
    ###########################| |  | (_) | |  | |_| | (_) | | | \__ \##################################
    ###########################|_|   \___/|_|   \__|_|\___/|_| |_|___/##################################
    ####################################################################################################                     
    ####################################################################################################                     
    ####################################################################################################                     
                                        
        portions = {}
        for p in pso.particles:
            norm_portions = {}
            for p2 in pso.particles:
                particle_portion = p2.portion
                total_portion_amount = sum(map(abs,particle_portion.values()))
                norm_portions[r"$x_{{{}}}(t={})$".format(p2.index, params.iterations)] = particle_portion[p.index]
            portions[r"$x_{{{}}}(t=0)$".format(p.index)] = norm_portions
        #portions = {f"{part.index}":part.portion for part in pso.particles}
        portions_data = pd.DataFrame(data=portions)


        portions_ring = {}
        for p in pso_ring.particles:
            norm_portions = {}
            for p2 in pso_ring.particles:
                particle_portion = p2.portion
                total_portion_amount = sum(map(abs,particle_portion.values()))
                norm_portions[r"$x_{{{}}}(t={})$".format(p2.index, params.iterations)] = particle_portion[p.index]
            portions_ring[r"$x_{{{}}}(t=0)$".format(p.index)] = norm_portions

        portions_data_ring = pd.DataFrame(data=portions_ring)

        total_portions = {part.index: 0 for part in pso.particles}
        highest_portion_index = 0
        highest_portion = 0
        
        second_highest_portion_index = 0
        second_highest_portion = 0

        for p in pso.particles:
            portion = p.portion
            for key in portion.keys():
                total_portions[key] += portion[key]
        
        for key in total_portions.keys():
            if total_portions[key] > highest_portion:
                highest_portion_index = key
                highest_portion = total_portions[key]
            if total_portions[key] > second_highest_portion and key != highest_portion_index:
                second_highest_portion = total_portions[key]
                second_highest_portion_index = key

        all_portion_changes = []

        for i in range(params.iterations):
            current_portion_changes = {p.index: 0 for p in pso.particles}
            for p in pso.particles:
                current_portion = p.history[i].portion
                if i > 0:
                    last_portion = p.history[i-1].portion
                    for key in current_portion.keys():
                        current_portion_changes[key] = current_portion_changes[key] + (current_portion[key] - last_portion[key])
                else:
                    for key in current_portion.keys():
                        current_portion_changes[key] = current_portion_changes[key] + current_portion[key]
                        
            all_portion_changes.append(current_portion_changes)

        all_portion_changes_ring = []
        for i in range(params.iterations):
            current_portion_changes = {p.index: 0 for p in pso_ring.particles}
            for p in pso_ring.particles:
                current_portion = p.history[i].portion.copy()
                if i > 0:
                    last_portion = p.history[i-1].portion.copy()
                    for key in current_portion.keys():
                        current_portion_changes[key] = current_portion_changes[key] + (current_portion[key] - last_portion[key])
                else:
                    for key in current_portion.keys():
                        current_portion_changes[key] = current_portion_changes[key] + current_portion[key]
                        
            all_portion_changes_ring.append(current_portion_changes)

        
        total_portions_ring = {part.index: 0 for part in pso_ring.particles}
        highest_portion_ring = 0
        highest_portion_index_ring = 0

        second_highest_portion_ring = 0
        second_highest_portion_index_ring = 0

        for p in pso_ring.particles:
            portion = p.portion
            for key in portion.keys():
                total_portions_ring[key] += portion[key]


        for key in total_portions_ring.keys():
            if total_portions_ring[key] > highest_portion_ring:
                highest_portion_index_ring = key
                highest_portion_ring = total_portions_ring[key]
            if total_portions[key] > second_highest_portion_ring and key != highest_portion_index_ring:
                second_highest_portion_ring = total_portions[key]
                second_highest_portion_index_ring = key
        

        for i in pso_ring.particles:
            shared_best_ring[i.index] = i.shared_best
        if not graphical:
            data_ranks = pd.DataFrame.from_dict(ranks_array)
            data_ranks_ring = pd.DataFrame.from_dict(ranks_array_ring)
            data_improvs = pd.DataFrame.from_dict(improvs_array)
            data_improvs1 = pd.DataFrame.from_dict(improvs_array1)
            data_improvs_ring = pd.DataFrame.from_dict(improvs_array_ring)
            data_improvs1_ring = pd.DataFrame.from_dict(improvs_array_ring1)
            data_portions = pd.DataFrame.from_dict(all_portion_changes)
            data_portions_ring = pd.DataFrame.from_dict(all_portion_changes_ring)
            shared_data = {}
            for i in pso_ring.particles:
                shared_data[i.index] = i.shared_best

            saving = {"pso_global_best": pso.global_best_index, "pso_ring_global_best": pso_ring.global_best_index, "global_fitness_history": global_fitness_history, "global_fitness_history_ring": global_fitness_history_ring, 
                "sorted_norm_improvs": sorted_norm_improvs, "sorted_norm_improvs_ring": sorted_norm_improvs_ring, "sorted_norm_improvs1": sorted_norm_improvs1, 
                "sorted_norm_improvs1_ring": sorted_norm_improvs1_ring, "data_ranks": data_ranks, "data_ranks_ring": data_ranks_ring, "best_improvs": best_improvs, "best_improvs1": best_improvs1,
                "best_improvs_ring": best_improvs_ring, "best_improvs1_ring": best_improvs1_ring, "data_improvs": data_improvs, "data_improvs1": data_improvs1, "data_improvs_ring": data_improvs_ring,
                "data_improvs1_ring": data_improvs1_ring, "shared_best": shared_data, "data_portions": data_portions, "highest_portion_index": highest_portion_index, "highest_postion_index_ring": highest_portion_index_ring,
                "data_portions_ring": data_portions_ring, "second_highest_portion_index": second_highest_portion_index, "second_highest_portion_index_ring": second_highest_portion_index_ring, "portions_data": portions_data, "portions_data_ring": portions_data_ring}

            standard_pal = sns.color_palette()
            sns.set_theme(style="darkgrid")
            folder =  f"evaluation/run{runs+1}"
            #plt.show()


            if not os.path.exists("evaluation"):
                os.mkdir("evaluation")

            if not os.path.exists(folder):
                os.mkdir(f"evaluation/run{runs+1}")

            if not os.path.exists(f"{folder}/portions"):
                os.mkdir(f"{folder}/portions")
            
            with open(f"{folder}/data.pkl", "wb") as data_saving:
                pickle.dump(saving, data_saving, pickle.HIGHEST_PROTOCOL)
            
            with open(f"{folder}/end_result.txt", "w") as f:
                f.write(f"gbest: {pso.global_best_index} with {pso.global_best}\n")
                f.write(f"lbest: {pso_ring.global_best_index} with {pso_ring.global_best}")
            
            Evaluation.evaluate_portions_ot(data_portions, highest_portion_index, second_highest_portion_index, pso.global_best_index, True)
            plt.savefig(f"{folder}/portions/portions_ot.png", dpi=250, bbox_inches='tight')
            
            Evaluation.evaluate_portions_ot(data_portions_ring, highest_portion_index_ring, second_highest_portion_index_ring, pso_ring.global_best_index, False)
            plt.savefig(f"{folder}/portions/portions_ot_ring.png", dpi=250, bbox_inches='tight')
            
            Evaluation.evaluate_portions(total_portions)
            plt.savefig(f"{folder}/portions/portions.png", dpi=250)

            Evaluation.evaluate_portions_stacked(portions_data)
            plt.savefig(f"{folder}/portions/stacked_portions.png", dpi=250)
            
            Evaluation.evaluate_portions(total_portions_ring)
            plt.savefig(f"{folder}/portions/portions_ring.png", dpi=250)

            Evaluation.evaluate_portions_heatmap(portions_data)
            plt.savefig(f"{folder}/portions/heatmap.png", dpi=250)
            
            Evaluation.evaluate_portions_heatmap(portions_data_ring)
            plt.savefig(f"{folder}/portions/heatmap_ring.png", dpi=250)

            Evaluation.evaluate_portions_stacked(portions_data_ring)
            plt.savefig(f"{folder}/portions/stacked_portions_ring.png", dpi=250) 

            Evaluation.evaluate_global_fitness(global_fitness_history, global_fitness_history_ring)
            plt.savefig(f"{folder}/fitness.png", dpi=300)
            
            Evaluation.evaluate_norm_improvs(sorted_norm_improvs1, sorted_norm_improvs1_ring, sorted_norm_improvs, sorted_norm_improvs_ring, pso.global_best_index, pso_ring.global_best_index)
            plt.savefig(f"{folder}/improvs.png", dpi=300)
            
            Evaluation.evaluate_ranks(data_ranks, data_ranks_ring, best_improvs, best_improvs_ring, best_improvs1, best_improvs1_ring, pso.global_best_index, pso_ring.global_best_index)
            plt.savefig(f"{folder}/ranks.png", dpi=250)

            Evaluation.evaluate_improvs(data_improvs, data_improvs_ring, data_improvs1, data_improvs1_ring, best_improvs, best_improvs_ring, best_improvs1, best_improvs1_ring, pso.global_best_index, pso_ring.global_best_index)
            plt.savefig(f"{folder}/normed_improvs.png", dpi=300)

            Evaluation.evaluate_shared(shared_data, sorted_norm_improvs_ring, pso_ring.global_best_index)
            plt.savefig(f"{folder}/shared.png", dpi=250)

            plt.close('all')
        runs += 1

    median_for_gbest_acc = np.median(np.array(acc_improvs_gbest))
    median_for_lbest_acc = np.median(np.array(acc_improvs_lbest))
    median_for_gbest_dec = np.median(np.array(dec_improvs_gbest))
    median_for_lbest_dec = np.median(np.array(dec_improvs_lbest))

    median_for_gbest_fitness = np.median(np.array(best_fitness_gbest))
    median_for_lbest_fitness = np.median(np.array(best_fitness_lbest))
    
    index_median_gbest_acc = 0

    for i in range(len(acc_improvs_gbest)):
        if acc_improvs_gbest[i] == median_for_gbest_acc:
            index_median_gbest_acc = i+1

    index_median_gbest_dec = 0

    for i in range(len(dec_improvs_gbest)):
        if dec_improvs_gbest[i] == median_for_gbest_dec:
            index_median_gbest_dec = i+1
    
    index_median_lbest_acc = 0

    for i in range(len(acc_improvs_lbest)):
        if acc_improvs_lbest[i] == median_for_lbest_acc:
            index_median_lbest_acc = i+1

    index_median_lbest_dec = 0

    for i in range(len(dec_improvs_lbest)):
        if dec_improvs_lbest[i] == median_for_lbest_dec:
            index_median_lbest_dec = i+1
    
    index_median_gbest_fitness = 0

    for i in range(len(best_fitness_gbest)):
        if best_fitness_gbest[i] == median_for_gbest_fitness:
            index_median_gbest_fitness = i+1
    
    index_median_lbest_fitness = 0

    for i in range(len(best_fitness_lbest)):
        if best_fitness_lbest[i] == median_for_lbest_fitness:
            index_median_lbest_fitness = i+1

    with open("evaluation/results.txt", "w") as f:
        f.write(f"median gbest fitness: {index_median_gbest_fitness}\n")
        f.write(f"median lbest fitness: {index_median_lbest_fitness}")
    #    f.write(f"median gbest acc: {index_median_gbest_acc}\n")
    #    f.write(f"median gbest dec: {index_median_gbest_dec}\n")
    #    f.write(f"median lbest acc: {index_median_lbest_acc}\n")
    #    f.write(f"median lbest dec: {index_median_lbest_dec}\n")
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
            for part in pso_ring.particles:
                current = part.history[i].pos
                best_idx = part.history[i].best_idx
                x_pos.append(current[0])
                if len(current) > 1:
                    y_pos.append(current[1])
                else:
                    y_pos.append(part.index)
                current_improvs = all_fitness_improvs[i]
                if part.index == 13:
                    ax.text(current[0], current[1], str(part.index), fontsize=9)
                """if len(current) > 1:
                    ax.text(current[0], current[1], str(part.index), fontsize=9)
                else:
                    ax.text(current[0], part.index, str(part.index), fontsize=9)"""
                if part.index in current_improvs:
                    improvs.append(current_improvs[part.index])
                else:
                    improvs.append(0)
            ax.set_xlim(-30,30)
            ax.set_ylim(-30,30)
            ax.scatter(x_pos, y_pos, c=improvs, cmap="RdYlGn", norm=norm)
            ax.set_title(f"Generation {i}")
            if len(current) > 1:
                #ax.scatter(pso.global_best_pos[0], pso.global_best_pos[1], color="blue")
                ax.scatter(1,1, color="blue")


        ani = animation.FuncAnimation(fig, update, frames=params.iterations, interval=500, repeat=False)
        save_name = f"vis_{params.w}_{params.c1}_{params.c2}.mp4"
        #ani.save(save_name, writer="ffmpeg", dpi=500)
        #plt.ioff()
        plt.show()
