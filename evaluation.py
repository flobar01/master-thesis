import seaborn as sns
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
import matplotlib.animation as animation
import pandas as pd
from parameters import params

class Evaluation:

    def evaluate_global_fitness(global_fitness_history, global_fitness_history_ring, pso, pso_ring):
        plt.clf()
        fig = plt.figure()
        ax = sns.lineplot(global_fitness_history, label="GBest")
        sns.lineplot(global_fitness_history_ring, label="LBest")
        ax.set_ylim(0,100)
            #fig.suptitle(f"Global accumulated fitness")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Global Fitness")

    def evaluate_norm_improvs(sorted_norm_improvs1, sorted_norm_improvs1_ring, sorted_norm_improvs, sorted_norm_improvs_ring, pso, pso_ring):
        plt.clf()
        fig = plt.figure(figsize=(12.08, 7.2))
        subfigs = fig.subfigures(1, 2, wspace=0.08)
        
        plt.subplots_adjust(bottom=0.1, top=0.9, wspace=0.4, hspace=0.4)
        standard_pal = sns.color_palette()
        pal = {index: standard_pal[1] if str(index) == str(pso.global_best_index) else standard_pal[0] for index in sorted_norm_improvs.keys() }
        pal_ring = {index: standard_pal[1] if index == pso_ring.global_best_index else standard_pal[0] for index in sorted_norm_improvs_ring.keys() }
        pal1 = {index: standard_pal[1] if str(index) == str(pso.global_best_index) else standard_pal[0] for index in sorted_norm_improvs1.keys() }
        pal1_ring = {index: standard_pal[1] if index == pso_ring.global_best_index else standard_pal[0] for index in sorted_norm_improvs1_ring.keys() }
        # Left column
        left = subfigs[0]
        left.suptitle("Accumulated Impact Metric")
        axes_left = left.subplots(2, 1)

        # Right column
        right = subfigs[1]
        right.suptitle("Decreasing Impact Metric")
        axes_right = right.subplots(2, 1)
        x=list(sorted_norm_improvs.values())
        x1 = list(sorted_norm_improvs1.values())
        x_ring = list(sorted_norm_improvs_ring.values())
        x1_ring = list(sorted_norm_improvs1_ring.values())

        y = list(sorted_norm_improvs.keys())
        y1 = list(sorted_norm_improvs1.keys())
        y_ring = list(sorted_norm_improvs_ring.keys())
        y1_ring = list(sorted_norm_improvs1_ring.keys())
        # Plot
        sns.barplot(sorted_norm_improvs,      ax=axes_left[0], palette=pal,x=y, y=x, hue=y, legend=False)
        sns.barplot(sorted_norm_improvs_ring, ax=axes_left[1], palette=pal_ring, x=y_ring, y=x_ring, hue=y_ring, legend=False)

        sns.barplot(sorted_norm_improvs1,      ax=axes_right[0], palette=pal1, x=y1,y=x1, hue=y1, legend=False)
        sns.barplot(sorted_norm_improvs1_ring, ax=axes_right[1], palette=pal1_ring, x=y1_ring, y=x1_ring, hue=y1_ring, legend=False)

        axes = [
            [axes_left[0], axes_right[0]],
            [axes_left[1], axes_right[1]],
        ]
        for j in range(len(axes)):
            for i in range(len(axes[j])):
                ax = axes[j][i]
                ax.margins(x=0.04)
                if j == 0 and i == 0:
                    ax.set_title(f"GBest")
                if j == 0 and i == 1:
                    ax.set_title(f"GBest")
                if j == 1 and i == 0:
                    ax.set_title(f"LBest")
                if j == 1 and i == 1:
                    ax.set_title(f"LBest")
                #ax.set_xlabel("Index of particle")

    def evaluate_ranks(data_ranks, data_ranks_ring, best_improvs, best_improvs_ring, best_improvs1, best_improvs1_ring, pso, pso_ring):
        plt.clf()
        grey_color = "#00000010"
        standard_pal = sns.color_palette()
        pal = {index: standard_pal[0] if index == best_improvs else standard_pal[1] if index == pso.global_best_index else standard_pal[2] if index == best_improvs1 else grey_color for index in data_ranks.columns}
        pal_ring = {index: standard_pal[0] if index == best_improvs_ring else standard_pal[1] if index == pso_ring.global_best_index else standard_pal[2] if index == best_improvs1_ring else grey_color for index in data_ranks_ring.columns}

        fig, axes = plt.subplots(nrows = 2, figsize=(10,8))
        fig.set_size_inches(12,10)
        sns.lineplot(data_ranks, ax=axes[0], palette=pal, dashes=False)
        sns.lineplot(data_ranks_ring, ax=axes[1], palette=pal_ring, dashes=False)
        for i in range(2):
            ax = axes[i]
            handles, labels = ax.get_legend_handles_labels()
            to_remove = {}
            if i == 0:
                to_remove = {str(x) if x != best_improvs and x != pso.global_best_index and x != best_improvs1 else "" for x in data_ranks.columns}
            else:
                to_remove = {str(x) if x != best_improvs_ring and x != pso_ring.global_best_index and x != best_improvs1_ring else "" for x in data_ranks_ring.columns}

            for line, label in zip(ax.lines, labels):
                if label in to_remove:
                    line.set_linewidth(1)
            if i == 0:
                filtered = [(h, l) if l == str(pso.global_best_index) else (h,"lol") for h, l in zip(handles, labels) if l not in to_remove]
            else:
                filtered = [(h, l) if l == str(pso.global_best_index) else (h,"lol") for h, l in zip(handles, labels) if l not in to_remove]
            handles, labels = zip(*filtered)
            ax.legend(handles, labels)
            if i == 0:
                ax.set_title(f"GBest")
            else:
                ax.set_title(f"LBest")
            if i == 1:
                ax.set_xlabel("Time Step")
        
    def evaluate_improvs(data_improvs, data_improvs_ring, data_improvs1, data_improvs1_ring, best_improvs, best_improvs_ring, best_improvs1, best_improvs1_ring, pso, pso_ring):
        plt.clf()
        grey_color = "#00000010"
        standard_pal = sns.color_palette()
        pal = {index: standard_pal[0] if index == best_improvs else standard_pal[1] if index == pso.global_best_index else standard_pal[2] if index==best_improvs1 else grey_color for index in data_improvs.columns}
        pal_ring = {index: standard_pal[0] if index == best_improvs_ring else standard_pal[1] if index == pso_ring.global_best_index else standard_pal[2] if index==best_improvs1_ring else grey_color for index in data_improvs_ring.columns}
        pal1 = {index: standard_pal[0] if index == best_improvs1 else standard_pal[1] if index == pso.global_best_index else grey_color for index in data_improvs1.columns}
        pal1_ring = {index: standard_pal[0] if index == best_improvs1_ring else standard_pal[1] if index == pso_ring.global_best_index else grey_color for index in data_improvs1_ring.columns}

        fig, axes = plt.subplots(2,1)
        plt.subplots_adjust(bottom=0.1, top=0.9, wspace=0.4, hspace=0.4)
        #plt.yscale("symlog", linthresh=10)
        sns.lineplot(data_improvs, ax=axes[0], palette=pal, dashes=False)
        sns.lineplot(data_improvs_ring, ax=axes[1], palette=pal_ring, dashes=False)
        for j in range(len(axes)):
            ax = axes[j]
            handles, labels = ax.get_legend_handles_labels()
            to_remove = {}
            if j == 0:
                to_remove = {str(x) if x != best_improvs and x != pso.global_best_index and x != best_improvs1 else "" for x in data_improvs.columns}
            if j == 1:
                to_remove = {str(x) if x != best_improvs_ring and x != pso_ring.global_best_index and x != best_improvs1_ring else "" for x in data_improvs_ring.columns}

            for line, label in zip(ax.lines, labels):
                if label in to_remove:
                    line.set_linewidth(1)
            filtered = [(h, l) for h, l in zip(handles, labels) if l not in to_remove]
            handles, labels = zip(*filtered)
            ax.legend(handles, labels)

            if j == 0:
                ax.set_title("GBest")
            if j == 1:
                ax.set_title("LBest")
            if j==1:
                ax.set_xlabel("Time step")

    def evaluate_shared(pso_ring, sorted_improvs):
        plt.clf()
        plt.figure()
        shared_data = {}
        for i in pso_ring.particles:
            shared_data[i.index] = i.shared_best
        x = list(shared_data.keys())
        y = list(shared_data.values())
        sort = list(sorted_improvs.keys())
        standard_pal = sns.color_palette()
        pal = {index: standard_pal[1] if index == pso_ring.global_best_index else standard_pal[0] for index in shared_data.keys()}
        print(sort)
        sort.reverse()
        print(sort)
        ax = sns.barplot(shared_data, x=x, y=y, order=sort, palette=pal, hue=x, legend=False)
        ax.set_ylabel("Number of shares")