import seaborn as sns
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
import matplotlib.animation as animation
import pandas as pd
from parameters import params
import numpy as np

class Evaluation:

    def evaluate_global_fitness(global_fitness_history, global_fitness_history_ring):
        plt.clf()
        fig = plt.figure()
        ax = sns.lineplot(global_fitness_history, label="GBest")
        sns.lineplot(global_fitness_history_ring, label="LBest")
            #fig.suptitle(f"Global accumulated fitness")
        ax.set_xlabel("Time Step")
        ax.set_yscale('log')
        ax.set_ylabel("Global Fitness")

    def evaluate_norm_improvs(sorted_norm_improvs1, sorted_norm_improvs1_ring, sorted_norm_improvs, sorted_norm_improvs_ring, pso_best_index, pso_ring_best_index):
        plt.clf()
        fig = plt.figure(figsize=(12.08, 7.2))
        subfigs = fig.subfigures(1, 2)
        
        #plt.subplots_adjust(bottom=0.1, top=0.9, wspace=0.4, hspace=0.4)

        standard_pal = sns.color_palette()
        pal = {index: standard_pal[1] if str(index) == str(pso_best_index) else standard_pal[0] for index in sorted_norm_improvs.keys() }
        pal_ring = {index: standard_pal[1] if index == pso_ring_best_index else standard_pal[0] for index in sorted_norm_improvs_ring.keys() }
        pal1 = {index: standard_pal[1] if str(index) == str(pso_best_index) else standard_pal[0] for index in sorted_norm_improvs1.keys() }
        pal1_ring = {index: standard_pal[1] if index == pso_ring_best_index else standard_pal[0] for index in sorted_norm_improvs1_ring.keys() }
        
        left = subfigs[0]
        left.suptitle("Accumulated Impact Metric")
        axes_left = left.subplots(1, 1)

        right = subfigs[1]
        right.suptitle("Decreasing Impact Metric")
        axes_right = right.subplots(1, 1)
        x=list(sorted_norm_improvs.values())
        x1 = list(sorted_norm_improvs1.values())
        x_ring = list(sorted_norm_improvs_ring.values())
        x1_ring = list(sorted_norm_improvs1_ring.values())

        y = list(sorted_norm_improvs.keys())
        y1 = list(sorted_norm_improvs1.keys())
        y_ring = list(sorted_norm_improvs_ring.keys())
        y1_ring = list(sorted_norm_improvs1_ring.keys())
        # Plot
        sns.barplot(sorted_norm_improvs,      ax=axes_left, palette=pal,x=y, y=x, hue=y, legend=False)
        sns.barplot(sorted_norm_improvs_ring, ax=axes_right, palette=pal1, x=y1, y=x1, hue=y1, legend=False)
        #axes_left.set_ylim(0,0.21)
        #axes_right.set_ylim(0,0.21)
        #sns.barplot(sorted_norm_improvs1,      ax=axes_right[0], palette=pal1, x=y1,y=x1, hue=y1, legend=False)
        #sns.barplot(sorted_norm_improvs1_ring, ax=axes_right[1], palette=pal1_ring, x=y1_ring, y=x1_ring, hue=y1_ring, legend=False)

        axes = [
            #[axes_left[0]],# axes_right[0]],
            #[axes_left[1]],# axes_right[1]],
        ]
        for j in range(len(axes)):
            for i in range(len(axes[j])):
                ax = axes[j][i]
                ax.margins(x=0.04)
                if j == 0:
                    ax.set_title(f"GBest")
                if j == 1:
                    ax.set_title(f"LBest")
                #ax.set_xlabel("Index of particle")

    def evaluate_ranks(data_ranks, data_ranks_ring, best_improvs, best_improvs_ring, best_improvs1, best_improvs1_ring, pso_best_index, pso_ring_best_index):
        plt.clf()
        grey_color = "#00000030"
        standard_pal = sns.color_palette()
        pal = {index: standard_pal[0] if index == best_improvs else standard_pal[1] if index == pso_best_index else standard_pal[2] if index == best_improvs1 else grey_color for index in data_ranks.columns}
        pal_ring = {index: standard_pal[0] if index == best_improvs_ring else standard_pal[1] if index == pso_ring_best_index else standard_pal[2] if index == best_improvs1_ring else grey_color for index in data_ranks_ring.columns}

        line_styles= {str(best_improvs): "--", 
                    str(best_improvs1): ":",
                    str(pso_best_index): "-."}
        fig, axes = plt.subplots(nrows = 1, figsize=(7,5))
        #fig.set_size_inches(12,10)
        ax = sns.lineplot(data_ranks, palette=pal, dashes=False)
        #sns.lineplot(data_ranks_ring, ax=axes[1], palette=pal_ring, dashes=False)
        for i in range(1):
            #ax = axes[i]
            handles, labels = ax.get_legend_handles_labels()
            to_remove = {}
            if i == 0:
                to_remove = {str(x) if x != best_improvs and x != pso_best_index and x != best_improvs1 else "" for x in data_ranks.columns}
            else:
                to_remove = {str(x) if x != best_improvs_ring and x != pso_ring_best_index and x != best_improvs1_ring else "" for x in data_ranks_ring.columns}

            for line, label in zip(ax.lines, labels):
                if label in to_remove:
                    line.set_linewidth(1)
                else:
                    line.set_linewidth(1.5)
                #if label in line_styles:
                    #line.set_linestyle(line_styles[label])
            if i == 0:
                filtered = [(h, r'Best Fitness') if l == str(best_improvs) else (h, r'2.Best Fitness') if l == str(best_improvs1) else (h, r'$\vec{g}(t)$') if l == str(pso_best_index) else (h, l) for h, l in zip(handles, labels) if l not in to_remove]
            else:
                filtered = [(h, r'$\bar{I}^{Acc}_g$') if l == str(best_improvs_ring) else (h, r'$\bar{I}^{Dec}_g$') if l == str(best_improvs1_ring) else (h, r'$\vec{g}(t)$') if l == str(pso_ring_best_index) else (h, l) for h, l in zip(handles, labels) if l not in to_remove]
            filtered[0],filtered[1] = filtered[1],filtered[0]
            handles, labels = zip(*filtered)
            ax.yaxis.set_inverted(True)
            ax.legend(handles, labels)
            plt.tight_layout()
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
            ax.set_ylabel("Fitness Rank")
            ax.set_xlabel("Time Step")
        
    def evaluate_improvs(data_improvs, data_improvs_ring, data_improvs1, data_improvs1_ring, best_improvs, best_improvs_ring, best_improvs1, best_improvs1_ring, pso_best_index, pso_ring_best_index):
        plt.clf()
        grey_color = "#00000010"
        standard_pal = sns.color_palette()
        pal = {index: standard_pal[0] if index == best_improvs else standard_pal[1] if index == pso_best_index else standard_pal[2] if index==best_improvs1 else grey_color for index in data_improvs.columns}
        pal_ring = {index: standard_pal[0] if index == best_improvs_ring else standard_pal[1] if index == pso_ring_best_index else standard_pal[2] if index==best_improvs1_ring else grey_color for index in data_improvs_ring.columns}
        pal1 = {index: standard_pal[0] if index == best_improvs1 else standard_pal[1] if index == pso_best_index else grey_color for index in data_improvs1.columns}
        pal1_ring = {index: standard_pal[0] if index == best_improvs1_ring else standard_pal[1] if index == pso_ring_best_index else grey_color for index in data_improvs1_ring.columns}

        fig, axes = plt.subplots(nrows = 1, figsize=(12.07, 7.2))
        #plt.subplots_adjust(bottom=0.1, top=0.9, wspace=0.4, hspace=0.4)
        #plt.yscale("symlog", linthresh=10)
        ax = sns.lineplot(data_improvs, palette=pal, dashes=False)
        #sns.lineplot(data_improvs_ring, ax=axes[1], palette=pal_ring, dashes=False)
        for j in range(1):
            handles, labels = ax.get_legend_handles_labels()
            to_remove = {}
            if j == 0:
                to_remove = {str(x) if x != best_improvs and x != pso_best_index and x != best_improvs1 else "" for x in data_improvs.columns}
            if j == 1:
                to_remove = {str(x) if x != best_improvs_ring and x != pso_ring_best_index and x != best_improvs1_ring else "" for x in data_improvs_ring.columns}
            for line, label in zip(ax.lines, labels):
                if label in to_remove:
                    line.set_linewidth(1)
            if j == 0:
                filtered = [(h, r'$\bar{I}^{Acc}_g$') if l == str(best_improvs) else (h, r'$\bar{I}^{Dis}_g$') if l == str(best_improvs1) else (h, r'$\vec{g}(t)$') if l == str(pso_best_index) else (h, l) for h, l in zip(handles, labels) if l not in to_remove]
            else:
                filtered = [(h, r'$\bar{I}^{Acc}_g$') if l == str(best_improvs_ring) else (h, r'$\bar{I}^{Dec}_g$') if l == str(best_improvs1_ring) else (h, r'$\vec{g}(t)$') if l == str(pso_ring_best_index) else (h, l) for h, l in zip(handles, labels) if l not in to_remove]
            handles, labels = zip(*filtered)
            ax.legend(handles, labels)

            plt.tight_layout()
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

            ax.set_title("LBest")
            ax.set_xlabel("Time step")
            ax.set_ylabel("Fitness Improvement")

    def evaluate_shared(shared_data, sorted_improvs, pso_ring_best_index):
        plt.clf()
        plt.figure()
        x = list(shared_data.keys())
        y = list(shared_data.values())
        sort = list(sorted_improvs.keys())
        standard_pal = sns.color_palette()
        pal = {index: standard_pal[1] if index == pso_ring_best_index else standard_pal[0] for index in shared_data.keys()}
        sort.reverse()
        ax = sns.barplot(shared_data, x=x, y=y, order=sort, palette=pal, hue=x, legend=False)
        ax.set_ylabel("Number of shares")

    def evaluate_portions(portions):
        plt.clf()
        plt.figure()
        ax = sns.barplot(portions)
        ax.set_yscale('log')

    def evaluate_portions_heatmap(data_portion, starting_ranks):
        plt.clf()
        plt.figure()
        ordered_particles = sorted(starting_ranks, key=starting_ranks.get)
        ordered_columns = [fr"$x_{{{i}}}(t=0)$" for i in ordered_particles]
        ordered_rows = [fr"$x_{{{i}}}(t=25)$" for i in ordered_particles]
        df_ordered = data_portion.loc[ordered_rows, ordered_columns]
        ax = sns.heatmap(df_ordered, cmap="seismic_r", annot=True, center=0, annot_kws={"fontsize":8})
        ax.tick_params(labelsize=5)
        
    def evaluate_portions_stacked(norm_portions):
        bottom = np.zeros(len(norm_portions))

        fig, ax = plt.subplots(figsize=(12, 6))

        colors = sns.color_palette("tab20", norm_portions.shape[1])

        for j, row in enumerate(norm_portions.index):
            values = norm_portions.loc[row].to_numpy()
            ax.bar(
                norm_portions.columns,
                values,
                bottom=bottom,
                label=row,
                color=colors[j]
            )
            bottom += values

        ax.legend(title="Particle", bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()
    
    def evaluate_portions_ot(data_portion, highest_portion_index, second_highest_portion_index, pso_best_index, is_gbest):
        plt.clf()
        grey_color = "#00000010"
        standard_pal = sns.color_palette()
        pal = {index: standard_pal[0] if index == highest_portion_index else standard_pal[1] if index == pso_best_index else standard_pal[2] if index == second_highest_portion_index else grey_color for index in data_portion.columns}

        fig, axes = plt.subplots(nrows = 1, figsize=(10,8))
        #fig.set_size_inches(12,10)
        ax = sns.lineplot(data_portion, palette=pal, dashes=False)
        #sns.lineplot(data_ranks_ring, ax=axes[1], palette=pal_ring, dashes=False)
        for i in range(1):
            #ax = axes[i]
            handles, labels = ax.get_legend_handles_labels()
            to_remove = {}
            if i == 0:
                to_remove = {str(x) if x != highest_portion_index and x != pso_best_index and x != second_highest_portion_index else "" for x in data_portion.columns}

            for line, label in zip(ax.lines, labels):
                if label in to_remove:
                    line.set_linewidth(1)
            
            if i == 0:
                filtered = [(h, r'$\bar{I}_g$') if l == str(highest_portion_index) else (h, r'$\bar{I}_{g2}$') if l == str(second_highest_portion_index) else (h, r'$\vec{g}$') if l == str(pso_best_index) else (h, l) for h, l in zip(handles, labels) if l not in to_remove]
            handles, labels = zip(*filtered)
            
            ax.legend(handles, labels)
            plt.tight_layout()
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
            ax.set_ylabel("Sum of coefficients")
            ax.set_xlabel("Time Steps")


    def evaluate_portions_on_output_particle(portions):
        plt.clf()
        medians = portions.quantile()
        ax = sns.boxplot(data=portions, color="gray")
        ax.tick_params(labelsize=7)
        ax.set_ylabel("Impact on Output Particle (Rank)", fontsize=8)
        ax.set_xlabel("Fitness of Starting Position (Rank)", fontsize=8)