import seaborn as sns
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
import matplotlib.animation as animation
from parameters import params

class Evaluation:

    def evaluate_global_fitness(global_fitness_history, global_fitness_history_ring, pso, pso_ring):
        plt.clf()
        fig, axes = plt.subplots(ncols=2, figsize=(12,6))
        sns.lineplot(global_fitness_history, ax=axes[0])
        sns.lineplot(global_fitness_history_ring, ax=axes[1])
        for i in range(2):
            ax = axes[i]
            fig.suptitle(f"Global fitness w: {params.w}, c1: {params.c1}, c2: {params.c2}")
            if i == 0:
                ax.set_title(f"normal PSO best: {pso.global_best}")
            else:
                ax.set_title(f"ring PSO best: {pso_ring.global_best}")
            ax.set_xlabel("Generation")

    def evaluate_norm_improvs(sorted_norm_improvs1, sorted_norm_improvs1_ring, sorted_norm_improvs, sorted_norm_improvs_ring, pso, pso_ring):
        plt.clf()
        fig, axes = plt.subplots(2,2, figsize=(17,11))
        sns.barplot(sorted_norm_improvs, ax=axes[0][0])
        sns.barplot(sorted_norm_improvs_ring, ax=axes[1][0])
        sns.barplot(sorted_norm_improvs1, ax=axes[0][1])
        sns.barplot(sorted_norm_improvs1_ring, ax=axes[1][1])
        for j in range(len(axes)):
            for i in range(len(axes[j])):
                ax = axes[j][i]
                fig.suptitle(f"Normed overall improvements w: {params.w}, c1: {params.c1}, c2: {params.c2}")
                if j == 0 and i == 0:
                    ax.set_title(f"normal PSO best: {pso.global_best_index}")
                if j == 0 and i == 1:
                    ax.set_title(f"normal PSO decreasing best: {pso.global_best_index}")
                if j == 1 and i == 0:
                    ax.set_title(f"ring PSO best: {pso_ring.global_best_index}")
                if j == 1 and i == 1:
                    ax.set_title(f"ring PSO decreasing best: {pso_ring.global_best_index}")
                ax.set_xlabel("Index of particle")

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
            filtered = [(h, l) for h, l in zip(handles, labels) if l not in to_remove]
            handles, labels = zip(*filtered)
            ax.legend(handles, labels)
            fig.suptitle(f" Rank evaluation w: {params.w}, c1: {params.c1}, c2: {params.c2}")
            if i == 0:
                ax.set_title(f"normal PSO best: {pso.global_best_index}")
            else:
                ax.set_title(f"ring PSO best: {pso_ring.global_best_index}")
            ax.set_xlabel("Generation")
        
    def evaluate_improvs(data_improvs, data_improvs_ring, data_improvs1, data_improvs1_ring, best_improvs, best_improvs_ring, best_improvs1, best_improvs1_ring, pso, pso_ring):
        plt.clf()
        grey_color = "#00000010"
        standard_pal = sns.color_palette()
        pal = {index: standard_pal[0] if index == best_improvs else standard_pal[1] if index == pso.global_best_index else grey_color for index in data_improvs.columns}
        pal_ring = {index: standard_pal[0] if index == best_improvs_ring else standard_pal[1] if index == pso_ring.global_best_index else grey_color for index in data_improvs_ring.columns}
        pal1 = {index: standard_pal[0] if index == best_improvs1 else standard_pal[1] if index == pso.global_best_index else grey_color for index in data_improvs1.columns}
        pal1_ring = {index: standard_pal[0] if index == best_improvs1_ring else standard_pal[1] if index == pso_ring.global_best_index else grey_color for index in data_improvs1_ring.columns}

        fig, axes = plt.subplots(2,2, figsize=(17,11))
        #plt.yscale("symlog", linthresh=10)
        sns.lineplot(data_improvs, ax=axes[0][0], palette=pal, dashes=False)
        sns.lineplot(data_improvs_ring, ax=axes[1][0], palette=pal_ring, dashes=False)
        sns.lineplot(data_improvs1, ax=axes[0][1], palette=pal1, dashes=False)
        sns.lineplot(data_improvs1_ring, ax=axes[1][1], palette=pal1_ring, dashes=False)

        for j in range(len(axes)):
            for i in range(len(axes[j])):
                ax = axes[j][i]
                handles, labels = ax.get_legend_handles_labels()
                to_remove = {}
                if i == 0 and j == 0:
                    to_remove = {str(x) if x != best_improvs and x != pso.global_best_index else "" for x in data_improvs.columns}
                if j == 1 and i == 0:
                    to_remove = {str(x) if x != best_improvs_ring and x != pso_ring.global_best_index else "" for x in data_improvs_ring.columns}
                if j == 0 and i == 1:
                    to_remove = {str(x) if x != best_improvs1 and x != pso.global_best_index else "" for x in data_improvs1.columns}
                if j == 1 and i == 1:
                    to_remove = {str(x) if x != best_improvs1_ring and x != pso_ring.global_best_index else "" for x in data_improvs1_ring.columns}

                for line, label in zip(ax.lines, labels):
                    if label in to_remove:
                        line.set_linewidth(1)
                filtered = [(h, l) for h, l in zip(handles, labels) if l not in to_remove]
                handles, labels = zip(*filtered)
                ax.legend(handles, labels)

                fig.suptitle(f"Evaluation improvs w: {params.w}, c1: {params.c1}, c2: {params.c2}")
                if j == 0 and i == 0:
                    ax.set_title(f"normal PSO best: {pso.global_best_index}")
                if j == 0 and i == 1:
                    ax.set_title(f"normal PSO decreasing best: {pso.global_best_index}")
                if j == 1 and i == 0:
                    ax.set_title(f"ring PSO best: {pso_ring.global_best_index}")
                if j == 1 and i == 1:
                    ax.set_title(f"ring PSO decreasing best: {pso_ring.global_best_index}")
                ax.set_xlabel("Generation")

    def evaluate_shared(pso_ring):
        plt.clf()
        plt.figure()
        shared_data = {}
        for i in pso_ring.particles:
            shared_data[i.index] = i.shared_best
        ax = sns.barplot(shared_data)
        ax.set_xlabel("Index")
        ax.set_ylabel("# particle shared global best")