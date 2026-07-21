import pickle
from evaluation import Evaluation
import matplotlib.pyplot as plt 
import seaborn as sns
from parameters import params

if __name__ == "__main__":
    sns.set_theme(style="darkgrid")
    with open('evaluation/run31/data.pkl', 'rb') as run_data:
        data = pickle.load(run_data)

        pso_global_best = data["pso_global_best"]
        pso_ring_global_best = data["pso_ring_global_best"]
        global_fitness_history = data["global_fitness_history"]
        global_fitness_history_ring = data["global_fitness_history_ring"]

        #Evaluation.evaluate_global_fitness(global_fitness_history, global_fitness_history_ring)
        #plt.savefig("fitness.png", dpi=300)

        sorted_norm_improvs1 = data["sorted_norm_improvs1"]
        sorted_norm_improvs1_ring = data["sorted_norm_improvs1_ring"]
        sorted_norm_improvs = data["sorted_norm_improvs"]
        sorted_norm_improvs_ring = data["sorted_norm_improvs_ring"]
        
        Evaluation.evaluate_norm_improvs(sorted_norm_improvs1, sorted_norm_improvs1_ring, sorted_norm_improvs, sorted_norm_improvs_ring, pso_global_best, pso_ring_global_best)
        #plt.savefig("improvs27.png", dpi=300)

        data_ranks = data["data_ranks"]
        data_ranks_ring = data["data_ranks_ring"]
        best_improvs = data["best_improvs"]
        best_improvs_ring = data["best_improvs_ring"]
        best_improvs1 = data["best_improvs1"]
        best_improvs1_ring = data["best_improvs1_ring"]
        
        #Evaluation.evaluate_ranks(data_ranks, data_ranks_ring, best_improvs, best_improvs_ring, best_improvs1, best_improvs1_ring, pso_global_best, pso_ring_global_best)
        #plt.savefig("ranks26.png", dpi=250)

        data_improvs = data["data_improvs"]
        data_improvs_ring = data["data_improvs_ring"]
        data_improvs1 = data["data_improvs1"]
        data_improvs1_ring = data["data_improvs1_ring"]
        shared_best = data["shared_best"]


        #Evaluation.evaluate_improvs(data_improvs, data_improvs_ring, data_improvs1, data_improvs1_ring, best_improvs, best_improvs_ring, best_improvs1, best_improvs1_ring, pso_global_best, pso_ring_global_best)
        #plt.savefig("normed_improvs27.png", dpi=100, bbox_inches='tight')

        #Evaluation.evaluate_shared(shared_best, sorted_norm_improvs_ring, pso_ring_global_best)
        #plt.savefig("shared.png", dpi=250)

        data_portions = data["data_portions"]
        data_portions_ring = data["data_portions_ring"]
        highest_portion_index = data["highest_portion_index"]
        second_highest_portion_index = data["second_highest_portion_index"]

        Evaluation.evaluate_portions_ot(data_portions, 8, 5, pso_global_best, True)
        plt.savefig(f"portions_ot_gbest.png", dpi=250, bbox_inches='tight')
        
        Evaluation.evaluate_portions_ot(data_portions_ring, 5, 3, pso_ring_global_best, True)
        plt.savefig(f"portions_ot_lbest.png", dpi=250, bbox_inches='tight')

        #Evaluation.evaluate_ranks(data_ranks, data_ranks_ring, 8, best_improvs_ring, 5, best_improvs1_ring, pso_global_best, pso_ring_global_best)
        #plt.savefig("testing/ranks.png", dpi=250, bbox_inches='tight')

        norm_portions = data["portions_data"]
        norm_portions_ring = data["portions_data_ring"]

        #Evaluation.evaluate_portions_heatmap(norm_portions_ring)
        #plt.savefig(f"testing/heatmap_ring.png", dpi=250, bbox_inches='tight')
        
        #Evaluation.evaluate_portions_heatmap(norm_portions)
        #plt.savefig(f"testing/heatmap.png", dpi=250, bbox_inches='tight')