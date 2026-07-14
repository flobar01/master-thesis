import pickle
from evaluation import Evaluation
import matplotlib.pyplot as plt 
import seaborn as sns

if __name__ == "__main__":
    sns.set_theme(style="darkgrid")
    with open('evaluation/run27/data.pkl', 'rb') as run_data:
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
        plt.savefig("improvs27.png", dpi=300)

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


        Evaluation.evaluate_improvs(data_improvs, data_improvs_ring, data_improvs1, data_improvs1_ring, best_improvs, best_improvs_ring, best_improvs1, best_improvs1_ring, pso_global_best, pso_ring_global_best)
        plt.savefig("normed_improvs27.png", dpi=100, bbox_inches='tight')

        #Evaluation.evaluate_shared(shared_best, sorted_norm_improvs_ring, pso_ring_global_best)
        #plt.savefig("shared.png", dpi=250)