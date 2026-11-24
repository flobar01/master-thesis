# Impact of Particles on the Swarm in PSO
This is a Framework to investigate the impact of different particles on the other particles of the swarm in Particle Swarm Optimization. The Framework includes LBest and GBest topology implementation and Rastrigin, Ackley, Rosenbrock and Sphere fitness functions. During the runs of the PSO the particles store additional data to provide insights that are visualized in multiple ways.

<details>
<summary><b>Content</b></summary>
  
- `/saves_*`
  - simulated runs are saved into that folder so runs don't have to be redone multiple times
  - saves time while evaluating
- `/evaluation_*`
  - multiple different evaluations that are already done 
- `pso*.py`, `particle*.py`
  - logic for the swarm behaviour
  - divided into ring and star topology
  - also responsible for the collected data of the particles
- `evaluation.py`
  - functions to visualize data into multiple different plots
  - some show multiple plots when needed
- `evaluation_main.py`
  - can be used to load a specific run of the `/saves_*` folder to investigate specific runs
- `parameters.py`
  - simply the used parameters for the pso runs    
- `main.py`
  - runs the PSO simulation
  - collects data during the runs
  - using functions of the `evaluation.py` to visualize data
</details>
