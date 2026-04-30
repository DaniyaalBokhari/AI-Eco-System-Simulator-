# AI-Optimized Ecosystem Simulation
## Using Evolutionary Algorithms and Machine Learning for Ecological Balance

---

## Project Overview

A complete Python application that simulates a dynamic ecosystem (plants, herbivores, carnivores) on configurable N×N grids, and uses **four AI algorithms** to discover optimal parameter configurations that maximise ecosystem stability and biodiversity.

---

## Architecture

```
ecosystem_sim/
├── ecosystem.py      # Grid-based simulation engine (25+ parameters)
├── algorithms.py     # GA, PSO, Neural Network, Decision Tree (from scratch)
├── experiments.py    # Six structured experiments with statistical analysis
├── gui.py            # Full Tkinter GUI with real-time visualisation
├── main.py           # Entry point (GUI / CLI / test modes)
├── requirements.txt
└── results/          # Auto-generated experiment outputs
    ├── exp1_ga_vs_pso.json
    ├── exp2_sensitivity.json
    ├── exp3_nn.json
    ├── exp6_dt.json
    ├── nn_model.pkl
    ├── results_dashboard.png
    └── grid_visualization.png
```

---

## Quick Start

```bash
# Install dependencies
pip install numpy matplotlib scipy

# Launch GUI
python main.py

# Run experiments in terminal (no GUI needed)
python main.py --cli

# Quick demo mode
python main.py --quick

# Run smoke tests
python main.py --test
```

---

## Ecosystem Simulation Engine (`ecosystem.py`)

### 25 Tunable Parameters

| Group        | Parameter               | Default | Range       | Description                          |
|--------------|------------------------|---------|-------------|--------------------------------------|
| Plants       | init_plants            | 50      | 10–200      | Starting plant count                 |
|              | plant_growth_rate      | 0.35    | 0.05–0.80   | Probability of new plant per step    |
|              | plant_energy           | 10      | 5–20        | Energy provided when eaten           |
|              | plant_max_density      | 0.60    | 0.20–0.90   | Max fraction of grid filled          |
| Herbivores   | init_herbivores        | 20      | 5–100       | Starting herbivore count             |
|              | herb_energy_init       | 20      | 10–50       | Initial energy                       |
|              | herb_energy_max        | 40      | 20–80       | Max storable energy                  |
|              | herb_move_cost         | 1.0     | 0.5–3.0     | Energy cost per move                 |
|              | herb_eat_gain          | 10      | 5–25        | Energy gained from eating plant      |
|              | herb_repro_threshold   | 25      | 15–50       | Energy needed to reproduce           |
|              | herb_repro_cost        | 8       | 4–20        | Energy cost of reproduction          |
|              | herb_repro_prob        | 0.30    | 0.05–0.60   | Reproduction probability             |
|              | herb_starve_rate       | 1.5     | 0.5–4.0     | Passive energy loss per step         |
| Carnivores   | init_carnivores        | 8       | 2–40        | Starting carnivore count             |
|              | carn_energy_init       | 30      | 15–60       | Initial energy                       |
|              | carn_energy_max        | 60      | 30–100      | Max storable energy                  |
|              | carn_move_cost         | 2.0     | 1.0–5.0     | Energy cost per move                 |
|              | carn_eat_gain          | 20      | 10–40       | Energy gained from eating herbivore  |
|              | carn_repro_threshold   | 45      | 25–80       | Energy needed to reproduce           |
|              | carn_repro_cost        | 12      | 6–30        | Energy cost of reproduction          |
|              | carn_repro_prob        | 0.20    | 0.05–0.40   | Reproduction probability             |
|              | carn_starve_rate       | 2.0     | 0.5–5.0     | Passive energy loss per step         |
| Environment  | season_amplitude       | 0.40    | 0.0–0.80    | Seasonal growth oscillation          |
|              | hemisphere             | 1.0     | -1 to +1    | +1=Northern, -1=Southern             |
|              | carrying_capacity      | 1.00    | 0.50–1.50   | Multiplier on max plant density      |

### Fitness Function
```
Fitness = 0.5 × (Survival_Months / 120) + 0.3 × Shannon_Diversity + 0.2 × Stability
```

### Shannon Diversity Index
```
H = -Σ p_i × ln(p_i)   (normalised by ln(3) → [0, 1])
```

### Seasonal Mechanics
```
season_factor = 1 + A × sin(step × hemisphere × π/6)
```
Growth rates scale by this factor; Northern/Southern hemispheres are phase-opposed.

---

## Algorithms (`algorithms.py`)

### 1. Genetic Algorithm
| Parameter          | Value              |
|-------------------|--------------------|
| Population size   | 50 individuals     |
| Selection         | Tournament (k=3)   |
| Crossover         | Uniform (p=0.5/gene) |
| Mutation rate     | 10% of genes       |
| Mutation strength | 5% of param range (Gaussian) |
| Elitism           | Top 2 always survive |

```python
ga = GeneticAlgorithm(grid_size=20, pop_size=50)
result = ga.run(n_generations=30)
# result['best_params'], result['best_fitness'], result['history']
```

### 2. Particle Swarm Optimization
| Parameter          | Value |
|-------------------|-------|
| Particles         | 30    |
| Inertia weight w  | 0.7   |
| Cognitive c₁      | 1.5   |
| Social c₂         | 1.5   |
| Velocity clamp    | ±20% of param range |

```python
pso = ParticleSwarmOptimization(grid_size=20, n_particles=30, w=0.7, c1=1.5, c2=1.5)
result = pso.run(n_iterations=40)
```

### 3. Neural Network
| Property        | Value                 |
|----------------|-----------------------|
| Architecture   | 25 → 64 → 32 → 1     |
| Hidden layers  | ReLU activation       |
| Output layer   | Linear activation     |
| Loss           | Mean Squared Error    |
| Optimiser      | SGD with momentum     |
| Regularisation | L2 weight decay       |
| Initialisation | He (Kaiming)          |

Predicts ecosystem survival time given 25 parameter inputs.

```python
nn = NeuralNetwork(learning_rate=0.002)
nn.fit(X_train, y_train, epochs=100)
predictions = nn.predict(X_test)
nn.save('nn_model.pkl')
```

### 4. Decision Tree
| Property         | Value            |
|-----------------|------------------|
| Criterion       | Entropy (information gain) |
| Max depth       | 5                |
| Min samples     | 10               |
| Output          | Human-readable IF/THEN rules |

```
Rule 1:
IF herb_repro_prob ≤ 0.549 AND herb_energy_init ≤ 45.274
  THEN ecosystem COLLAPSES (≤60 months)
  [Confidence: 100%, Samples: 240]

Rule 2:
IF herb_repro_prob ≤ 0.549 AND herb_energy_init > 45.274 AND init_carnivores ≤ 7.809
  THEN ecosystem SURVIVES (>60 months)
  [Confidence: 67%, Samples: 3]
```

---

## Six Experiments (`experiments.py`)

### Experiment 1: GA vs PSO Comparison
- Runs N independent replicates of each algorithm
- Computes: best/mean/worst fitness, convergence speed, wall-clock time
- **Statistical test:** Welch's t-test (two-tailed, α=0.05) on final fitness distributions
- Output: winner, p-value, effect size

### Experiment 2: Parameter Sensitivity Analysis
- One-at-a-time (OAT) analysis: varies each parameter across full range while holding others at default
- Records fitness variance as sensitivity measure
- Pearson correlation between each parameter and fitness over random samples
- Output: ranked parameter importance, top/bottom 5 parameters

### Experiment 3: Neural Network Evaluation
- Generates 1000+ simulation samples as training data
- Trains NN, evaluates on held-out test set
- Reports: MAE, RMSE, R²; target MAE < 10, R² > 0.8
- Permutation feature importance

### Experiment 4: Hemisphere Comparison
- Compares Northern (hemisphere=+1) vs Southern (hemisphere=-1) seasonal phases
- Tested across all four grid sizes (10×10, 20×20, 30×30, 40×40)
- Welch's t-test per grid size to assess seasonal significance

### Experiment 5: Evolutionary Dynamics Tracking
- Runs full GA (30 generations) and records per-generation statistics
- Tracks best ecosystem's population trajectory over time
- Outputs: fitness landscape, genetic diversity, population dynamics

### Experiment 6: Decision Tree Rule Extraction
- Trains DT on simulation data with binary survival label
- Extracts all IF/THEN rules with confidence and sample counts
- Reports feature importance, train/test accuracy
- Human-readable interpretability analysis

---

## GUI Features (`gui.py`)

| Feature                | Description                                                  |
|-----------------------|--------------------------------------------------------------|
| Grid Size Selector    | Radio buttons: 10×10, 20×20, 30×30, 40×40                   |
| Algorithm Selector    | Dropdown: GA, PSO, Neural Network, Decision Tree             |
| Live Grid             | Colour-coded organism display, updates every simulation step |
| Population Graphs     | Real-time plots: plants, herbivores, carnivores over time    |
| Diversity & Stability | Live Shannon diversity and stability metric plots            |
| Fitness Convergence   | Optimisation convergence curve during algorithm run          |
| Comparison Mode       | GA vs PSO side-by-side convergence overlay                   |
| Parameter Sliders     | 25 sliders grouped by category with reset button            |
| Algorithm Config      | Spinboxes for GA generations, PSO iterations, NN epochs      |
| Experiment Runner     | One-click run of all 6 experiments with live log             |
| Organism Visibility   | Toggle plants/herbivores/carnivores on/off                  |
| Speed Control         | Simulation speed slider (0.02s – 0.40s per step)           |
| Export: PNG           | Save current graph as PNG (150 DPI)                          |
| Export: CSV           | Save population/fitness history as CSV                       |
| Export: JSON          | Save full session report                                     |
| Best Params Display   | Shows optimised parameters found by last algorithm run       |
| Status Bar            | Real-time status messages                                    |
| Menu Bar              | File, View, Help menus                                       |

---

## Grid Sizes & Scalability

| Grid    | Cells | Relative Scale | Use Case                    |
|---------|-------|----------------|-----------------------------|
| 10×10   | 100   | 1×             | Fast prototyping, testing   |
| 20×20   | 400   | 4×             | Standard experiments        |
| 30×30   | 900   | 9×             | Detailed population studies |
| 40×40   | 1600  | 16×            | Large-scale scalability     |

---

## Experimental Results

### Sensitivity Analysis (Top Parameters)
1. `init_herbivores` — herbivore starting population most affects dynamics
2. `plant_max_density` — carrying capacity strongly constrains plant availability
3. `herb_energy_max` — energy ceiling determines reproductive viability

### Decision Tree Rules (98.3% accuracy)
- **Key finding:** `herb_repro_prob > 0.549` is the single strongest survival predictor
- `init_carnivores` and `carn_energy_max` are secondary discriminators
- Simple 5-rule tree achieves near-perfect classification

### GA vs PSO
- Both algorithms converge to fitness ~0.40–0.56 on 10×10 grid
- PSO slightly faster convergence; GA has higher variance (more exploration)
- Statistical difference not always significant at α=0.05 for short runs

---

## Dependencies

```
numpy>=1.24       # Array operations, random sampling
matplotlib>=3.7   # GUI plots, static figure export
scipy>=1.10       # Optional: precise t-test p-values (falls back to normal approximation)
tkinter           # GUI (standard library)
```

**No scikit-learn, no PyTorch, no TensorFlow** — all algorithms implemented from scratch.

---

## Team Contributions

| Module           | Responsibility                                    |
|-----------------|---------------------------------------------------|
| `ecosystem.py`  | Simulation engine, organism logic, metrics        |
| `algorithms.py` | GA, PSO, NN backprop, Decision Tree from scratch  |
| `experiments.py`| Experimental design, statistical analysis         |
| `gui.py`        | GUI design, real-time visualisation, export       |

---

## File Reference

| File                         | Description                             |
|-----------------------------|-----------------------------------------|
| `results/exp1_ga_vs_pso.json` | GA vs PSO fitness comparison data      |
| `results/exp2_sensitivity.json` | Parameter sensitivity rankings       |
| `results/exp3_nn.json`      | NN accuracy metrics                     |
| `results/exp6_dt.json`      | Decision tree rules and accuracy        |
| `results/nn_model.pkl`      | Serialised trained Neural Network       |
| `results/results_dashboard.png` | 6-panel results figure             |
| `results/grid_visualization.png` | All 4 grid sizes rendered         |
