# A Bayesian Optimisation Approach to Inventory Management in the Bristol Stock Exchange

**Kaloyan Kirov — Department of Physics, University of Bath**

---

## Overview

This repository contains all code, data, and figures produced for a dissertation investigating whether a purpose-designed algorithmic trader, with its parameters tuned through Bayesian optimisation, could compete profitably against established strategies in the Bristol Stock Exchange (BSE) simulator. The central question was whether making inventory management an explicit part of a trader's design, rather than a structural side effect of its quoting behaviour, would produce more consistent profitability across a range of simulated market conditions.

---

## Background

The Bristol Stock Exchange is an open source Python framework developed by Professor Dave Cliff at the University of Bristol for use in teaching and research. It simulates a financial market in which algorithmic traders interact through a Limit Order Book, executing trades via a Continuous Double Auction. Six baseline traders are available in the BSE, ranging from the trivially simple Giveaway trader to the Parameterised Response Stochastic Hill Climber, and are used throughout this project as the competitive field against which the Custom Trader is evaluated.

The standard BSE profit definition treats profit as the margin captured on each individual trade and assumes fixed buyer and seller roles throughout a session. This project introduces a modified profit measure that accounts for the full mark to market value of a trader's position at each time step:

P = C_f + (I * SP) - C_0

where C_f is the trader's final cash balance, I is their inventory (shares held), SP is the mean price of the most recent 50 transactions, and C_0 is the initial cash balance of 10,000. This formulation means that accumulating a large unclosed inventory position at session end directly reduces profit, making inventory management a meaningful design objective.

---

## Methodology

### Preliminary Study

The first stage of the research ran all six baseline traders against one another across four market conditions: Stable, Shock Up, Shock Down, and Multi Regime. Each condition ran for 150 independent trials of 3,600 time steps, with five traders of each type per session. The results revealed that inventory accumulation and the speed of trade-to-trade adaptation were the two primary drivers of profitability under this profit definition, and directly motivated the design of the Custom Trader.

### Custom Trader Design

The Custom Trader uses a ZIP style margin update rule controlled by a single learning rate parameter alpha, without the smoothing term used in ZIP. It supplements this with two inventory management mechanisms: a position limit L that applies a proportional bias to the margin once inventory exceeds half the limit, and a hard defensive mode above the full limit that reduces the trader's competitiveness until inventory returns to a safe range. A volatility threshold parameter sigma_v provides an independent route into defensive mode when recent price volatility, measured as the standard deviation of the 50 most recent transaction prices, exceeds the threshold.

### Bayesian Optimisation

The three parameters alpha, L, and sigma_v were tuned using Bayesian optimisation with a Gaussian Process surrogate model and Expected Improvement as the acquisition function, implemented via the Scikit Optimize library. The search ran for 100 iterations over the parameter space alpha in [0.001, 0.30], L in [20, 800], and sigma_v in [3.0, 15.0], with 20 random initial evaluations before the GP began guiding the search. Each evaluation simulated 10 sessions per market condition and recorded the mean profit of the Custom Trader across those sessions.

### Main Simulation

The optimised Custom Trader was evaluated against all six baseline traders across 150 independent trials per condition, replicating the environment used in the preliminary study. The final evaluation used the optimal parameters identified by the Bayesian optimisation.

### Bootstrap Resampling

To assess whether the observed profit rankings were statistically robust, bootstrap resampling was applied to each trader's distribution of 150 per-trial profits. Confidence intervals were constructed from 10,000 resamples, and a long-term cumulative profit projection was built from 1,000 bootstrap iterations each of 500 simulated future sessions.

---

## Repository Structure

```
.
├── BSE.py                          Bristol Stock Exchange simulator (Cliff, 2024)
├── inventory_wrapper.py            Inventory tracking framework and Custom Trader
├── run_simulation.py               Runs Bayesian optimisation then main simulation
├── run_final_evaluation.py         Runs the final evaluation with fixed optimal parameters
├── 01_preliminary_analysis.ipynb   Preliminary study figures (Figure 1)
├── 02_bayesian_optimisation.ipynb  Bayesian optimisation analysis (Figures 2 and 3)
├── 03_main_results.ipynb           Main results figures (Figures 4 through 7)
├── LICENSE                         MIT licence for BSE.py
├── data/
│   ├── preliminary_results.csv     Per-trial results from the preliminary study
│   ├── preliminary_inventory.csv   Inventory time series from the preliminary study
│   ├── bayesian_search_log.csv     Full log of all 100 Bayesian optimisation iterations
│   ├── optimal_parameters.csv      Optimal parameter values identified by the GP
│   ├── inventory_timeseries.csv    Per-timestep inventory data from the main simulation
│   ├── final_results.csv           Per-trial results from the final evaluation
│   └── summary_statistics.csv      Summary statistics from the final evaluation
└── figures/
    ├── fig1_zip_profit_evolution.png     Mean ZIP profit across conditions (preliminary)
    ├── fig2_gp_parameter_space.png       Gaussian Process parameter space exploration
    ├── fig3_inventory_trajectories.png   Mean inventory trajectories for CT, ZIP and PRSH
    ├── fig4_profit_distributions.png     Profit distributions across all conditions
    ├── fig5_defensive_mode.png           Custom Trader defensive mode activation
    ├── fig6_bootstrap_distributions.png  Bootstrap mean profit distributions
    └── fig7_cumulative_profit.png        Long-term cumulative profit projection
```

---

## Reproducing the Results

To reproduce the full pipeline from scratch:

**1. Run the Bayesian optimisation and main simulation**

```bash
python run_simulation.py
```

This will perform 100 iterations of Bayesian optimisation to find the optimal parameters for the Custom Trader, then run 150 trials per market condition with all seven trader types. Results are written to `data/`.

**2. Run the final evaluation**

```bash
python run_final_evaluation.py
```

This runs the final evaluation using the optimal parameters saved in `data/optimal_parameters.csv`, producing the results used for the report figures.

**3. Generate the figures**

Open and run each notebook in order. The notebooks read from `data/` and write figures to `figures/`.

The pre-computed data files in `data/` and pre-generated figures in `figures/` are already included in this repository, so step 3 can be run without first completing steps 1 and 2.

---

## Dependencies

```
python >= 3.10
numpy
pandas
matplotlib
scipy
scikit-optimize
```

Install with:

```bash
pip install numpy pandas matplotlib scipy scikit-optimize
```

---

## Acknowledgements

The Bristol Stock Exchange simulator (`BSE.py`) was developed by Professor Dave Cliff of the University of Bristol and is used here under its MIT licence. The original source is available at https://github.com/davecliff/BristolStockExchange.

The profit redefinition and inventory framework used throughout this project were developed in discussion with Joe Page, a fellow researcher working independently on the BSE.
