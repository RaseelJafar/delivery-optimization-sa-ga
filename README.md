# Optimization Strategies for Local Package Delivery (SA + GA)

simulate and optimize a local package delivery shop by assigning packages to vehicles and planning routes to minimize the total traveled distance.

## Problem Summary
- Each package has: destination (x,y), weight (kg), and priority (1–5).
- Each vehicle has a capacity (kg).
- Objective: minimize total distance traveled by all vehicles (Euclidean distance).
- Priority should generally be respected, but it can be violated if it greatly increases cost.

## Implemented Algorithms
- Simulated Annealing (SA)
- Genetic Algorithm (GA)

At runtime, the user can choose which algorithm to generate a solution.

## Inputs / Data
- `data.txt`

## Output
The program generates plots of the produced solutions:
- `sa_solution_plot.png`
- `ga_solution_plot.png`

## How to Run
```bash
python main.py
