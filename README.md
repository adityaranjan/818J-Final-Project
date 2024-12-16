# 818J-Final-Project
Final Project for CMSC 818J Fall 2024

Here is a sample README file for the repository containing the baseline and optimized simulators for a single layer of Graph Neural Network (GNN) inference:

```markdown
# GNN Single Layer Inference Simulator

This repository contains two Python scripts that simulate the inference process of a single layer in Graph Neural Networks (GNNs). The simulators are designed to evaluate the performance of baseline and optimized architectures in terms of latency and cycle efficiency.

## Files

- **baseline.py**: This file contains the code for the baseline simulator, which uses a traditional architecture with separate aggregation and combination phases. The simulator is implemented using an array of Multiply-and-Accumulate (MAC) units and a systolic array for matrix multiplication.

- **optimized.py**: This file contains the code for the optimized simulator, which integrates aggregation and combination phases using a SIMD array of multipliers and a modified balanced adder tree. This approach allows for fine-grained overlap between operations, aiming to reduce latency and improve efficiency.

## Overview

Graph Neural Networks (GNNs) are powerful tools for processing graph-structured data. They typically operate in two phases:
- **Aggregation Phase**: Collects information from neighboring nodes.
- **Combination Phase**: Updates node features using learned weights.

The baseline simulator models these phases separately, while the optimized simulator aims to fuse them, reducing processing time by overlapping operations.

## Features

- **Baseline Simulator**:
  - Utilizes 1024 MAC units by default.
  - Implements a 32x32 systolic array by default for matrix multiplication.
  - Separates aggregation and combination phases.

- **Optimized Simulator**:
  - Utilizes 1024 multipliers by default with a modified adder tree.
  - Overlaps aggregation and combination phases.

## Usage

Both simulators are designed to work with datasets from PyTorch Geometric's Planetoid collection. The default dataset used is "Cora". To run either simulator, execute the script in a Python environment with the required dependencies installed.

Example usage:
```
# For Baseline Simulator
simulator = BaselineSimulator(Planetoid(name="Cora", root="path/to/dataset"))
simulator.load_data()
simulator.set_weight_matrix()
agg_steps, comb_steps = simulator.simulate()

# For Optimized Simulator
simulator = OptimizedSimulator(Planetoid(name="Cora", root="path/to/dataset"))
simulator.load_data()
simulator.set_weight_matrix()
agg_steps, total_steps = simulator.simulate()
```

## Performance Metrics

The simulators provide various performance metrics, including:
- Number of cycles for aggregation and combination phases.
- Latency metrics: average, maximum, and minimum latencies per vertex.

## Additional Documentation

Each code file includes detailed documentation on classes and methods used within the simulators. Please refer to these files for further insights into their implementation.
```
