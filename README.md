# Towards More Principled Foundations of XAI Through Uncertainty-aware Counterfactuals

This repository contains the code for the experiments of the paper: "Towards More Principled Foundations of XAI Through Uncertainty-aware Counterfactuals".

To get started, please run `uv sync` to create the virtual enviroment.
Additionally one needs to run `uv pip install probly` to get the newest version of the Uncertainty package `probly`.

The repository is structured as follows:
- All the proposed optimization procedures can be found in *src/uncertainty_cfs/property_procedures*
- The best sweep results can be found in *best_sweep_results_[method_identifier]_.csv*
- The tabular experiments can be found in *experiments_tabular/*, the datasets in *tabular/*.
- Saved model weights are contained in *models/*
- Scripts for training can be found in *training/*.
- All evaluated hyperparameters are stated in *sweep_configurations.txt*, *sweep_configurations_clue.txt* and *sweep_configurations_face.txt*.
