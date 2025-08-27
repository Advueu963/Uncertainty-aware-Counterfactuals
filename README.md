# MasterThesis

This repository contains the entire code for the MasterThesis "Uncertainty as a Unifying Framework for Counterfactual Explanations" supervised by Kascper Sokol, Maximilian Muschalik and Eyke Hüllermeier.

The virtual enviroment is created through `uv?.
Please install `uv` and then use `uv sync` to obtain the virtual enviroment.

`property_procedures` is the main folder containing all the new proposed optimisation functions for the different properties.

`epiuc` and `carla` are other packages regarding uncertainty aware models and baseline methods for counterfactual generation, respectively.

`visualize_properties_sgd` and `visualize_properties_adam` contain experimental files for obtaining the large images shown in the thesis.

The files `<...>_properties_all_datasets.py` are the experiments to obtain the metric tables as `.csv` file.

The weights of the ensemble can be found and used in `models/`
