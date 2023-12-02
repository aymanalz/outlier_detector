# outlier_detector

The module outlier_detector.py has two classes: (1) Detector and (2) ScoreModel. The Detector class has multiple functions to perform noise detection. 
The entry point function to use this class is “purify”

## Test Examples:
All benchmark tests and used dataset can be found in “outlier_detector/tests”
To run test problems run the scripts listed below. These scripts conduct multiple numerical experiments and save results in a folder named “results”.  
The datasets include 2 synthetic problems (1D and 6D datasets), and a real-world problem (public supply in the southwest USA). 

### Synthetic Test Problems 
- Run one_dimension_case.py
- Run hartmann_6d.py
- run evaluating_data_model.py
- fig_effect_of_seed_number.py
- fig_effect_of_noise_signal_ratio.py
- run fig_compare_smapler_functions.py

### Public Supply Problem (outlier_detector/tests/ca_wu)
A complete copy of the dataset for the Public Supply can be found at https://doi.org/10.5066/P9FUL880. 
A subset of the data for California, Arizona, and Nevda are extracted and used for the testing (south_westh.csv)
- Run ca_case.py
- Run evaluate_equifinality.py
- run long_run.py







