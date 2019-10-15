Directory contents
==================

Input_simulation: Scripts for input generation and simulation of StateMod
Old_data: Old compressed analysis data (not synced on GitHub)
output: Any output files from submission scripts
Output_analysis: Scripts for output extraction and analysis
Qgen: Data and scripts necessary to generate XBM and IWR inputs for each experiment
SALib: Python library for sensitivity analysis used by some experiments (not synced on GitHub)
Statemod_files: All StateMod files needed to generate inputs for experiments
Structures_files: Information files about structures in the basin
Summary_info: Summary data from historical record and experiments

Experiments:
------------
LHsamples_narrowed_1000: Narrowed LHSample of 1000 SOW x 10 realizations for all uncertain factors
LHsamples_original_1000: Original LHSample of 1000 SOW x 10 realizations for all uncertain factors
LHsamples_original_1000_AnnQonly: Original LHSample of 1000 SOW x 10 realizations for only streamflow factors
LHsamples_wider_1000: Wider (absurd) LHSample of 1000 SOW x 10 realizations for all uncertain factors

	Each contains:
	Experiment_files: StateMod input files necessary to run experiment
	Factor_mapping: Robustness heatmaps and logistic regression factor mapping for all structures
	Infofiles: Collected demand and shortage for each structure for all realizations
	Magnitude_sensitivity_analysis: Sensitivity analysis results on shortage magnitudes
	MultiyearShortageCurves: Duration curves for multiyear shortages
	RatioShortageCurves: Duration curves for shortage/demand ratios
	ShortagePercentileCurves: Duration curves for shortage magnitudes
	ShortageSensitivityCurves: Duration curves with embedded sensitivity analysis results
