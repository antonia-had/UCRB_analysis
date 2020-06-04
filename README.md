Assuming you have:
------------------
1. StateMod files from https://github.com/antonia-had/cdss-app-statemod-fortran
2. LHsamples_original_1000.txt in Qgen folder
3. StateMod simlink to original
4. Generated XBM and IWR StateMod files for UCRB

Steps:
------
1. Generate all other necessary experiment files and submit runs using input_statemod_uncurtailed.py and .sh
2. Extract all relevant outputs using infofile_uncurtailed.py and .sh
3. Simulate uncurtailement demands and shortages for TBD using input_statemod_uncurtailment_only.py
4. Run sensitivity analysis for both magnitude and duration of shortage using sensitivity_analysis.py and .sh
5. Generate shortage duration curves and sensitivity analysis curves using shortage_duration_curves.py and .sh
6. Get streamflows for 15-mile reach (needed for robustness analysis) using 15mile_streamflow.py and .sh
7. Perform scenario discovery using LR_factor_mapping.py and .sh

These scripts can be used to replicate:
 - Hadjimichael, A., Quinn, J.D., Reed, P.M., Yates, D. Wilson, E., Basdekas, L., Defining robustness, vulnerabilities, and consequential scenarios for diverse stakeholder interests within the Upper Colorado River Basin. Earth’s Future, doi: 10.1029/2020EF001503
 - Hadjimichael, A., Quinn, J.D., Reed, Advancing diagnostic model evaluation to better understand water shortage mechanisms in institutionally complex river basins. Water Resources Research (submitted)
