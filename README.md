Pipeline to perform frank fits initially inspired by the procedure used by Sean Andrews in Andrews et al. 2021. 

Expanded for the continuum analysis of the exoALMA sources (exoALMA V, Curone and exoALMA) including galario fits for getting the geometrical parameters.



# From ms table to uv tables

Works only for ALMA data, not for VLA data (due to different polarization settings)

1. Create a source folder (e.g., `AA_Tau`). Within it, create subfolders `frank_fit` and `galario_fit` for respective fits.
2. Inside `frank_fit`, create folders `data`, `figs`, and `fits`.
3. Download `AA_Tau_time_ave_continuum.ms` and place it in `frank_fit/data`, along with `ExportMS_split_contspw_1ch.py`.
4. Use 'ExportMS_split_contspw_1ch.py' to:
    a. Produce 'AA_Tau_continuum.ms' with continuum spws averaged to 1 channel (starting ms should already be averaged to 30s).
    b. Generate visibility files for frank and galario ('AA_Tau_continuum.vis.npz' in 'frank_fit/data' and 'AA_Tau_galario_uvtable.txt' in 'galario_fit').
    
Ensure to set the 'contspws' parameter correctly by specifying the appropriate numbers for the continuum-only spectral windows. Tests have demonstrated that including line spectral windows (of course flagging the corresponding line emissions) can increase noise levels.



# galario 

## galario 1D 

In case of mostly axisymmetric sources, use a 1D profile as the model.

1. Prepare the python file (here 'galario_fit_AA_Tau_exoALMA_PointSource_4rings.py'). Fix the 'wle' paramenter and the model with the right model emission for the specific source. Fix also the 'nwalkers' and 'nsteps'. 
2. Run galario:
	python galario_fit_AA_Tau_exoALMA_PointSource_4rings.py
3. Experiment with multiple runs for convergence.


## galario 2D

In case of highly non-axisymmetric sources use a 2D model (here considering CQ Tau).

1. Inspect the parameters of the 2D model using 'model_galario_2D_new.ipynb'
2. Run galario with the file 'galario_2D_fit_CQ_Tau_exoALMA_2rings_2arcs_uniform_pos.py'. The initial walker positions are randomly selected using a flat probability function across a wide range of values. This approach thoroughly explores all potential values, which is particularly beneficial when dealing with numerous fitting parameters.
3. After a first run, employ 'galario_2D_fit_CQ_Tau_exoALMA_2rings_2arcs_2nd_step.py'. Here, the initial point 'p0' is chosen with a small Gaussian variation based on the best fit parameters obtained from the previous step.



# frank

1. Copy in the 'frank_fit' folder the files:
    - 'data_imaging.py' function to CLEAN the data.
    - 'diskdictionary.py' file to store values used by all the other files.
    - 'ImportMS.py' function called by 'model_imaging.py' and 'resid_imaging.py' to produce the model and residuals ms tables ('AA_Tau_continuum.model.ms' and 'AA_Tau_continuum.resid.ms') from 'AA_Tau_continuum.ms'
    - 'model_imaging.py' function to CLEAN the frank model visibilities as if they were data, using the same uv locations
    - 'resid_imaging' function to CLEAN residuals obtained by subtracting the frank fit from the data
    - 'setdata.py' is the main file that calls all the others, run frank, produces all figures in the right folders.
2. Modify the 'diskdictionary.py' file with the right values of incl, PA, dx, dy from the galario fit (or your favorite method) and all the other parameters used by frank and the CLEAN runs. 
3. Modify the 'setdata.py' with the right target name and then run everything with
	python setdata.py
