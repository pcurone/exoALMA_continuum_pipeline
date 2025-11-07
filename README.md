This is the pipeline used to analyze the continuum emission of the [exoALMA Large Program](https://www.exoalma.com), with results presented in [Curone et al. 2025 (exoALMA IV)](https://ui.adsabs.harvard.edu/abs/2025ApJ...984L...9C/abstract).

![](./image_exoALMA_pipeline_AA_Tau.png)

This pipeline was initially inspired by the procedure developed in  [Andrews et al. 2021](https://ui.adsabs.harvard.edu/abs/2021ApJ...916...51A/abstract) ([GitHub repo](https://github.com/seanandrews/DSHARP_CPDs)).

In addition to standard Python packages, you’ll need the following tools installed:
- [CASA](https://casa.nrao.edu/casa_obtaining.shtml).
- For galario:
  - [galario](https://mtazzari.github.io/galario/). _(Note: On macOS with ARM chips, galario installation may require creating an x86 environment using Rosetta 2 and installing via Conda. In general, it is recommended to install galario via Conda, creating a dedicated environment with Python 3.8)_.
  - [uvplot](https://github.com/mtazzari/uvplot).
  - [emcee](https://emcee.readthedocs.io/en/stable/user/install/), [corner](https://corner.readthedocs.io/en/latest/install/), and [h5py](https://docs.h5py.org/en/latest/build.html).
- For frank:
  - [frank](https://github.com/discsim/frank).
  - [gofish](https://github.com/richteague/gofish).

This pipeline consists of four main steps, organized by folders and to be followed in order:
1. `data/` folder: contains a script to be run in a CASA terminal to average the original measurement set in frequency and time, and to export the uv tables for both galario and frank.
2. `galario_fit/` folder: scripts to run galario (either in 1D or 2D). The goal of this step is to retrieve the geometrical parameters of the disk (inclination, position angle, RA and Dec offsets between the disk center and the observation phase center). Given the nature of the galario fit, this step is by far the most time-consuming. However, it can be skipped if the geometrical parameters are already known from the literature or other methods.
3. `frank_fit/` folder: scripts to run the frank model, perform CLEANing, and generate the model and residual images.
4. `analysis/` folder: a suite of Python notebooks to carry out further analysis on the data and the obtained frank model fit.

# 1. From Measurement Set (MS) to uv tables

This pipeline was originally designed to process ALMA data only, averaged to one channel per spectral window. For a more general approach (different numbers of channels per spectral window and/or VLA data), use `ExportMS_split_general.py` instead of `ExportMS_split_contspw_1ch.py` and `ImportMS_general.py` instead of `ImportMS.py`


1. Create a source folder (here we consider `AA_Tau`). Within it, copy the `data`, `frank_fit` and `galario_fit` folders from this repository into it.
2. Place your starting continuum MS file (here `AA_Tau_time_ave_continuum.ms`, which can be downloaded from the public release of the [exoALMA calibrated datasets](https://bulk.cv.nrao.edu/exoalma/ms_public_release/)) in the `data` folder. The `data` folder should also contain the `ExportMS_split_contspw_1ch.py` script.
3. In the `ExportMS_split_contspw_1ch.py` script, select the right `target` name, `MS_filename` and spectral windows.
4. Open a CASA terminal in the `data` folder and execute:
   ```
   execfile('ExportMS_split_contspw_1ch.py')
   ``` 
   This script:
   1. Produces `AA_Tau_continuum.ms` with continuum spws averaged to 1 channel and 30s bins in time.
   2. Generates uv tables for frank and galario (`AA_Tau_continuum.vis.npz` for frank and `AA_Tau_galario_uvtable.txt` for galario)


# 2. galario [(Tazzari et al. 2018)](https://ui.adsabs.harvard.edu/abs/2018MNRAS.476.4527T/abstract)

Check the [galario documentation](https://mtazzari.github.io/galario/) for details (in particular the 'Getting started' tutorial).

## galario 1D 

For mostly axisymmetric sources, use a 1D profile model. Within the `galario_fit` folder:

1. Add your parametric model to `galario_1D_model_functions.py`, if not already included.
2. Adjust the fitting parameters in the “MODIFY HERE” section of `galario_fit_1D.py`. Here, you can choose your model, number of walkers and steps, radial grid, initial guess for the model parameters, and priors.
3. Run the fit:
   ```
   python galario_fit_1D.py
   ```
4. Experiment with multiple runs for convergence.
5. You can easily plot the results and save the galario model in txt files from a stored backend using the `plot_results_save_model_galario_fit_1D.py` script, after properly fixing the “MODIFY HERE” section:
   ```
   python plot_results_save_model_galario_fit_1D.py
   ```
   Results will be saved in the `products` folder.


## galario 2D

For highly non-axisymmetric sources (here, `CQ Tau`), use a 2D model. Within the `galario_fit` folder:

1. Add your parametric model to `galario_2D_model_functions.py`, if not already included.
2. Optionally, you can visually check your 2D model with the `prepare_model_galario_2D.ipynb` notebook.
3. Adjust parameters in the “MODIFY HERE” section of `galario_fit_2D.py`. Here, you can choose your model, number of walkers and steps, and priors. You can also choose whether to start from manually chosen initial positions of the walkers or draw the initial positions of the walkers from a uniform distribution over the intervals set by the priors (this approach thoroughly explores all potential values, which is particularly beneficial when dealing with numerous fitting parameters). The choice is made with the `uniform_pos` boolean variable.
4. Run the fit:
   ```
   python galario_fit_2D.py
   ```
5. Experiment with multiple runs for convergence.
6. You can easily plot the results and save the galario model in txt files from a stored backend using the `plot_results_save_model_galario_fit_2D.py` script, after properly fixing the “MODIFY HERE” section:
   ```
   python plot_results_save_model_galario_fit_2D.py
   ```
   Results will be saved in the `products` folder.

# 3. frank [(Jennings et al. 2020)](https://ui.adsabs.harvard.edu/abs/2020MNRAS.495.3209J/abstract)

Refer to the [frank documentation](https://discsim.github.io/frank/) for detailed guidance. 

Within the `frank_fit` folder you will find:
- `data_imaging.py` function to CLEAN the data. Info on the image is saved in `CLEAN\robust{robust}\Info_image_data_{target}_robust0.5.txt`.
- `diskdictionary.py` file to store parameters of the disk, the CLEANing and the frank fit.
- `model_imaging.py` function to CLEAN the frank model visibilities as if they were data, using the same uv locations of the observation.
- `resid_imaging` function to CLEAN the residuals obtained by subtracting the frank fit from the observed data.
- `ImportMS.py` function called by `model_imaging.py` and `resid_imaging.py` to produce the model and residuals MS tables (`frank_fit/CLEAN/AA_Tau_continuum.model.ms` and `frank_fit/CLEAN/AA_Tau_continuum.resid.ms`) from `data/AA_Tau_continuum.ms`
- `run_CLEAN_frank.py` is the main file. It performs the frank fit, saving the results in the `frank_fit/fits` folder, and runs the CLEANing of the data, the frank model, and the residuals, saving the images in the `frank_fit/figs` folder.

There are also two extra folders that are meant to perform further tests on the frank fit (and therefore are not necessary to run):
- `frank_fit_bootstrap/` containing the `frank_bootstrap.py` file, which runs many iterations of the frank fit varying the geometrical parameters in order to provide a reasonable estimate of the uncertainty on the frank profile. The results of the bootstrap are read by `read_results.py`, which can be used to estimate uncertainties on the radial extent (R68, R90, R95) of the dust continuum disk.
- `frank_fit_test_hyperparameters/` containing the `run_frank_test_hyperparameters.py` file, which runs fits using various values of the frank hyperparameters to test their effect on the results.


How to proceed:
1. Modify the `diskdictionary.py` file with the right values of `incl`, `PA`, `dx`, `dy` from the galario fit (or your favorite method) and all the other parameters used by frank and CLEAN. The script will automatically create subdirectories for each robust parameter used (`crobust`, remember also to modify accordingly the `RMS` value when changing the robust).
2. Modify the `run_CLEAN_frank.py` in the first lines with:
   1. The right `target` name.
   2. The plot settings you prefer (`img_lim` and `index_ticks`).
   3. What you want to run. Specifically, write `True` or `False` to run or not the following:
      1. `im_dat` to CLEAN the observed data.
      2. `frank`  to perform the frank fit.
      3. `im_res` to CLEAN the residuals (observed data - frank model).
      4. `annotate_res` to annotate the residual image (saved in the `figs` folder) with the geometrical parameters that have been used and possible ellipses.
      5. `im_mdl` to CLEAN the frank model.
3. Run everything with
   ```
   python run_CLEAN_frank.py
   ```
4. You can create summary plots comparing observed data and frank model with the `plot_images.ipynb` notebook.

# 3.5 CLEAN images of galario model and residuals

It is possible to obtain CLEAN images of the galario model and residuals in a way fully analogous to the frank CLEAN pipeline.

Steps:
1. Go to the `galario_fit/galario_mod_res_CLEAN/` folder.
2. Modify `run_CLEAN_galario.py` in the same way you would modify `frank_fit/run_CLEAN_frank.py`.  
   The script will read the CLEAN and geometry parameters from `frank_fit/diskdictionary.py` to ensure consistency between frank and galario imaging.
3. Run the pipeline:
   ```
   python run_CLEAN_galario.py
   ```

# 4. Analysis
Within the `analysis` folder, the following Python notebooks are available to perform further analyses on the data and on the output of the frank fit:
- `visibility_plot_frank_fit.ipynb`: Plots the real and imaginary part of the visibilities as a function of the deprojected baseline, for both the data and the frank fit.
- `polar_plots.ipynb`: Produces polar plots (radius vs. azimuthal angle) of the observed data and the frank residuals.
- `intensity_profile_CLEAN.ipynb`: Computes the azimuthally averaged intensity radial profile of the CLEAN image.
- `intensity_profile_radii_frank_fit.ipynb`: Extracts the intensity radial profile of the frank model and estimates the continuum radial extents (R68, R90, R95).
- `define_annular_substructures_frank_fit.ipynb`: Identifies and characterizes annular substructures (rings and gaps) from the frank fit (see Section 4.1 of the exoALMA IV paper).
- `compute_flux_density_with_uncertainty.ipynb`: Measures the integrated flux density of the disk using a mask, and estimates its statistical uncertainty by calculating the rms of the flux density inside several non-overlapping masks placed away from the disk, in regions containing only noise.
- `compute_NAI.ipynb`: Computes the nonaxisymmetry index (NAI) of the source (see Section 4.2 of the exoALMA IV paper).
- `compute_lambda_out.ipynb`: Determines $\lambda_\mathrm{out}$, the scale length of the faint outer disk emission falloff (see Section 5.4 of the exoALMA IV paper).



### Acknowledgments
I would like to thank the collaborators who reported bugs or whose discussions helped improve the pipeline: Daniele Fasano, Carolina Agurto, Andrés Zuleta, and Octave Mullie.
