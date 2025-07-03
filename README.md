This is the pipeline used to analyze the continuum emission of the [exoALMA Large Program](https://www.exoalma.com), with results presented in [Curone et al. 2025 (exoALMA IV)](https://ui.adsabs.harvard.edu/abs/2025ApJ...984L...9C/abstract).

This pipeline was initially inspired by the procedure developed in  [Andrews et al. 2021](https://ui.adsabs.harvard.edu/abs/2021ApJ...916...51A/abstract) ([GitHub repo](https://github.com/seanandrews/DSHARP_CPDs)).

In addition to standard Python packages, you’ll need the following tools installed:
- [CASA](https://casa.nrao.edu/casa_obtaining.shtml).
- For galario:
  - [galario](https://mtazzari.github.io/galario/). _(Note: On macOS with ARM chips, galario installation may require creating an x86 environment using Rosetta 2 and installing via Conda)_.
  - [uvplot](https://github.com/mtazzari/uvplot).
  - [emcee](https://emcee.readthedocs.io/en/stable/user/install/), [corner](https://corner.readthedocs.io/en/latest/install/), and [h5py](https://docs.h5py.org/en/latest/build.html).
- For frank:
  - [frank](https://github.com/discsim/frank).
  - [gofish](https://github.com/richteague/gofish).



# From Measurement Set (MS) to uv tables

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


# galario [(Tazzari et al. 2018)](https://ui.adsabs.harvard.edu/abs/2018MNRAS.476.4527T/abstract)

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
5. You can simply plot the results from a stored backend using the `plot_results_galario_fit_1D.py`, after properly fixing the “MODIFY HERE” section:
   ```
   python plot_results_galario_fit_1D.py
   ```


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
6. You can simply plot the results from a stored backend with:
   ```
   python plot_results_galario_fit_2D.py
   ```

# frank [(Jennings et al. 2020)](https://ui.adsabs.harvard.edu/abs/2020MNRAS.495.3209J/abstract)

Refer to the [frank documentation](https://discsim.github.io/frank/) for detailed guidance. 

Within the `frank_fit` folder you will find:
- `data_imaging.py` function to CLEAN the data.
- `diskdictionary.py` file to store parameters of the disk, the CLEANing and the frank fit.
- `model_imaging.py` function to CLEAN the frank model visibilities as if they were data, using the same uv locations of the observation.
- `resid_imaging` function to CLEAN the residuals obtained by subtracting the frank fit from the observed data.
- `ImportMS.py` function called by `model_imaging.py` and `resid_imaging.py` to produce the model and residuals MS tables (`frank_fit/CLEAN/AA_Tau_continuum.model.ms` and `frank_fit/CLEAN/AA_Tau_continuum.resid.ms`) from `data/AA_Tau_continuum.ms`
- `run_CLEAN_frank.py` is the main file. It performs the frank fit, saving the results in the `frank_fit/fits` folder, and runs the CLEANing of the data, the frank model, and the residuals, saving the images in the `frank_fit/figs` folder.

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
