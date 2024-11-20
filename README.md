Pipeline to perform frank fits initially inspired by the procedure used by Sean Andrews in Andrews et al. 2021. 

Expanded for the continuum analysis of the exoALMA sources (exoALMA IV, Curone and exoALMA) including galario fits for getting the geometrical parameters.

Codes required to be installed (apart from all the usual python packages):
- CASA (https://casa.nrao.edu/casa_obtaining.shtml).
- galario (https://mtazzari.github.io/galario/). _Problems to install it on an Apple machine with ARM Chip, so I ended up creating an x86 environment using Rosetta 2 where I installed galario through the usual conda.
- uvplot (https://github.com/mtazzari/uvplot).
- emcee (https://emcee.readthedocs.io/en/stable/user/install/) and corner (https://corner.readthedocs.io/en/latest/install/).
- frank (https://github.com/discsim/frank).
- gofish (https://github.com/richteague/gofish).



# From ms file to uv tables

Works only for ALMA data, not for VLA data (due to different polarization settings)

1. Create a source folder (e.g., `AA_Tau`). Within it, copy the `frank_fit` and `galario_fit` you find here.
2. Download your starting continuum ms file (for instance `AA_Tau_time_ave_continuum.ms`) and place it in `frank_fit/data`, where you will also find `ExportMS_split_contspw_1ch.py`.
3. Within `frank_fit/data`, use `ExportMS_split_contspw_1ch.py` to:
   1. Produce `AA_Tau_continuum.ms` with continuum spws averaged in frequency to 1 channel and in time to 30s bins.
   2. Generate uv tables for frank and galario (`AA_Tau_continuum.vis.npz` for frank and `AA_Tau_galario_uvtable.txt` for galario to be moved to the `galario_fit` folder)

# galario (to be updated)

Look here for the galario documentation and the 'Getting started' tutorial https://mtazzari.github.io/galario/

## galario 1D 

In the case of mostly axisymmetric sources, use a 1D profile as the model. Within the `galario_fit` folder:

1. Prepare the python file (here `galario_fit_AA_Tau_exoALMA_PointSource_4rings.py`). Fix the `wle` parameter and the model with the right model emission for the specific source. Fix also the `nwalkers` and `nsteps`. 
2. Run galario:
   ```
   python galario_fit_AA_Tau_exoALMA_PointSource_4rings.py
   ```
3. Experiment with multiple runs for convergence.


## galario 2D

In the case of highly non-axisymmetric sources use a 2D model (here considering CQ Tau). Within the `galario_fit` folder:

1. Inspect the parameters of the 2D model using `model_galario_2D_new.ipynb`
2. Run galario with the file `galario_2D_fit_CQ_Tau_exoALMA_2rings_2arcs_uniform_pos.py`. The initial walker positions are randomly selected using a flat probability function across a wide range of values. This approach thoroughly explores all potential values, which is particularly beneficial when dealing with numerous fitting parameters.
3. After a first run, employ `galario_2D_fit_CQ_Tau_exoALMA_2rings_2arcs_2nd_step.py`. Here, the initial point `p0` is chosen with a small Gaussian variation based on the best-fit parameters obtained from the previous step.



# frank

You can check here the nice and detailed frank documentation https://discsim.github.io/frank/

Within the `frank_fit` folder you'll find:
- `data_imaging.py` function to CLEAN the data.
- `diskdictionary.py` file to store parameters of the disk, the CLEANing and the frank fit.
- `model_imaging.py` function to CLEAN the frank model visibilities as if they were data, using the same uv locations of the observation.
- `resid_imaging` function to CLEAN the residuals obtained by subtracting the frank fit from the observed data.
- `ImportMS.py` function called by `model_imaging.py` and `resid_imaging.py` to produce the model and residuals ms tables (`data/AA_Tau_continuum.model.ms` and `data/AA_Tau_continuum.resid.ms`) from `data/AA_Tau_continuum.ms`
- `run_CLEAN_frank.py` is the main file. It performs the frank fit, saving the results in the `fits` folder, and runs the CLEANing of the data, the frank model, and the residuals, saving the images in the `figs` folder.

How to proceed:
1. Modify the `diskdictionary.py` file with the right values of `incl`, `PA`, `dx`, `dy` from the galario fit (or your favorite method) and all the other parameters used by frank and CLEAN.
2. Modify the `run_CLEAN_frank.py` in the first lines with:
   1. The right `target` name.
   2. The plot settings you prefer (`img_lim` and `index_ticks`).
   3. What you want to run. Specifically, write `True` or `False` to run or not the following:
      1. `im_dat` to CLEAN the observed data.
      2. `frank`  to perform the frank fit.
      3. `im_res` to CLEAN the residuals (observed data - frank model).
      4. `annotate_res` to annotate the residual image (saved in the `figs` folder) with the geometrical parameters that have been used.
      5. `im_mdl` to CLEAN the frank model.
3. Run everything with
   ```
   python run_CLEAN_frank.py
   ```
4. You can create summary plots comparing observed data and frank model with the `plot_images.ipynb` notebook.
