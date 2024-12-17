'''
Script to create model and residual ms files from a frank model.
This version is able to manage ms files with different number of channel per spw
and different number of polarizations.
In order to employ correctly this script, it is important that the uvtable used in frank has
been produced in a similar manner (so mixing the ms and tb CASA task to take into account
possible different number of channel per spw and different number of polarizations).

Don't worry if you get a warning about the ms.putdataold task:
with CASA version 6.6.1.17, the newer ms.putdata had some bugs
'''


import os
import numpy as np

def ImportMS(msfile, modelfile, suffix='model', make_resid=False):

    if make_resid:
        print(' ')
        print('Creating the residuals ms file')
        print(' ')
        suffix = 'resid'
    else:
        print(' ')
        print('Creating the model ms file')
        print(' ')

    # strip off the '.ms'
    MS_path = msfile
    MS_filename = f'CLEAN/{os.path.splitext(os.path.basename(MS_path))[0]}'

    # copy the data MS into a model MS
    os.system(f'rm -rf {MS_filename}.{suffix}.ms')
    os.system(f'cp -r {msfile} {MS_filename}.{suffix}.ms')

    light_speed = 2.99792458e8 # m/s

    # Get info on spw
    ms.open(f'{MS_filename}.{suffix}.ms')
    ms.selectinit(reset=True)
    spw_info = ms.getspectralwindowinfo()
    ms.close()

    # Get the total number of spws
    tb.open(f'{MS_filename}.{suffix}.ms')
    spwid = tb.getcol("DATA_DESC_ID")
    tot_num_spw = int(np.amax(spwid)+1)
    tb.close()

    # Extract the frequency for each channel
    tb.open(f'{MS_filename}.{suffix}.ms/SPECTRAL_WINDOW')
    freq_chan_allspw = tb.getvarcol("CHAN_FREQ")
    tb.close()

    # Load the model visibilities and weight
    u_model = (np.load(modelfile+'.npz'))['u']
    v_model = (np.load(modelfile+'.npz'))['v']
    vis_model = (np.load(modelfile+'.npz'))['V']
    Re_model = vis_model.real
    Im_model = vis_model.imag
    weight_model = (np.load(modelfile+'.npz'))['weights']

    index_in = 0

    for ispw in range(tot_num_spw):

        #Extract info from the specific spw
        ms.open(f'{MS_filename}.{suffix}.ms', nomodify=False)
        ms.selectinit(reset=True)
        ms.selectinit(datadescid=ispw)
        totdata = ms.getdata(['data','u','v','weight','flag'])
        real = totdata['data'].real
        imag = totdata['data'].imag
        npol = spw_info[f'{ispw}']['NumCorr']
        nchan = spw_info[f'{ispw}']['NumChan']
        flag = totdata['flag']
        um = totdata['u']
        vm = totdata['v']
        weight = totdata['weight']

        # Delete unwanted keys from totdata so you don't get warnings when using ms.putdata
        del totdata['u']
        del totdata['v']
        del totdata['flag']

        # Treat data from each single channel
        for ichan in range(nchan):

            # Remove any lingering flagged columns.
            good = np.squeeze(np.any(flag[:,ichan, :], axis=0) == False)
            um_chan = um[good]
            vm_chan = vm[good]
            
            # Obtain u and v per unit wavelength for the specific channel
            freq_chan = freq_chan_allspw[f'r{ispw+1}'][ichan] # Hz
            wavelength_chan = light_speed / freq_chan 
            u_lambda_chan = um_chan / wavelength_chan
            v_lambda_chan = vm_chan / wavelength_chan

            # Select the correct rows from the model 
            nrows = um_chan.shape[0]
            Re_model_chan = Re_model[index_in:index_in+nrows]
            Im_model_chan = Im_model[index_in:index_in+nrows]
            Vis_model_chan = Re_model_chan + 1j*Im_model_chan
            weight_model_chan = weight_model[index_in:index_in+nrows]
            u_model_chan = u_model[index_in:index_in+nrows]
            v_model_chan = v_model[index_in:index_in+nrows]
            
            # Check if you're selecting the correct rows
            if not (np.array_equal(u_lambda_chan, u_model_chan) and np.array_equal(v_lambda_chan, v_model_chan)):
                raise ValueError("Selecting the wrong rows, u and v coordinates do not coincide between data and model!")
            
            # Replace with the model visibilities (equal in all polarizations)
            if make_resid:
                totdata['data'][:,ichan, good] -= Vis_model_chan
                totdata['weight'][:,good] = weight_model_chan / 2
            else: 
                totdata['data'][:,ichan, good] = Vis_model_chan
                totdata['weight'][:,good] = weight_model_chan / 2

            # Re-pack those model visibilities back into the .ms file
            ms.putdataold(totdata)     #### the new ms.putdata() has bugs with version 6.6.1.17 !!!!!! 

            # Update the index for counting the rows
            index_in = index_in + nrows

        # Close ms at the end of each spw
        ms.close()

