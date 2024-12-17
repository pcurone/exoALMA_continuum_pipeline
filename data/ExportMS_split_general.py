# To be run inside a CASA terminal

import os
import numpy as np 

target = 'PDS_66_noave'

MS_filename = f'{target}_time_ave_continuum'
listobs(vis=MS_filename+'.ms', listfile=MS_filename+'.ms.txt', overwrite=True) 

contspws = ' '   # Select the spws you're interested here
split(vis=MS_filename+'.ms', 
      width=8,
      spw=contspws,
      timebin='30s',
      datacolumn='data',
      outputvis=target+'_continuum.ms', 
      keepflags=False)

listobs(vis=target+'_continuum.ms', listfile=target+'_continuum.ms.txt', overwrite=True) 

###################################################
#####   Export uvtable for galario and frank  #####
###################################################

# Get the data tables out of the MS file
# Here mixing tb and ms functions to consider all spws which can have different number of channels
# This script can be used for data with both 2 and 4 polarizations (usually, ALMA and VLA data, respectively)

light_speed = 2.99792458e8 # m/s

# Get info on spw
ms.open(target+'_continuum.ms')
ms.selectinit(reset=True)
spw_info = ms.getspectralwindowinfo()
ms.close()

# Get the total number of spws
tb.open(target+'_continuum.ms')
spwid = tb.getcol("DATA_DESC_ID")
tot_num_spw = int(np.amax(spwid)+1)
tb.close()

# Extract the frequency for each channel
tb.open(target+'_continuum.ms'+'/SPECTRAL_WINDOW')
freq_chan_allspw = tb.getvarcol("CHAN_FREQ")
tb.close()

u_lambda_tot = np.array([])
v_lambda_tot = np.array([])
Re_vis_tot = np.array([])
Im_vis_tot = np.array([])
weight_tot = np.array([])
wavelength_tot = np.array([])

for ispw in range(tot_num_spw):

    #Extract info from the specific spw
    ms.open(target+'_continuum.ms')
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
    ms.close()

    # Treat data from each single channel
    for ichan in range(nchan):

        # Remove any lingering flagged columns.
        # Since keepflags=False in split only works when an entire row is completely flagged, it is possible (though hopefully rare) for some rows to be still partially flagged.
        # Here, we select only the rows where no data is flagged. Later, when creating the new model and residual MS tables (in the ImportMS.py file), 
        # we will inject the model or residual visibilities only in the right fully unflagged rows. 
        # This means that the partially flagged data from the observation will remain in the model or residual ms,
        # but partially flagged data are ignored when running tclean with stokes='I', so this won't be a problem.
        good = np.squeeze(np.any(flag[:,ichan, :], axis=0) == False)
        real_chan = real[:,ichan, good]
        imag_chan = imag[:,ichan, good]
        um_chan = um[good]
        vm_chan = vm[good]
        weight_chan = weight[:,good]

        # Average the polarizations
        # For ALMA data with XX and YY polarizations
        if npol==2:
            vis_XX = real_chan[0,:] + 1j*imag_chan[0,:]
            vis_YY = real_chan[1,:] + 1j*imag_chan[1,:]
            We_XX = weight_chan[0,:]
            We_YY = weight_chan[1,:]
            vis_chan = (vis_XX * We_XX + vis_YY * We_YY) / (We_XX + We_YY)
            wgt_chan = We_XX + We_YY

        # For VLA data with RR, RL, LR, and LL polarizations
        # We are interested only in RR and LL, from which we compute the total intensity I = RR + LL.
        elif npol==4:
            vis_RR = real_chan[0,:] + 1j*imag_chan[0,:]
            vis_LL = real_chan[3,:] + 1j*imag_chan[3,:]
            We_RR = weight_chan[0,:]
            We_LL = weight_chan[3,:]
            vis_chan = (vis_RR * We_RR + vis_LL * We_LL) / (We_RR + We_LL)
            wgt_chan = We_RR + We_LL
        
        # Obtain u and v per unit wavelength for the specific channel
        freq_chan = freq_chan_allspw[f'r{ispw+1}'][ichan] # Hz
        #print(f'spw {ispw}, channel {ichan}: frequency {freq_chan} Hz')
        wavelength_chan = light_speed / freq_chan 
        u_lambda_chan = um_chan / wavelength_chan
        v_lambda_chan = vm_chan / wavelength_chan

        u_lambda_tot = np.concatenate((u_lambda_tot, u_lambda_chan))
        v_lambda_tot = np.concatenate((v_lambda_tot, v_lambda_chan))
        Re_vis_tot = np.concatenate((Re_vis_tot, vis_chan.real))
        Im_vis_tot = np.concatenate((Im_vis_tot, vis_chan.imag))
        weight_tot = np.concatenate((weight_tot, wgt_chan))
        wavelength_tot = np.concatenate((wavelength_tot, np.ones(u_lambda_chan.shape[0])*wavelength_chan))


# For galario
# Create the formatted header manually
header = (
    f"# Extracted from {target}_continuum.ms\n"
    f"# wavelength[m]\n"
    f"{np.mean(wavelength_tot)}\n"
    f"# Columns:\tu[wavelength]\tv[wavelength]\tRe(V)[Jy]\tIm(V)[Jy]\tweight"
)
# Save the data
np.savetxt(
    f"{target}_galario_uvtable.txt",
    np.column_stack([u_lambda_tot, v_lambda_tot, Re_vis_tot, Im_vis_tot, weight_tot]),
    fmt='%10.6e',
    delimiter='\t',
    header=header,
    comments=""  # Removes the default '#' added by savetxt to the header
)

# Output to npz file for frank
os.system('rm -rf '+target+'_continuum.vis.npz')
np.savez(target+'_continuum.vis', u=u_lambda_tot, v=v_lambda_tot, Vis=Re_vis_tot+1j*Im_vis_tot, Wgt=weight_tot)
print("# Measurement set exported to "+target+'_continuum.vis.npz')

