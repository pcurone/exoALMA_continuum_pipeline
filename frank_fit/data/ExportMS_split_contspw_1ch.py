# To be run inside a CASA terminal

import os
import numpy as np

target = 'AA_Tau'

MS_filename = f'{target}_time_ave_continuum'
listobs(vis=MS_filename+'.ms', listfile=MS_filename+'.ms.txt', overwrite=True) 

contspws = '1,5,9,13,17,21,25,29,33,37,41,45'   # Select the spws you're interested here, in this case I'm selecting the continuum spws only
split(vis=MS_filename+'.ms', 
      width=8,
      spw=contspws,
      timebin='30s',
      datacolumn='data',
      outputvis=target+'_continuum.ms', 
      keepflags=False,
      timebin=None)

listobs(vis=target+'_continuum.ms', listfile=target+'_continuum.ms.txt', overwrite=True) 

#os.system('rm -rf ' + target+'_time_ave_continuum.ms')


###################################################
#####   Export uvtable for galario and frank  #####
###################################################

# get the data tables out of the MS file
tb.open(target+'_continuum.ms')
data = np.squeeze(tb.getcol("DATA"))
flag = np.squeeze(tb.getcol("FLAG"))
uvw = tb.getcol("UVW")
weight = tb.getcol("WEIGHT")
spwid = tb.getcol("DATA_DESC_ID")
times = tb.getcol("TIME")
tb.close()

# get frequency information
tb.open(target+'_continuum.ms/SPECTRAL_WINDOW')
freqlist = np.squeeze(tb.getcol("CHAN_FREQ"))
tb.close()

# get rid of any lingering flagged columns (but there shouldn't be any!)
good = np.squeeze(np.any(flag, axis=0) == False)
data = data[:,good]
weight = weight[:,good]
uvw = uvw[:,good]
spwid = spwid[good]
times = times[good]

# average the polarizations
Re = np.sum(data.real * weight, axis=0) / np.sum(weight, axis=0)
Im = np.sum(data.imag * weight, axis=0) / np.sum(weight, axis=0)
vis = Re + 1j*Im
wgt = np.sum(weight, axis=0)

# associate each datapoint with a frequency
get_freq = lambda ispw: freqlist[ispw]
freqs = get_freq(spwid)

# retrieve (u,v) positions in meters
um = uvw[0,:]
vm = uvw[1,:]
wm = uvw[2,:]
u, v = um * freqs / 2.9979e8, vm * freqs / 2.9979e8

clight = 299792458
wle = clight / freqs.mean()  # [m]


# For galario
np.savetxt(target+'_galario_uvtable.txt',
           np.column_stack([um, vm, vis.real, vis.imag, wgt]),
           fmt='%10.6e', delimiter='\t',
           header='Extracted from {}.\nwavelength[m] = {}\nColumns:\tu[m]\tv[m]\tRe(V)[Jy]\tIm(V)[Jy]\tweight'.format(target+'_continuum.ms', wle))
print("# Measurement set exported to "+target+'_galario_uvtable.txt')
os.system(f'mv {target}_galario_uvtable.txt  ../../galario_fit')

# output to npz file for frank
os.system('rm -rf '+target+'_continuum.vis.npz')
np.savez(target+'_continuum.vis', u=u, v=v, Vis=vis, Wgt=wgt)
print("# Measurement set exported to "+target+'_continuum.vis.npz')