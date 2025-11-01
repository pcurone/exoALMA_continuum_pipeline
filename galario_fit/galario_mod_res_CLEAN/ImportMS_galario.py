import os
import numpy as np

def ImportMS_galario(msfile, modelfile, suffix='model', make_resid=False):

    if make_resid:
        suffix = 'resid'

    # strip off the '.ms'
    MS_path = msfile
    MS_filename = f'CLEAN/{os.path.splitext(os.path.basename(MS_path))[0]}'

    # copy the data MS into a model MS
    os.system(f'rm -rf {MS_filename}.{suffix}.ms')
    os.system(f'cp -r {msfile} {MS_filename}.{suffix}.ms')

    # open the model file and load the data
    tb.open(MS_filename+'.'+suffix+'.ms')
    data = tb.getcol("DATA")
    flag = tb.getcol("FLAG")
    tb.close()

    # identify the unflagged columns (should be all of them!)
    unflagged = np.squeeze(np.any(flag, axis=0) == False)

    # load the model visibilities
    galario_model = np.loadtxt(modelfile)
    mdl = galario_model[:,2] + 1j*galario_model[:,3]

    # replace with the model visibilities (equal in both polarizations)
    if make_resid:
        data[:, :, unflagged] -= mdl
    else:
        data[:, :, unflagged] = mdl

    # re-pack those model visibilities back into the .ms file
    tb.open(MS_filename+'.'+suffix+'.ms', nomodify=False)
    tb.putcol("DATA", data)
    tb.flush()
    tb.close()
