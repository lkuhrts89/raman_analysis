# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 09:44:52 2020

@author: luku438c

pass to console python3.7 script name path file

Fit kinetics of calcite to cerrusite transformation from renshaw in situ raman data
"""
#%%
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import sys
import os


# call script from shell with parameter
path = sys.argv[1]
name = sys.argv[2]
ext = 'txt'
print(sys.argv[1]) 
print(sys.argv[2])


# Create Output directory
save_path = path+'/'+'fits_'+name

try:
    os.mkdir(save_path)
except OSError:
    print ("Creation of the directory %s failed" % save_path)
else:
    print ("Successfully created the directory %s " % save_path)

# Inlcude pcov information in fit files and make residue plot
# plot options as function




def cut_data_at_boundary(data, ref_col, boundary):
    """Get the index of min and max borders 
    of data_x numpy array that is closest to the function min max. Returns cut 2d dataset"""
    data_ref = data[ref_col]
    idx_start = (np.abs(data_ref-boundary[0])).argmin() # get index of min_x
    idx_end = (np.abs(data_ref-boundary[1])).argmin() # get index of max_x
    ndata_x = data[:,idx_end:idx_start] # cut dataset at min_x and max_x
    return ndata_x

def lorentzian( x, x0, a, gam ):
    return a * gam**2 / ( gam**2 + ( x - x0 )**2)

def fit_lorentzian(data, guess, ref_col, fit_boundary):
    '''
    1: Takes a 2d numpy array as input. 2: make a guess for (x0,a,gamma), 
    3: Give x values of ref_col to use fit boundary as tuple
    '''
    # test if guess is in fit boundary
    data = cut_data_at_boundary(data,ref_col,fit_boundary) # cut the dataset at boundaries
    print(data)
    popt, pcov = curve_fit(lorentzian, data[0], data[1], p0=(1078,40000,10)) # fit lorentzian with initial parametrs
    print(popt)
    fitted_curve = lorentzian(data[0],*popt) # generate fit function
    return fitted_curve, popt, pcov


#data = pd.read_csv('%s/%s%s'%(path,name,ext), sep='\t',header=0)
data = np.genfromtxt('%s/%s.%s'%(path,name,ext), delimiter='\t', skip_header=1, unpack=True, usecols=(0,1,2))

"""Slice data set"""
scan_cols = np.unique(data[0],return_counts=False)
slicer = int(len(data[0])/len(scan_cols))


#%%
"""Create Pandas dataframe with time as index column, data and fit values"""
data_matrix = pd.DataFrame(columns=['Time / s',
                                    'Raman Data',
                                    'Calcite popt x0',
                                    'Calcite pcov x0',
                                    'Calcite popt a',
                                    'Calcite pcov a',
                                    'Calcite popt gam',
                                    'Calcite pcov gam',
                                    'Calcite global max',
                                    'Cerrusite popt x0'
                                    'Cerrusite pcov x0',
                                    'Cerrusite popt a',
                                    'Cerrusite pcov a',
                                    'Cerrusite popt gam',
                                    'Cerrusite pcov gam',
                                    'Cerrusite global max',                      
                                    ])

counter = 1
while counter <= len(data[0]):
    data_slice = data[:,counter:slicer+counter-1]
    cerrusite = cut_data_at_boundary(data_slice,1,(1020,1060))
    calcite = cut_data_at_boundary(data_slice,1,(1070,1095))
    max_cerrusite = np.amax(cerrusite[2])
    max_calcite = np.amax(calcite[2])
    p0_cerrusite = (1040,2000,10)
    try:
        cerrusite_popt, cerrusite_pcov = curve_fit(lorentzian,cerrusite[1],cerrusite[2],p0=p0_cerrusite) # get max of cerrusite peak
        p0_cerrusite = cerrusite_popt
    except:
        pass
    p0_calcite = (1090,1000,2)
    try:
        calcite_popt, calcite_pcov = curve_fit(lorentzian,calcite[1],calcite[2],p0=p0_calcite) # get max of cerrusite peak
        p0_calcite = calcite_popt
    except:
        pass
    plt.figure()
    plt.subplot(1,2,1)
    plt.subplots_adjust(wspace=0.6)
    plt.xlabel('Raman shift / 1/cm')
    plt.ylabel('Intensity / a.u.')
    plt.ylim((0,3000))
    plt.plot(cerrusite[1],cerrusite[2],'ro',fillstyle='none')
    plt.plot(cerrusite[1],lorentzian(cerrusite[1],*cerrusite_popt))
    plt.subplot(1,2,2)
    plt.ylim((0,1000))
    plt.xlabel('Raman shift / 1/cm')
    plt.ylabel('Intensity / a.u.')
    plt.plot(calcite[1],calcite[2],'ro',fillstyle='none')
    plt.plot(calcite[1],lorentzian(calcite[1],*calcite_popt))
    plt.savefig('%s/%s_%03d.png'%(save_path,name,counter/slicer),dpi=100)
    plt.close()
    t = data_slice[0,0]
    data_matrix = data_matrix.append({'Time / s' : t ,
                                     'Raman Data' : data_slice[1:3],
                                     'Calcite popt x0' : calcite_popt[0],
                                     'Calcite pcov x0':calcite_pcov[0],
                                     'Calcite popt a' : calcite_popt[1],
                                     'Calcite pcov a' : calcite_pcov[1],
                                     'Calcite popt gam' : calcite_popt[2],
                                     'Calcite pcov gam' : calcite_pcov[2],
                                     'Calcite global max': max_calcite,
                                     'Cerrusite popt x0' : cerrusite_popt[0],
                                     'Cerrusite pcov x0' : cerrusite_pcov[0],
                                     'Cerrusite popt a' : cerrusite_popt[1],
                                     'Cerrusite pcov a' : cerrusite_pcov[1],
                                     'Cerrusite popt gam' : cerrusite_popt[2],
                                     'Cerrusite pcov gam' : cerrusite_pcov[2],
                                     'Cerrusite global max': max_cerrusite,     
                                     },
                                     ignore_index=True)
    print('%1.0f %% fitted'%(counter/len(data[0])*100))
    counter = counter + slicer
print(data_matrix)
#%%

print('Creating plot of time-dependent amplitudes values')
amplitudes = data_matrix[['Time / s','Calcite popt a','Cerrusite popt a']]
amplitudes.set_index('Time / s')
amplitudes.plot(x='Time / s',style='o-',fillstyle='none',loglog=True); plt.legend(loc='best');plt.savefig('%s/a_amplitudes_%s.png'%(save_path,name),dpi=600)

# %%

print('Creating plot of time-dependent peak positions values')
peak_pos = data_matrix[['Time / s','Calcite popt x0','Cerrusite popt x0']]
peak_pos.set_index('Time / s')
peak_pos.plot(x='Time / s',style='o-',fillstyle='none',loglog=True); plt.ylim((1020,1100)); plt.legend(loc='best');plt.savefig('%s/a_peak_pos_%s.png'%(save_path,name),dpi=600)

# %%

print('Creating plot of time-dependent gamma values')
gamma = data_matrix[['Time / s','Calcite popt gam','Cerrusite popt gam']]
gamma.set_index('Time / s')
gamma.plot(x='Time / s',style='o-',fillstyle='none',loglog=True); plt.ylabel('Peak Width / cm^-1'); plt.legend(loc='best');plt.savefig('%s/gamma_%s.png'%(save_path,name),dpi=600)

# %%

print('Creating plot of time-dependent maximum intensities')
gamma = data_matrix[['Time / s','Calcite global max','Cerrusite global max']]
gamma.set_index('Time / s')
gamma.plot(x='Time / s',style='o-',fillstyle='none',loglog=True); plt.ylabel('Max Intensity / cm^-1'); plt.legend(loc='best');plt.savefig('%s/max_intensity_%s.png'%(save_path,name),dpi=600)
