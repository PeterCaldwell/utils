#!/usr/bin/env python
"""
This code loops over all variables in a netcdf file and
notes variables that differ, then plot differences for those variables.

Note: you need to 'module load python' for this to work on NERSC machines
(in order to access netCDF4).
"""

from netCDF4 import Dataset
import pylab as pl
import numpy as np

#OPEN OLD AND NEW FILES (THESE SHOULD PROBABLY BE MADE INPUT ARGUMENTS?):
#============================================================
fo=Dataset('/project/projectdirs/acme/inputdata/atm/cam/solar/Solar_1850control_input4MIPS_c20171101.nc')
fn=Dataset('/project/projectdirs/acme/inputdata/atm/cam/solar/Solar_1950control_input4MIPS_c20171208.nc')

#LOOP OVER VARIABLES:
#============================
for var in fo.variables.keys():
    try:
        df=np.sum(np.abs(fo.variables[var][:]-fn.variables[var][:]))
    except:
        print var+' in orig not found in new'

    #TEST EACH VARIABLE:
    #=====================================
    if df<1e-16:
        print var+' is identical between old and new files.'
    else:
        shp=fo.variables[var].shape
        #if fits on line plot:
        if len(shp)==1: 
            pl.figure()
            pl.plot(fo.variables[var][:],'b-')
            pl.plot(fn.variables[var][:],'r--')
            pl.title(var+' b=orig, r=new')
        #if fits in pcolor
        elif len(shp)==2:
            pl.figure()
            pl.subplot(2,1,1)
            pl.pcolor(fo.variables[var][:])
            pl.title(var+' orig')
            pl.colorbar()
            pl.subplot(2,1,2)
            pl.pcolor(fn.variables[var][:])
            pl.title(var+' new')
            pl.colorbar()
        else:
            print var+' differs, but has too many dims to plot easily'

pl.show()
