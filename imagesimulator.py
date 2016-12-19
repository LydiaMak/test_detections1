from numpy import *
import numpy as np
import pyfits
import os
import pylab as plt


x = np.tile(np.arange(0,2048),(2048,1))
y=np.tile(np.arange(0,2048).reshape(2048,1),(1,2048))
Flux = np.zeros((2048,2048))
sigma=0.58
x_c = np.random.uniform(50, 2000, size=50)
y_c = np.random.uniform(50, 2000, size=50)
for i in range(x_c.size):
 print i   
 xcomp = (x - x_c[i])**2.
 ycomp = (y -y_c[i])**2.
 zcomp = -0.5*((xcomp + ycomp)/(sigma**2.))
 Flux =Flux+ 8.*(1./(2.*np.pi))*(1./(sigma**2.))*np.exp(zcomp)

bg =  0.003 * np.random.randn(Flux.shape[0], Flux.shape[1]) + 0.18
Flux = Flux + bg
print (np.sum(Flux))
hdu=pyfits.PrimaryHDU(Flux)
hdulist=pyfits.HDUList([hdu])
hdulist.writeto("sim_one_star.fits",clobber=True)
