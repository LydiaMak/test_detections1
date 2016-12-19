from numpy import *
import numpy as np
import pyfits
import os
import pylab as plt

t = pyfits.open('/local/lydia/Data/all_SWASP_data/test_detections/OriginalData/wDeblend_00001_1000/sci-results/src/src.fits')

sources = t[1].data
x = np.tile(np.arange(0,2048),(2048,1))
y=np.tile(np.arange(0,2048).reshape(2048,1),(1,2048))
Flux = np.zeros((2048,2048))
sigma=0.58

f = sources['base_PsfFlux_flux']
#f = f[0:9999]
#x_c = np.random.uniform(50,2000,size=f.size)
#y_c = np.random.uniform(50,2000,size=f.size)

x_c = sources['base_SdssCentroid_x']
y_c = sources['base_SdssCentroid_y']

deblend = sources['deblend_nChild']
#for j in np.arange(1,10):
   #for i in np.arange(1,100):
  #x_c=10*i
  #y_c=200*j
for i in range(x_c.size):
  if deblend[i]==0:
     print i
     ind = np.logical_and(np.logical_and(x > (x_c[i] - 50), x < (x_c[i] + 50)), np.logical_and(y > (y_c[i] - 50),y < (y_c[i] + 50)))
     xcomp = (x[ind] - x_c[i])**2.
     ycomp = (y[ind] -y_c[i])**2.
     zcomp = -0.5*((xcomp + ycomp)/(sigma**2.))
     Flux[ind] =Flux[ind]+ f[i]*(1./(2.*np.pi))*(1./(sigma**2.))*np.exp(zcomp)

bg =  0.003 * np.random.randn(Flux.shape[0], Flux.shape[1])
Flux = Flux + bg
print (np.sum(Flux))
header=pyfits.getheader('/local/lydia/Data/all_SWASP_data/raw/DAS1_001823758.fts')

if not os.path.exists("input"):
 os.makedirs("input")

pyfits.writeto("./input/sim_v.fits",Flux,header,clobber=True )
np.savetxt('fluxes.txt', np.c_[x_c,y_c,f])
