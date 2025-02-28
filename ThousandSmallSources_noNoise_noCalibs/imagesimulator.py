from numpy import *
import numpy as np
import pyfits
import os
import pylab as plt


x = np.tile(np.arange(0,2048),(2048,1))
y=np.tile(np.arange(0,2048).reshape(2048,1),(1,2048))
Flux = np.zeros((2048,2048))
sigma=0.58


x_c = np.random.uniform(50,2000,size=1000)
y_c = np.random.uniform(50,2000,size=1000)


#for j in np.arange(1,10):
   #for i in np.arange(1,100):
  #x_c=10*i
  #y_c=200*j
for i in range(x_c.size):
  print i
  ind = np.logical_and(np.logical_and(x > (x_c[i] - 50), x < (x_c[i] + 50)), np.logical_and(y > (y_c[i] - 50),y < (y_c[i] + 50)))
  xcomp = (x[ind] - x_c[i])**2.
  ycomp = (y[ind] -y_c[i])**2.
  zcomp = -0.5*((xcomp + ycomp)/(sigma**2.))
  Flux[ind] =Flux[ind]+ 8.*(1./(2.*np.pi))*(1./(sigma**2.))*np.exp(zcomp)

bg =  0.003 * np.random.randn(Flux.shape[0], Flux.shape[1])
Flux = Flux + bg
print (np.sum(Flux))
header=pyfits.getheader('/local/lydia/Data/all_SWASP_data/raw/DAS1_001823758.fts')

if not os.path.exists("input"):
 os.makedirs("input")

pyfits.writeto("./input/sim_v.fits",Flux,header,clobber=True )
