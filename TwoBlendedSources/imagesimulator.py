from numpy import *
import numpy as np
import pyfits
import os
import pylab as plt


x = np.tile(np.arange(0,2048),(2048,1))
y=np.tile(np.arange(0,2048).reshape(2048,1),(1,2048))
Flux = np.zeros((2048,2048))
sigma=0.58

#fr = 5.*np.ones(20)
#xr_c = np.random.uniform(50,200,size=fr.size)
#yr_c = np.random.uniform(50,200,size=fr.size)

#deblend = sources['parent']

x_c = np.array([1000,995,300,2000])
y_c = np.array([1000,999,55,1500])
f = np.array([2.,2.,0.5,4])

#x_c = np.append(x_c, xr_c)
#y_c = np.append(y_c, yr_c)
#f = np.append(f, fr)

for i in range(x_c.size):
 # if deblend[i] == 0:
    print i
    ind = np.logical_and(np.logical_and(x > (x_c[i] - 50), x < (x_c[i] + 50)), np.logical_and(y > (y_c[i] - 50),y < (y_c[i] + 50)))
    xcomp = (x[ind] - x_c[i])**2.
    ycomp = (y[ind] -y_c[i])**2.
    zcomp = -0.5*((xcomp + ycomp)/(sigma**2.))
    Flux[ind] =Flux[ind]+ f[i]*(1./(2.*np.pi))*(1./(sigma**2.))*np.exp(zcomp)

bg =  0.003 * np.random.randn(Flux.shape[0], Flux.shape[1])+0.18
Flux = Flux + bg
print (np.sum(Flux))
print (Flux[0,0])
header=pyfits.getheader('/local/lydia/Data/all_SWASP_data/raw/DAS1_001823758.fts')

biasData = array([pyfits.getdata('/local/lydia/Data/all_SWASP_data/raw/DAS1_Bias_0018230%d.fts' % b) for b in range(66,76)])
bias = median(biasData,axis=0)
bias_header=pyfits.getheader('/local/lydia/Data/all_SWASP_data/raw/DAS1_Bias_001823066.fts')

darkData = array([pyfits.getdata('/local/lydia/Data/all_SWASP_data/raw/DAS1_Therm_0018240%d.fts' % da) for da in range(52,62)])
dark = median(darkData-bias,axis=0)
dark_header= pyfits.getheader('/local/lydia/Data/all_SWASP_data/raw/DAS1_Therm_001824052.fts')

flatData = array([pyfits.getdata('/local/lydia/Data/all_SWASP_data/raw/DAS1_Flat_00182%d.fts' % fl) for fl in range(3076,3101)+ range(4027,4052)])
flat = median(flatData-bias,axis=0)
flat_header=pyfits.getheader('/local/lydia/Data/all_SWASP_data/raw/DAS1_Flat_001823076.fts')

bias =bias[0:2048,0:2048]

dark=dark[0:2048,0:2048]

flat =flat[0:2048,0:2048]

Flux  = (Flux*flat)+dark*0.1+bias

if not os.path.exists("input"):
 os.makedirs("input")

if not os.path.exists("input/data_v"):
 os.makedirs("input/data_v")
pyfits.writeto("./input/data_v/sim_v.fits",Flux,header,clobber=True )
np.savetxt('fluxes.txt', np.c_[x_c,y_c,f])

pyfits.writeto("./input/data_v/bias.fits",bias,bias_header,clobber=True )
pyfits.writeto("./input/data_v/dark.fits",dark,dark_header,clobber=True )
pyfits.writeto("./input/data_v/flat.fits",flat,flat_header,clobber=True )

