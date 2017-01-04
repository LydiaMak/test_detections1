import os
import sys
import numpy as np
import matplotlib.pyplot as pyplot
import lsst.utils
import lsst.afw.table as afwTable
import lsst.afw.image as afwImage
from lsst.meas.algorithms import LoadIndexedReferenceObjectsTask
from lsst.pipe.tasks.photoCal import PhotoCalTask
import lsst.daf.persistence as dafPersist
from lsst.meas.astrom import AstrometryTask
import lsst.meas.algorithms as measAlg
from lsst.meas.algorithms.detection import SourceDetectionTask
from lsst.meas.base import SingleFrameMeasurementTask
import lsst.afw.display.ds9 as ds9
from lsst.meas.algorithms import SubtractBackgroundTask
import lsst.afw.display as afwDisplay

mypath = '/local/lydia/Data/all_SWASP_data/test_detections/OriginalData'
dataDir = '/local/lydia/Data/all_SWASP_data/test_detections/OriginalData/output_astrometry'
butler = dafPersist.Butler(dataDir)
exposure = afwImage.ExposureF(os.path.join(mypath, "output_astrometry","postISRCCD", "postISRCCD.fits"))
dataId = {'run': 'data', 'filter': 'v'}
exp = butler.get('calexp', dataId)
#exposure = butler.get('postISRCCD',dataId)


im = afwImage.ExposureF(exposure)
im1 = afwImage.ExposureF(exp)


sour = butler.get("src", dataId)
n = len(sour)
print (n)
print (sour.schema)
smi = exposure.getMaskedImage()
smi1 = exp.getMaskedImage()
srcCat = afwTable.SourceCatalog.readFits(os.path.join(mypath, "output_astrometry", "sci-results","src","src.fits"))
img = smi.getImage()
img1 = smi1.getImage()
nimg = img.getArray()
nimg1 = img1.getArray()
#pyplot.imshow(nimg, cmap='gray')
#pyplot.gcf().savefig("test.png")

ds9.mtv(smi1,frame=4)
ds9.mtv(smi1.getVariance(), frame=5)
ds9.mtv(smi1.getMask(), frame=6)

ds9.mtv(smi,frame=0)
ds9.mtv(smi.getVariance(), frame=1)
ds9.mtv(smi.getMask(), frame=2)
n = len(srcCat)
backgroundConfig = SubtractBackgroundTask.ConfigClass()
backgroundTask = SubtractBackgroundTask(config=backgroundConfig)
bgRes = backgroundTask.run(exposure=exposure)
background = bgRes.background
bgRes1 = backgroundTask.run(exposure=exp)
background1 = bgRes1.background
    # compute mean and variance of the background
backgroundImage = background.getImage()
backgroundImage1 = background1.getImage()
ds9.mtv(backgroundImage,frame=3)

ds9.mtv(backgroundImage1,frame=7)
print type(im)

#m = afwDisplay.Mosaic()

#images = [smi,smi1]
#for i in range(len(images)):
#    m.append(images[i])


#mosaic = m.makeMosaic()
#ds9.mtv(mosaic)
