
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
from tqdm import tqdm
import os
from skimage.metrics import peak_signal_noise_ratio, mean_squared_error
from numba import jit
from time import time
from multiprocessing import Pool

def showMultiImages(arr, name = 'wind', scale = 1):
  newArr=  []
  for a in arr:
    newArr.append(cv2.resize(a, (0,0 ), fx = scale, fy = scale))
  cv2.imshow(name, np.concatenate(newArr, axis=1))
  cv2.waitKey(1)

def getImage(index, grayscale = False, scale = 0.5):
  if grayscale:
    grayscale = 0
  else:
    grayscale = 1
  gt = cv2.imread('data/Image' + str(index) + '.png', grayscale)
  gt = cv2.resize(gt, (0,0), fx = scale, fy = scale)
  return gt



def addNoise(image, noiseType, p = 0.001, mean = 0,  sigma = 0.3):
    if noiseType == 'GAUSSIAN':
      sigma *= 255
      # noise = np.random.normal(mean, sigma, (image.shape[0],image.shape[1])) 
      noise = np.zeros_like(image)
      noise = cv2.randn(noise, mean, sigma)
      ret = cv2.add(image, (noise))
      return ret
    elif noiseType == 'SALTNPEPPER':
      output = image.copy()
      noise = np.random.rand(image.shape[0], image.shape[1])
      output[noise < p] = 0
      output[noise > (1-p)] = 255

      return output



@jit(nopython=True)
def nonLocalMeans(noisy, params = tuple(), verbose = True):

  bigWindowSize, smallWindowSize, h  = params
  

  padwidth = bigWindowSize//2
  image = noisy.copy()
  paddedImage = np.zeros((image.shape[0] + bigWindowSize,image.shape[1] + bigWindowSize))
  paddedImage = paddedImage.astype(np.uint8)
  paddedImage[padwidth:padwidth+image.shape[0], padwidth:padwidth+image.shape[1]] = image
  paddedImage[padwidth:padwidth+image.shape[0], 0:padwidth] = np.fliplr(image[:,0:padwidth])
  paddedImage[padwidth:padwidth+image.shape[0], image.shape[1]+padwidth:image.shape[1]+2*padwidth] = np.fliplr(image[:,image.shape[1]-padwidth:image.shape[1]])
  paddedImage[0:padwidth,:] = np.flipud(paddedImage[padwidth:2*padwidth,:])
  paddedImage[padwidth+image.shape[0]:2*padwidth+image.shape[0], :] =np.flipud(paddedImage[paddedImage.shape[0] - 2*padwidth:paddedImage.shape[0] - padwidth,:])
  iterator = 0
  totalIterations = image.shape[1]*image.shape[0]*(bigWindowSize - smallWindowSize)**2

  if verbose:
    print("TOTAL ITERATIONS = ", totalIterations)

  outputImage = paddedImage.copy()

  smallhalfwidth = smallWindowSize//2

  for imageX in range(padwidth, padwidth + image.shape[1]):
    for imageY in range(padwidth, padwidth + image.shape[0]):
      
      bWinX = imageX - padwidth
      bWinY = imageY - padwidth

      compNbhd = paddedImage[imageY - smallhalfwidth:imageY + smallhalfwidth+1,imageX-smallhalfwidth:imageX+smallhalfwidth+1]
      
      
      pixelColor = 0
      totalWeight = 0

      for sWinX in range(bWinX, bWinX + bigWindowSize - smallWindowSize, 1):
        for sWinY in range(bWinY, bWinY + bigWindowSize - smallWindowSize, 1):          
          smallNbhd = paddedImage[sWinY:sWinY+smallWindowSize+1,sWinX:sWinX+smallWindowSize+1]
          euclideanDistance = np.sqrt(np.sum(np.square(smallNbhd - compNbhd)))
          weight = np.exp(-euclideanDistance/(h))
          totalWeight += weight
          pixelColor += weight*paddedImage[sWinY + smallhalfwidth, sWinX + smallhalfwidth]
          iterator += 1

          if verbose:
            percentComplete = iterator*100/totalIterations
            if percentComplete % 5 == 0:
              print('% COMPLETE = ', percentComplete)


          # if  iterator % 999 == 0:
          #   temppadded = outputImage.copy()  
          #   cv2.rectangle(temppadded, (imageX-smallhalfwidth, imageY - smallhalfwidth), (imageX+smallhalfwidth,imageY + smallhalfwidth), (0, 0, 0 ), 1)
          #   cv2.rectangle(temppadded, (sWinX, sWinY), (sWinX+smallWindowSize,sWinY+smallWindowSize), (255, 0, 0 ), 1)
          #   cv2.rectangle(temppadded, (bWinX, bWinY), (bWinX+bigWindowSize, bWinY+bigWindowSize), (255, 0, 0 ), 1)
          #   cv2.imshow('searchWindow', temppadded)
          #   cv2.imshow('smallnbhd', cv2.resize(smallNbhd,(300,300)))
          #   cv2.waitKey(1)
      pixelColor /= totalWeight
      outputImage[imageY, imageX] = pixelColor

  return outputImage[padwidth:padwidth+image.shape[0],padwidth:padwidth+image.shape[1]]



# http://dsvision.github.io/an-approach-to-non-local-means-denoising.html


def log( index, gtImg, noisy, gfiltered, nlmfiltered,  params, gaussian = False, salted = False):
  f = open('output/logs/' +str(index)+'-LOG.csv','a')
  if gaussian:
    f.write('Gaussian Noise\n')
  elif salted:
    f.write('Salt and Pepper Noise\n')

  f.write('Params: ' + str(params) + '\n')
  f.write('NOISY,GAUSSIAN FILTER on NOISE,NLM FILTER on NOISE\n')
  # f.write('|-----|------------------|----------|\n')
  f.write(str(peak_signal_noise_ratio(gtImg, noisy)))
  f.write(',')
  f.write(str(peak_signal_noise_ratio(gtImg, gfiltered)))
  f.write(',')
  f.write(str(peak_signal_noise_ratio(gtImg, nlmfiltered)))
  f.write('\n')
  f.write(str(mean_squared_error(gtImg, noisy)))
  f.write(',')
  f.write(str(mean_squared_error(gtImg, gfiltered)))
  f.write(',')
  f.write(str(mean_squared_error(gtImg, nlmfiltered)))
  f.write('\n\n')

  
    





def denoise(index, verbose = False):
  print('STARTING IMAGE', index)
  f = open('output/logs/' +str(index)+'-LOG.csv','w')
  f.close()
  

  sigma = 0.1
  p = 0.01
  scale = 2
  gtImg = getImage(index, grayscale = True, scale = scale)
  gNoised = addNoise(gtImg, 'GAUSSIAN', sigma = sigma)
  saltNoised = addNoise(gtImg, 'SALTNPEPPER', p = p)

  kernelSize = 3
  kernel = (kernelSize , kernelSize)
  
  

  gaussian = True
  if gaussian:
    gParams = {
      'bigWindow' : 20,
      'smallWindow':6,
      'h':14,
      'scale':scale,
    }

    nlmFilteredGNoised = nonLocalMeans(gNoised, params = (gParams['bigWindow'], gParams['smallWindow'],gParams['h']), verbose = verbose)

    gFilteredGNoised = cv2.GaussianBlur(gNoised,kernel,0)
    
    log( index, gtImg, gNoised, gFilteredGNoised, nlmFilteredGNoised,  gParams, gaussian = True)
    
    cv2.imwrite('OUTPUT/NOISED/' + str(index) + '-GNOISE.png', gNoised)
    cv2.imwrite('OUTPUT/NLMFILTER/' + str(index) + '-NLM-Gauss.png', nlmFilteredGNoised)
    cv2.imwrite('OUTPUT/GFILTER/' + str(index) + '-GF-Gauss.png', gFilteredGNoised)
    # showMultiImages((gtImg, gNoised, nlmFilteredGNoised, gFilteredGNoised), 'gauss')


  salted = True
  if salted:
    saltParams = {
      'bigWindow' : 20,
      'smallWindow':6,
      'h':16,
      'scale':scale,
    }
    nlmFilteredSalted = nonLocalMeans(saltNoised, params = (saltParams['bigWindow'], saltParams['smallWindow'],saltParams['h']), verbose = verbose)
    gFilteredSalted= cv2.GaussianBlur(saltNoised,kernel,0)
    
    
    log( index, gtImg, saltNoised, gFilteredSalted, nlmFilteredSalted,  saltParams, salted = True)
    
    cv2.imwrite('OUTPUT/NOISED/' + str(index) + '-SPNOISE.png', saltNoised)
    cv2.imwrite('OUTPUT/NLMFILTER/' + str(index) + '-NLM-Salted.png', nlmFilteredSalted)
    cv2.imwrite('OUTPUT/GFILTER/' + str(index) + '-GF-Salted.png', gFilteredSalted)
    # showMultiImages((gtImg, saltNoised, nlmFilteredSalted, gFilteredSalted),'salt')
  
  cv2.imwrite('OUTPUT/GT/' + str(index) + '-GT.png', gtImg)
  print("--------COMPLETED IMAGE", index, '-----------')
  
  
  
  
  
  

  




if __name__ == '__main__':

	# pool = Pool(processes=os.cpu_count())
	# pool.map(denoise, [1, 2, 3, 4, 5, 6, 7, 9, 10, 11])
  # for i in range(1, 11):
  #   
  denoise(11)

# denoise(5)


cv2.waitKey(0)
