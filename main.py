
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, mean_squared_error
from numba import jit
from multiprocessing import Pool
import os



def getImage(index, grayscale = False, scale = 0.5):
  '''
  Helper function that returns images given a certain image index
  '''
  if grayscale:
    grayscale = 0
  else:
    grayscale = 1
  gt = cv2.imread('data/Image' + str(index) + '.png', grayscale)
  gt = cv2.resize(gt, (0,0), fx = scale, fy = scale)
  return gt


def addNoise(image, noiseType, p = 0.001, mean = 0,  sigma = 0.3):
  ''' 
  This function takes an image and returns an image that has been noised with the given input parameters.
  p - Probability threshold of salt and pepper noise.
  noisetype - 
  '''
  if noiseType == 'GAUSSIAN':
    sigma *= 255 #Since the image itself is not normalized
    noise = np.zeros_like(image)
    noise = cv2.randn(noise, mean, sigma)
    ret = cv2.add(image, noise) #generate and add gaussian noise
    return ret
  elif noiseType == 'SALTNPEPPER':
    output = image.copy()
    noise = np.random.rand(image.shape[0], image.shape[1])
    output[noise < p] = 0
    output[noise > (1-p)] = 255
    return output



@jit(nopython=True)
def nonLocalMeans(noisy, params = tuple(), verbose = True):
  '''
  Performs the non-local-means algorithm given a noisy image.
  params is a tuple with:
  params = (bigWindowSize, smallWindowSize, h)
  Please keep bigWindowSize and smallWindowSize as even numbers
  '''

  bigWindowSize, smallWindowSize, h  = params
  padwidth = bigWindowSize//2
  image = noisy.copy()

  # The next few lines creates a padded image that reflects the border so that the big window can be accomodated through the loop
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


  # For each pixel in the actual image, find a area around the pixel that needs to be compared
  for imageX in range(padwidth, padwidth + image.shape[1]):
    for imageY in range(padwidth, padwidth + image.shape[0]):
      
      bWinX = imageX - padwidth
      bWinY = imageY - padwidth

      #comparison neighbourhood
      compNbhd = paddedImage[imageY - smallhalfwidth:imageY + smallhalfwidth + 1,imageX-smallhalfwidth:imageX+smallhalfwidth + 1]
      
      
      pixelColor = 0
      totalWeight = 0

      # For each comparison neighbourhood, search for all small windows within a large box, and compute their weights
      for sWinX in range(bWinX, bWinX + bigWindowSize - smallWindowSize, 1):
        for sWinY in range(bWinY, bWinY + bigWindowSize - smallWindowSize, 1):   
          #find the small box       
          smallNbhd = paddedImage[sWinY:sWinY+smallWindowSize + 1,sWinX:sWinX+smallWindowSize + 1]
          euclideanDistance = np.sqrt(np.sum(np.square(smallNbhd - compNbhd)))
          #weight is computed as a weighted softmax over the euclidean distances
          weight = np.exp(-euclideanDistance/h)
          totalWeight += weight
          pixelColor += weight*paddedImage[sWinY + smallhalfwidth, sWinX + smallhalfwidth]
          iterator += 1

          if verbose:
            percentComplete = iterator*100/totalIterations
            if percentComplete % 5 == 0:
              print('% COMPLETE = ', percentComplete)

      pixelColor /= totalWeight
      outputImage[imageY, imageX] = pixelColor

  return outputImage[padwidth:padwidth+image.shape[0],padwidth:padwidth+image.shape[1]]



def log( index, gtImg, noisy, gfiltered, nlmfiltered,  params, gaussian = False, salted = False):
  '''
  This function logs the results in a .csv file.
  The skimage library is used to compute the MSE and PSNR
  '''

  f = open('OUTPUT/LOGS/' +str(index)+'-LOG.csv','a')
  if gaussian:
    f.write('Gaussian Noise\n')
  elif salted:
    f.write('Salt and Pepper Noise\n')

  f.write('Params: ' + str(params) + '\n')
  f.write('NOISY,GAUSSIAN FILTER on NOISE,NLM FILTER on NOISE\n')
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

  
    


def denoise(index, verbose = False, gaussian = True, salted = True):
  '''
  Helper function that:
  - takes an index
  - gets the images
  - adds noise
  - Denoises with various filters and logs the output
  - Saves all images

  '''
  print('DENOISING IMAGE', index)

  #For logistical purposes
  f = open('output/logs/' +str(index)+'-LOG.csv','w')
  f.close()
  
  
  scale = 2 #Scale factor of the image
  gtImg = getImage(index, grayscale = True, scale = scale)

  # Noise parameters
  sigma = 0.15 #Gaussian sigma
  p = 0.035 #Threshold for SNP noise

  gNoised = addNoise(gtImg, 'GAUSSIAN', sigma = sigma) 
  saltNoised = addNoise(gtImg, 'SALTNPEPPER', p = p)

  # Parameters for denoising using gaussian filter
  kernelSize = 3
  kernel = (kernelSize , kernelSize)
  
  if gaussian:
    #NLM filter parameters
    gParams = {
      'bigWindow' : 20,
      'smallWindow':6,
      'h':14,
      'scale':scale,
    }

    #perform NLM filtering
    nlmFilteredGNoised = nonLocalMeans(gNoised, params = (gParams['bigWindow'], gParams['smallWindow'],gParams['h']), verbose = verbose)

    #perform gaussian filtering
    gFilteredGNoised = cv2.GaussianBlur(gNoised,kernel,0)
    
    #log the results
    log(index, gtImg, gNoised, gFilteredGNoised, nlmFilteredGNoised,  gParams, gaussian = True)
    
    #write images to file
    cv2.imwrite('OUTPUT/NOISED/' + str(index) + '-GNOISE.png', gNoised)
    cv2.imwrite('OUTPUT/NLMFILTER/' + str(index) + '-NLM-Gauss.png', nlmFilteredGNoised)
    cv2.imwrite('OUTPUT/GFILTER/' + str(index) + '-GF-Gauss.png', gFilteredGNoised)


  
  if salted:
    #NLM filter parameters
    saltParams = {
      'bigWindow' : 20,
      'smallWindow':6,
      'h':16,
      'scale':scale,
    }

    #perform NLM filtering
    nlmFilteredSalted = nonLocalMeans(saltNoised, params = (saltParams['bigWindow'], saltParams['smallWindow'],saltParams['h']), verbose = verbose)

    #perform gaussian filtering
    gFilteredSalted= cv2.GaussianBlur(saltNoised,kernel,0)
    
    #log the results
    log( index, gtImg, saltNoised, gFilteredSalted, nlmFilteredSalted,  saltParams, salted = True)
    
    #write images to file
    cv2.imwrite('OUTPUT/NOISED/' + str(index) + '-SPNOISE.png', saltNoised)
    cv2.imwrite('OUTPUT/NLMFILTER/' + str(index) + '-NLM-Salted.png', nlmFilteredSalted)
    cv2.imwrite('OUTPUT/GFILTER/' + str(index) + '-GF-Salted.png', gFilteredSalted)
  
  cv2.imwrite('OUTPUT/GT/' + str(index) + '-GT.png', gtImg)
  print("--------COMPLETED IMAGE", index, '-----------')
  


if __name__ == '__main__':
  #multiprocessing allows us to parallely finish off all images!
	pool = Pool(processes=os.cpu_count())
	pool.map(denoise, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])



