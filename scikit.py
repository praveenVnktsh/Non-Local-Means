import numpy as np

from skimage import  img_as_float
from skimage.restoration import denoise_nl_means

from skimage import io
import cv2


def showMultiImages(arr, name = 'wind', scale = 0.7):
  newArr=  []
  for a in arr:
    newArr.append(cv2.resize(a, (0,0 ), fx = scale, fy = scale))
  cv2.imshow(name, np.concatenate(newArr, axis=1))
  cv2.waitKey(1)

def PSNR(original, noisy, peak=100): ## TAKEN FROM http://dsvision.github.io/an-approach-to-non-local-means-denoising.html
    mse = np.mean((original-noisy)**2)
    ret = 10*np.log10(peak*peak/mse)
    return ret

image = img_as_float(io.imread('data/Image5.png', as_gray = True))
mean = 0
sigma = 0.1
noise = np.random.normal(mean, sigma, (image.shape[0],image.shape[1])) 
noisy = image + noise

denoise = denoise_nl_means(image, 6, 9, 0.08, multichannel=True, fast_mode = False)
showMultiImages((image, noisy, denoise))
print('Better by ', -PSNR(noisy, image) + PSNR(denoise, image), 'dB')

cv2.waitKey(0)