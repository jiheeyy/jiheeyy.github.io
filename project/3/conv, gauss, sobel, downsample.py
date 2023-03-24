import numpy as np

"""
   Mirror an image about its border.

   Arguments:
      image - a 2D numpy array of shape (sx, sy)
      wx    - a scalar specifying width of the top/bottom border
      wy    - a scalar specifying width of the left/right border

   Returns:
      img   - a 2D numpy array of shape (sx + 2*wx, sy + 2*wy) containing
              the original image centered in its interior and a surrounding
              border of the specified width created by mirroring the interior
"""


def mirror_border(image, wx=1, wy=1):
    assert image.ndim == 2, 'image should be grayscale'
    sx, sy = image.shape
    # mirror top/bottom
    top = image[:wx:, :]
    bottom = image[(sx - wx):, :]
    img = np.concatenate( \
        (top[::-1, :], image, bottom[::-1, :]), \
        axis=0 \
        )
    # mirror left/right
    left = img[:, :wy]
    right = img[:, (sy - wy):]
    img = np.concatenate( \
        (left[:, ::-1], img, right[:, ::-1]), \
        axis=1 \
        )
    return img


"""
   Pad an image with zeros about its border.

   Arguments:
      image - a 2D numpy array of shape (sx, sy)
      wx    - a scalar specifying width of the top/bottom border
      wy    - a scalar specifying width of the left/right border

   Returns:
      img   - a 2D numpy array of shape (sx + 2*wx, sy + 2*wy) containing
              the original image centered in its interior and a surrounding
              border of zeros
"""


def pad_border(image, wx=1, wy=1):
    assert image.ndim == 2, 'image should be grayscale'
    sx, sy = image.shape
    img = np.zeros((sx + 2 * wx, sy + 2 * wy))
    img[wx:(sx + wx), wy:(sy + wy)] = image
    return img


"""
   Remove the border of an image.

   Arguments:
      image - a 2D numpy array of shape (sx, sy)
      wx    - a scalar specifying width of the top/bottom border
      wy    - a scalar specifying width of the left/right border

   Returns:
      img   - a 2D numpy array of shape (sx - 2*wx, sy - 2*wy), extracted by
              removing a border of the specified width from the sides of the
              input image
"""


def trim_border(image, wx=1, wy=1):
    assert image.ndim == 2, 'image should be grayscale'
    sx, sy = image.shape
    img = np.copy(image[wx:(sx - wx), wy:(sy - wy)])
    return img


"""
   Return an approximation of a 1-dimensional Gaussian filter.

   The returned filter approximates:

   g(x) = 1 / sqrt(2 * pi * sigma^2) * exp( -(x^2) / (2 * sigma^2) )

   for x in the range [-3*sigma, 3*sigma]
"""


def gaussian_1d(sigma=1.0):
    width = np.ceil(3.0 * sigma)
    x = np.arange(-width, width + 1)
    g = np.exp(-(x * x) / (2 * sigma * sigma))
    g = g / np.sum(g)  # normalize filter to sum to 1 ( equivalent
    g = np.atleast_2d(g)  # to multiplication by 1 / sqrt(2*pi*sigma^2) )
    return g


"""
   CONVOLUTION IMPLEMENTATION

   Convolve a 2D image with a 2D filter.

   Arguments:
      image  - a 2D numpy array
      filt   - a 1D or 2D numpy array, with odd dimensions
      mode   - 'zero': preprocess using pad_border or 'mirror': preprocess using mirror_border.

   Returns:
      result - a 2D numpy array of the same shape as image, containing the
               result of convolving the image with filt
"""
def conv_2d(image, filt, mode='zero'):
   # make sure that both image and filter are 2D arrays
   assert image.ndim == 2, 'image should be grayscale'
   filt = np.atleast_2d(filt)
   ##########################################################################
   # Flip filter horizontally and vertically
   # Variables describing filter dimensions

   filt = np.fliplr(np.flipud(filt))
   filtrow = filt.shape[0]
   filtcol = filt.shape[1]

   if mode == 'zero':
      image = pad_border(image, wx=int(filtrow//2), wy=int(filtcol//2))
   elif mode == 'mirror':
      image = mirror_border(image, wx=int(filtrow//2), wy=int(filtcol//2))
   # Preprocess image: padding image
   # Variables describing image dimensions
   else:
      print('Mode not expected')

   imrow = image.shape[0]
   imcol = image.shape[1]

   # Initialize result array with same dimensions as input image
   result = np.zeros((imrow - filtrow + 1, imcol - filtcol + 1))

   # Convolution
   for i in range(imrow - filtrow + 1):  # go through each row
      if i > imrow - filtrow:
         break
      for j in range(imcol - filtcol + 1):  # go through each column
         if j > imcol - filtcol:
            break
         try:
            result[i, j] = (image[i:i + filtrow, j:j + filtcol] * filt).sum()
         except:
            break
   ##########################################################################
   return result


"""
   GAUSSIAN DENOISING

   Denoise an image by convolving it with a 2D Gaussian filter.

   Convolve the input image with a 2D filter G(x,y) defined by:

   G(x,y) = 1 / sqrt(2 * pi * sigma^2) * exp( -(x^2 + y^2) / (2 * sigma^2) )

   Arguments:
      image - a 2D numpy array
      sigma - standard deviation of the Gaussian

   Returns:
      img   - denoised image, a 2D numpy array of the same shape as the input
"""
# Helper function for Q2.
# Gets 2d filter from 1d Gaussian filter.
def gaussian_2d(sigma=1.0):
   g1 = gaussian_1d(sigma)
   g2 = np.outer(g1, g1)
   return g2

def denoise_gaussian(image, sigma=1.0):
   ##########################################################################
   gfilt = gaussian_2d(sigma)
   img = conv_2d(image, gfilt, mode='mirror')
   ##########################################################################
   return img

"""
   SMOOTHING AND DOWNSAMPLING

   Smooth an image by applying a gaussian filter, followed by downsampling with a factor k.

      In principle, the sigma in gaussian filter should respect the cut-off frequency
      1 / (2 * k) with k being the downsample factor and the cut-off frequency of
      gaussian filter is 1 / (2 * pi * sigma).

   Arguments:
     image - a 2D numpy array
     downsample_factor - an integer specifying downsample rate

   Returns:
     result - downsampled image, a 2D numpy array with spatial dimension reduced
"""


def smooth_and_downsample(image, downsample_factor=2):
    ##########################################################################
    image = denoise_gaussian(image, sigma=downsample_factor / np.pi)
    result = image[::downsample_factor, ::downsample_factor]
    ##########################################################################
    return result


"""
   SOBEL GRADIENT OPERATOR
   Compute an estimate of the horizontal and vertical gradients of an image
   by applying the Sobel operator.
   The Sobel operator estimates gradients dx(horizontal), dy(vertical), of
   an image I as:

         [ 1  0  -1 ]
   dx =  [ 2  0  -2 ] (*) I
         [ 1  0  -1 ]

         [  1  2  1 ]
   dy =  [  0  0  0 ] (*) I
         [ -1 -2 -1 ]

   where (*) denotes convolution.
   Note:
      (1) Your implementation should be as efficient as possible.
      (2) Avoid creating artifacts along the border of the image.
   Arguments:
      image - a 2D numpy array
   Returns:
      dx    - gradient in x-direction at each point
              (a 2D numpy array, the same shape as the input image)
      dy    - gradient in y-direction at each point
              (a 2D numpy array, the same shape as the input image)
"""

def sobel_gradients(image):
   ##########################################################################
   kernel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
   kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

   dx = conv_2d(image, kernel_x, mode='mirror')
   dy = conv_2d(image, kernel_y, mode='mirror')
   ##########################################################################
   return dx, dy
