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
   CONVOLUTION IMPLEMENTATION (10 Points)

   Convolve a 2D image with a 2D filter.

   Requirements:

   (1) Return a result the same size as the input image.

   (2) You may assume the filter has odd dimensions.

   (3) The result at location (x,y) in the output should correspond to
       aligning the center of the filter over location (x,y) in the input
       image.

   (4) When computing a product at locations where the filter extends beyond
       the defined image, treat missing terms as zero.  (Equivalently stated,
       treat the image as being padded with zeros around its border).

   You must write the code for the nested loops of the convolutions yourself,
   using only basic loop constructs, array indexing, multiplication, and
   addition operators.  You may not call any Python library routines that
   implement convolution.

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

   # raise NotImplementedError('conv_2d')
   ##########################################################################
   return result


"""
   GAUSSIAN DENOISING (5 Points)

   Denoise an image by convolving it with a 2D Gaussian filter.

   Convolve the input image with a 2D filter G(x,y) defined by:

   G(x,y) = 1 / sqrt(2 * pi * sigma^2) * exp( -(x^2 + y^2) / (2 * sigma^2) )

   You may approximate the G(x,y) filter by computing it on a
   discrete grid for both x and y in the range [-3*sigma, 3*sigma].

   See the gaussian_1d function for reference.

   Note:
   (1) Remember that the Gaussian is a separable filter.
   (2) Denoising should not create artifacts along the border of the image.
       Make an appropriate assumption in order to obtain visually plausible
       results along the border.

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
   # raise NotImplementedError('denoise_gaussian')
   ##########################################################################
   return img

def sobel_gradients(image):
   ##########################################################################
   kernel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
   kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

   dx = conv_2d(image, kernel_x, mode='mirror')
   dy = conv_2d(image, kernel_y, mode='mirror')
   # raise NotImplementedError('sobel_gradients')
   ##########################################################################
   return dx, dy

def smooth_and_downsample(image, downsample_factor=2):
    ##########################################################################
    image = denoise_gaussian(image, sigma=downsample_factor / np.pi)
    result = image[::downsample_factor, ::downsample_factor]
    # raise NotImplementedError('smooth_and_downsample')
    ##########################################################################
    return result

def upsample(image, scale_factor):
    rows, cols = image.shape
    new_rows, new_cols = int(rows * scale_factor), int(cols * scale_factor)
    new_image = np.zeros((new_rows, new_cols))
    for i in range(new_rows):
        for j in range(new_cols):
            row, col = i / scale_factor, j / scale_factor
            row, col = int(row), int(col)
            new_image[i, j] = image[row, col]
    mirror_image = mirror_border(new_image)
    for a in range (new_rows):
        for b in range(new_cols):
            new_image[a,b] = np.average(sqgrid(mirror_image, a+1, b+1, 3))
    return new_image

# Smallest value in array becomes 1 ... largest value becomes 99 if len(array) is 100
def assign_ranks(arr):
    arr = arr.astype(int)
    sorted_arr = np.sort(arr)[::1]
    ranks = np.arange(1,len(arr)+1)
    for i, a in enumerate(arr):
        idx = np.where(sorted_arr == a)[0][0]
        arr[i] = ranks[idx]
    return arr

# Calculate LS line from list of x,y coordinates
def least_squares_line(x, y):
   x_mean = np.mean(x)
   y_mean = np.mean(y)
   numerator = sum((x - x_mean) * (y - y_mean))
   denominator = sum((x - x_mean) ** 2)
   slope = numerator / denominator
   intercept = y_mean - slope * x_mean
   return slope, intercept

# Helper function for find_interest_points
# Used for nonmax suppression
# Return np array with value from 8 pixel neighbors + zero value
def surround(array, in_row, in_col):
   pad_array = pad_border(array)
   neighborhood = pad_array[in_row:in_row + 3, in_col:in_col + 3]
   nr, nc = neighborhood.shape
   neighborhood[nr // 2, nc // 2] = 0
   return neighborhood.flatten()

# Helper
# returns true if element is greater than 1.1 * values in list_, false otherwise
def ten_pct(element, neighbors):
   return all(element >= 1.1 * e for e in neighbors)

# Helper
# Keeps n largest values that are at least 10% larger than surrounding neighbors
def keep_n_largest_and_10(array, n):
   new = np.zeros(array.shape)

   flat_array = array.flatten()
   flat_array.sort()

   if len(flat_array) <= n:
      threshold = 0
   else:
      threshold = flat_array[-n]

   threshold_array = np.copy(array)
   threshold_array[threshold_array < threshold] = 0

   for index in np.argwhere(threshold_array > 0):
      if ten_pct(array[index[0], index[1]], surround(array, index[0], index[1])):
         new[index[0], index[1]] = array[index[0], index[1]]
      else:
         continue
   return new

# Keeps n largest values
def keep_n_largest(array, n):
   flat_array = array.flatten()
   flat_array.sort()
   threshold = flat_array[-n]
   array[array<threshold] = 0
   return array

# Assume zero padding
# Return square grid around array[in_row, in_col]
def sqgrid(array, in_row, in_col, width):
    a = width//2
    pad_array = pad_border(array, wx=a, wy=a)
    sqgrid = pad_array[(in_row):(in_row+width),(in_col):(in_col+width)]
    return sqgrid