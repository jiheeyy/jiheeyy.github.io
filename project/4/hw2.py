import numpy as np
from hw2_supp import *

"""
   INTEREST POINT OPERATOR (12 Points Implementation + 3 Points Write-up)

   Implement an interest point operator of your choice.

   Your operator could be:

   (A) The Harris corner detector (Szeliski 7.1.1)

               OR

   (B) The Difference-of-Gaussians (DoG) operator defined in:
       Lowe, "Distinctive Image Features from Scale-Invariant Keypoints", 2004.
       https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf

               OR

   (C) Any of the alternative interest point operators appearing in
       publications referenced in Szeliski or in lecture

              OR

   (D) A custom operator of your own design

   You implementation should return locations of the interest points in the
   form of (x,y) pixel coordinates, as well as a real-valued score for each
   interest point.  Greater scores indicate a stronger detector response.

   In addition, be sure to apply some form of spatial non-maximum suppression
   prior to returning interest points.

   Whichever of these options you choose, there is flexibility in the exact
   implementation, notably in regard to:

   (1) Scale

       At what scale (e.g. over what size of local patch) do you operate?

       You may optionally vary this according to an input scale argument.

       We will test your implementation at the default scale = 1.0, so you
       should make a reasonable choice for how to translate scale value 1.0
       into a size measured in pixels.

   (2) Nonmaximum suppression

       What strategy do you use for nonmaximum suppression?

       A simple (and sufficient) choice is to apply nonmaximum suppression
       over a local region.  In this case, over how large of a local region do
       you suppress?  How does that tie into the scale of your operator?

   For making these, and any other design choices, keep in mind a target of
   obtaining a few hundred interest points on the examples included with
   this assignment, with enough repeatability to have a large number of
   reliable matches between different views.

   If you detect more interest points than the requested maximum (given by
   the max_points argument), return only the max_points highest scoring ones.

   In addition to your implementation, include a brief write-up (in hw2.pdf)
   of your design choices.

   Arguments:
      image       - a grayscale image in the form of a 2D numpy array
      max_points  - maximum number of interest points to return
      scale       - (optional, for your use only) scale factor at which to
                    detect interest points
      mask        - (optional, for your use only) foreground mask constraining
                    the regions to extract interest points
   Returns:
      xs          - numpy array of shape (N,) containing x-coordinates of the
                    N detected interest points (N <= max_points)
      ys          - numpy array of shape (N,) containing y-coordinates
      scores      - numpy array of shape (N,) containing a real-valued
                    measurement of the relative strength of each interest point
                    (e.g. corner detector criterion OR DoG operator magnitude)
"""


def find_interest_points(image, max_points = 200, scale = 1.0, mask = None):
   # check that image is grayscale
   assert image.ndim == 2, 'image should be grayscale'
   ##########################################################################
   # Deal with mask?
   # This made scatter plots on edges image = denoise_gaussian(image, sigma=1)
   if mask is not None:
      image = image * mask.astype(int)

   dx = sobel_gradients(image)[0]
   dy = sobel_gradients(image)[1]

   Ixx = denoise_gaussian(dx**2)
   Iyy = denoise_gaussian(dy**2)
   Ixy = denoise_gaussian(dx*dy)

   # Use weighing kernel?
   # abs on h made corners significant
   h = (Ixx * Iyy - Ixy**2) - 0.06 * (Ixx + Iyy)**2

   # Eliminate edges
   h[h<0] = 0

   rows, cols = image.shape
   r_part = (rows // 50) + 2
   c_part = (cols // 50) + 2

   # Nonmax suppression
   for r in range(1,r_part):
      for c in range(1,c_part):
         r_max, c_max = min(rows,50*r), min(cols,50*c)
         r_min, c_min = max(r_max - 50, 50*(r-1)), max(c_max - 50, 50*(c-1))
         h[r_min:r_max, c_min:c_max] = keep_n_largest_and_10(h[r_min:r_max, c_min:c_max], 500)

   if np.count_nonzero(h) > max_points:
      h = keep_n_largest(h, max_points)
   else:
      pass
   # x is columns, y is rows, (0,0) left top corner

   ys, xs = np.nonzero(h)
   scores = h.flatten()
   scores = scores[scores != 0]

   ##########################################################################
   return xs, ys, scores

"""
   FEATURE DESCRIPTOR (12 Points Implementation + 3 Points Write-up)

   Implement a SIFT-like feature descriptor by binning orientation energy
   in spatial cells surrounding an interest point.

   Unlike SIFT, you do not need to build-in rotation or scale invariance.

   A reasonable default design is to consider a 3 x 3 spatial grid consisting
   of cell of a set width (see below) surrounding an interest point, marked
   by () in the diagram below.  Using 8 orientation bins, spaced evenly in
   [-pi,pi), yields a feature vector with 3 * 3 * 8 = 72 dimensions.

             ____ ____ ____
            |    |    |    |
            |    |    |    |
            |____|____|____|
            |    |    |    |
            |    | () |    |
            |____|____|____|
            |    |    |    |
            |    |    |    |
            |____|____|____|

                 |----|
                  width

   You will need to decide on a default spatial width.  Optionally, this can
   be a multiple of a scale factor, passed as an argument.  We will only test
   your code by calling it with scale = 1.0.

   In addition to your implementation, include a brief write-up (in hw2.pdf)
   of your design choices.

  Arguments:
      image    - a grayscale image in the form of a 2D numpy
      xs       - numpy array of shape (N,) containing x-coordinates
      ys       - numpy array of shape (N,) containing y-coordinates
      scale    - scale factor

   Returns:
      feats    - a numpy array of shape (N,K), containing K-dimensional
                 feature descriptors at each of the N input locations
                 (using the default scheme suggested above, K = 72)
"""

def extract_features(image, xs, ys, scale = 1.0):
   # check that image is grayscale
   assert image.ndim == 2, 'image should be grayscale'
   ##########################################################################
   image = denoise_gaussian(image)
   dx = sobel_gradients(image)[0]
   dy = sobel_gradients(image)[1]
   mag = np.sqrt(dx ** 2 + dy ** 2)
   theta = np.arctan2(dy, dx)
   bins = np.arange(-np.pi, np.pi, np.pi / 4)
   theta_bins = np.digitize(theta, bins)
   feats = np.array([])

   swidth = 9
   for a in range(len(xs)): # for every keypoint coordinate, retrieve 9 * 9 grid
      theta_array = sqgrid(theta_bins, ys[a], xs[a], swidth).flatten() # 81 items
      mag_array = sqgrid(mag, ys[a], xs[a], swidth).flatten() # 81 items
      feature_vector = np.array([])

      for b in range(swidth): # there are nine 3 * 3 boxes within the grid
         t = theta_array[(b*swidth) : (b*swidth)+swidth]
         m = mag_array[(b*swidth) : (b*swidth)+swidth]
         hist = np.zeros(8)

         for c in range(swidth): # each item in small grid with 9 items
            if t[c] == 0:
               continue
            else:
               hist[int(t[c]-1)] += m[c]

         if hist.sum() == 0:
            hist = np.full((1,8),1/8).flatten()
         else:
            hist = hist / hist.sum()
            if np.any(hist) > 0.2:
               hist = np.where(hist > .2, .2, hist)
               hist = hist / hist.sum()
            else:
               continue
         feature_vector = np.append(feature_vector, hist)

      if a == 0:
         feats = np.copy(feature_vector)
      else:
         feats = np.vstack((feats, feature_vector))
   ##########################################################################
   return feats

"""
   FEATURE MATCHING (7 Points Implementation + 3 Points Write-up)

   Given two sets of feature descriptors, extracted from two different images,
   compute the best matching feature in the second set for each feature in the
   first set.

   Matching need not be (and generally will not be) one-to-one or symmetric.
   Calling this function with the order of the feature sets swapped may
   result in different returned correspondences.

   For each match, also return a real-valued score indicating the quality of
   the match.  This score could be based on a distance ratio test, in order
   to quantify distinctiveness of the closest match in relation to the second
   closest match. It could optionally also incorporate scores of the interest
   points at which the matched features were extracted.  You are free to
   design your own criterion.

   In addition to your implementation, include a brief write-up (in hw2.pdf)
   of your design choices.

   Arguments:
      feats0   - a numpy array of shape (N0, K), containing N0 K-dimensional
                 feature descriptors (generated via extract_features())
      feats1   - a numpy array of shape (N1, K), containing N1 K-dimensional
                 feature descriptors (generated via extract_features())
      scores0  - a numpy array of shape (N0,) containing the scores for the
                 interest point locations at which feats0 was extracted
                 (generated via find_interest_point())
      scores1  - a numpy array of shape (N1,) containing the scores for the
                 interest point locations at which feats1 was extracted
                 (generated via find_interest_point())

   Returns:
      matches  - a numpy array of shape (N0,) containing, for each feature
                 in feats0, the index of the best matching feature in feats1
      scores   - a numpy array of shape (N0,) containing a real-valued score
                 for each match
"""

def match_features(feats0, feats1, scores0, scores1):
   ##########################################################################
   matches = np.array([])
   scores = np.array([])

   for i in range(len(feats0)):
      vec = feats0[i]
      vectors = np.copy(feats1)
      vectors -= np.tile(vec, (vectors.shape[0],1))
      vectors = np.square(vectors)
      d = np.sqrt(np.sum(vectors, axis=1)) # need to change to distance if inc scores

      # This makes match worse
      #pt_score = scores0[i]
      #scores_chi = np.copy(scores1)
      #scores_chi -= pt_score
      #scores_chi = (np.square(scores_chi)) / pt_score

      #scores_chi = assign_ranks(scores_chi)
      #scores_chi = scores_chi / 100
      #d = distance * (scores_chi + 0.5)

      best = np.where(d == np.min(d))
      without_best = np.where(d==np.min(d), np.max(d), d)
      second_best = np.where(without_best == np.min(without_best))
      multiple_best = np.append(best, second_best)

      matches = np.append(matches, multiple_best[0])
      scores = np.append(scores, d[int(multiple_best[0])]
                         /d[int(multiple_best[1])])

   ##########################################################################
   return matches, scores

"""
   HOUGH TRANSFORM (7 Points Implementation + 3 Points Write-up)

   Assuming two images of the same scene are related primarily by
   translational motion, use a predicted feature correspondence to
   estimate the overall translation vector t = [tx ty].

   Your implementation should use a Hough transform that tallies votes for
   translation parameters.  Each pair of matched features votes with some
   weight dependant on the confidence of the match; you may want to use your
   estimated scores to determine the weight.

   In order to accumulate votes, you will need to decide how to discretize the
   translation parameter space into bins.

   In addition to your implementation, include a brief write-up (in hw2.pdf)
   of your design choices.

   Arguments:
      xs0     - numpy array of shape (N0,) containing x-coordinates of the
                interest points for features in the first image
      ys0     - numpy array of shape (N0,) containing y-coordinates of the
                interest points for features in the first image
      xs1     - numpy array of shape (N1,) containing x-coordinates of the
                interest points for features in the second image
      ys1     - numpy array of shape (N1,) containing y-coordinates of the
                interest points for features in the second image
      matches - a numpy array of shape (N0,) containing, for each feature in
                the first image, the index of the best match in the second
      scores  - a numpy array of shape (N0,) containing a real-valued score
                for each pair of matched features

   Returns:
      tx      - predicted translation in x-direction between images
      ty      - predicted translation in y-direction between images
      votes   - a matrix storing vote tallies; this output is provided for
                your own convenience and you are free to design its format
"""

def hough_votes(xs0, ys0, xs1, ys1, matches, scores):
   ##########################################################################
   x_diff = np.array([])
   y_diff = np.array([])

   for m in range(len(matches)):
      x_diff = np.append(x_diff, (xs1[int(matches[m])] - xs0[m]))
      y_diff = np.append(y_diff, (ys1[int(matches[m])] - ys0[m]))

   distsq = 0
   for (a,b) in zip(x_diff,y_diff):
      e = a**2 + b**2
      if e > distsq:
         distsq = e
      else:
         pass

   max_rho = int(np.ceil(np.sqrt(distsq)))
   rho_bins = np.arange(-max_rho, max_rho+1)

   theta_bins = np.arange(-np.pi, np.pi, np.pi / 180)
   len_theta_bins = len(theta_bins)
   cos_bins = np.cos(theta_bins)
   sin_bins = np.sin(theta_bins)

   votes = np.zeros((len(rho_bins), len_theta_bins))
   voters = np.empty(int(len_theta_bins*len(rho_bins)), dtype=object)
   voters[...] = [[] for _ in range(voters.shape[0])]

   for j in range(len(x_diff)):
      x = x_diff[j]
      y = y_diff[j]

      for t in range(len_theta_bins):
         r = round(x * cos_bins[t] + y * sin_bins[t]) + max_rho
         votes[r,t] += scores[j]
         voters[int((r*360) + t)].append(j)

   max_index = np.unravel_index(votes.argmax(), votes.shape)
   inliers = voters[int((max_index[0]*360) + max_index[1])]

   keyx = np.array([])
   keyy = np.array([])
   for n in inliers:
      keyx = np.append(keyx, x_diff[n])
      keyy = np.append(keyy, y_diff[n])

   slope, intercept = least_squares_line(keyx, keyy)
   angle = np.arctan(slope)

   tx = intercept * np.cos(angle)
   ty = intercept * np.sin(angle)

   ##########################################################################
   return tx, ty, votes

"""
    OBJECT DETECTION (10 Points Implementation + 5 Points Write-up)

    Implement an object detection system which, given multiple object
    templates, localizes the object in the input (test) image by feature
    matching and hough voting.

    The first step is to match features between template images and test image.
    To prevent noisy matching from background, the template features should
    only be extracted from foreground regions.  The dense point-wise matching
    is then used to compute a bounding box by hough voting, where box center is
    derived from voting output and the box shape is simply the size of the
    template image.

    To detect potential objects with diversified shapes and scales, we provide
    multiple templates as input.  To further improve the performance and
    robustness, you are also REQUIRED to implement a multi-scale strategy
    either:
       (a) Implement multi-scale interest points and feature descriptors OR
       (b) Repeat a single-scale detection procedure over multiple image scales
           by resizing images.

    In addition to your implementation, include a brief write-up (in hw2.pdf)
    of your design choices on multi-scale implementaion and samples of
    detection results (please refer to display_bbox() function in visualize.py).

    Arguments:
        template_images - a list of gray scale images.  Each image is in the
                          form of a 2d numpy array which is cropped to tightly
                          cover the object.

        template_masks  - a list of binary masks having the same shape as the
                          template_image.  Each mask is in the form of 2d numpy
                          array specyfing the foreground mask of object in the
                          corresponding template image.

        test_img        - a gray scale test image in the form of 2d numpy array
                          containing the object category of interest.

    Returns:
         bbox           - a numpy array of shape (4,) specifying the detected
                          bounding box in the format of
                             (x_min, y_min, x_max, y_max)

"""

# I attempted object_detection with multiscale feature, but could I finish it.
# Therefore, I commented out object_detection with multiscale feature
# Functional single scale version of object_detection is below.
"""
def object_detection(template_images, template_masks, test_img):
   ##########################################################################
   test_xs, test_ys, test_scores = find_interest_points(test_img, max_points=200, scale=1.0, mask=None)
   test_feats = extract_features(test_img, test_xs, test_ys, scale=1.0)
   tbox = np.array([0,0,0,0])
   for i in range(len(template_images)):
      s = 3*np.random.random(1)[0]
      o_temp_img = template_images[i]
      temp_img = template_images[i]

      if s < 1:
         temp_img = smooth_and_downsample(temp_img, downsample_factor=round(1 / s))
      elif s == 1:
         continue
      else:
         temp_img = upsample(temp_img, round(1 / s))

      temp_xs, temp_ys, temp_scores = find_interest_points(temp_img, max_points=200, scale=1.0, mask=template_masks[i])
      temp_feats = extract_features(temp_img, temp_xs, temp_ys, scale=1.0)

      matches, scores = match_features(temp_feats, test_feats, temp_scores, test_scores)
      mask = scores >= np.percentile(scores, 50)  # Get better performing half of matches, scores
      matches = matches[mask]
      scores = scores[mask]

      tx, ty = hough_votes(temp_xs, temp_ys, test_xs, test_ys, matches, scores)[0:2]
      h, w = o_temp_img.shape
      tb = np.array([tx, ty, tx + w, ty + h])
      tbox = np.vstack((tbox, tb))
   bbox = np.average(tbox, axis=0)
   ##########################################################################
   return bbox
"""

def object_detection(template_images, template_masks, test_img):
   ##########################################################################
   test_xs, test_ys, test_scores = find_interest_points(test_img, max_points=200, scale=1.0, mask=None)
   test_feats = extract_features(test_img, test_xs, test_ys, scale=1.0)
   tbox = np.array([0,0,0,0])
   for i in range(len(template_images)):
      temp_img = template_images[i]

      temp_xs, temp_ys, temp_scores = find_interest_points(temp_img, max_points=200, scale=1.0, mask=template_masks[i])
      temp_feats = extract_features(temp_img, temp_xs, temp_ys, scale=1.0)

      matches, scores = match_features(temp_feats, test_feats, temp_scores, test_scores)
      mask = scores >= np.percentile(scores, 50)  # Get better performing half of matches, scores
      matches = matches[mask]
      scores = scores[mask]

      tx, ty = hough_votes(temp_xs, temp_ys, test_xs, test_ys, matches, scores)[0:2]
      h, w = temp_img.shape
      tb = np.array([tx, ty, tx + w, ty + h])
      tbox = np.vstack((tbox, tb))
   bbox = np.average(tbox, axis=0)
   ##########################################################################
   return bbox