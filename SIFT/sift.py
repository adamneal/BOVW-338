 # -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 00:25:00 2021

@author: Tyler
"""

import numpy as np
from cv2 import GaussianBlur, resize, INTER_LINEAR, INTER_NEAREST
from scipy.ndimage.filters import convolve
from scipy.linalg import lstsq, det
from Keypoint import Keypoint
from functools import cmp_to_key

num_intervals=3
#sigma=2**(1/num_intervals)
sigma=1.6
assumed_blur=0.5
float_tolerance=1e-7


def digital_gaussian_scale_space(image):
    """gets base image"""
    image = resize(image, (0, 0), fx=2, fy=2, interpolation=INTER_LINEAR)
    sigma_diff = np.sqrt(max((sigma ** 2) - ((2 * assumed_blur) ** 2), 0.01))
    base_image = GaussianBlur(image, (0, 0), sigmaX=sigma_diff, sigmaY=sigma_diff)  # the image blur is now sigma instead of assumed_blur
    return base_image
"""change to not use cv2"""

def compute_number_octaves(image_shape):
    """calculates maximum number of octaves possible for the image"""
    return int(round(np.log(min(image_shape))/np.log(2)-1))

def gaussian_filter(sigma):
    """creates a gaussian filter convolved from the sigma value"""
    size=2*np.ceil(3*sigma)+1
    x,y=np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1] 
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2))) / (2*np.pi*sigma**2)
    return g/g.sum()

#def generate_gaussian_kernel(sigma, num_intervals):
#    """generates the kernel for each image to be blurred by other version"""
#    k=2**(1/num_intervals)
#    gaussian_kernel=gaussian_filter(sigma*k)
#   
#    return gaussian_kernel

def generate_gaussian_kernel(sigma, num_intervals):
    num_images_per_octave = num_intervals + 3
    k = 2 ** (1. / num_intervals)
    gaussian_kernels = np.zeros(num_images_per_octave)  # scale of gaussian blur necessary to go from one blur scale to the next within an octave
    gaussian_kernels[0] = sigma
    
    for image_index in range(1, num_images_per_octave):
        sigma_previous = (k ** (image_index - 1)) * sigma
        sigma_total = k * sigma_previous
        gaussian_kernels[image_index] = np.sqrt(sigma_total ** 2 - sigma_previous ** 2)
    return gaussian_kernels

"""def generate_Gaussian_Pyramid(image, num_octaves, gaussian_kernel,s):
    gaussian_pyramid=[]
    for i in range(num_octaves):
        gaussian_image_octave = []
        gaussian_image_octave.append(image)
        for j in range(s+2):
            next_image=convolve(gaussian_image_octave[-1], gaussian_kernel)
            gaussian_image_octave.append(next_image)
        gaussian_pyramid.append(gaussian_image_octave)
        octave_base=gaussian_image_octave[-3]
        image=resize(octave_base, (int(octave_base.shape[1] / 2), int(octave_base.shape[0] / 2)), interpolation=INTER_NEAREST)
        #resize without using cv2
    return gaussian_pyramid"""

def generate_Gaussian_Pyramid(image, num_octaves, gaussian_kernels,s):
    gaussian_images = []

    for octave_index in range(num_octaves):
        gaussian_images_in_octave = []
        gaussian_images_in_octave.append(image)  # first image in octave already has the correct blur
        for gaussian_kernel in gaussian_kernels[1:]:
            image = GaussianBlur(image, (0, 0), sigmaX=gaussian_kernel, sigmaY=gaussian_kernel)
            gaussian_images_in_octave.append(image)
        gaussian_images.append(gaussian_images_in_octave)
        octave_base = gaussian_images_in_octave[-3]
        image = resize(octave_base, (int(octave_base.shape[1] / 2), int(octave_base.shape[0] / 2)), interpolation=INTER_NEAREST)
    return gaussian_images

def generate_DoG_pyramid(gaussian_pyramid):
    dog_pyramid=[]
    for octave in gaussian_pyramid:
        DoG_octave=[]
        for i in range(1,len(octave)):
            DoG_octave.append(octave[i]-octave[i-1])
       # DoG_octave=np.concatenate([o[:,:,np.newaxis] for o in octave],axis=2)
        dog_pyramid.append(DoG_octave)
    return dog_pyramid

def findScaleSpaceExtrema(gaussian_pyramid, dog_pyramid,contrast_threshold,image_border_width):
    threshold=np.floor(0.5*contrast_threshold/num_intervals*255)
    keypoints=[]
    
    for octave_index, dog_images in enumerate(dog_pyramid):
        for image_index, (first_image,second_image,third_image) in enumerate(zip(dog_images,dog_images[1:],dog_images[2:])):
            for i in range(image_border_width, first_image.shape[0]-image_border_width):
                for j in range(image_border_width, first_image.shape[1]-image_border_width):
                    if isPixelExtremum(first_image[i-1:i+2, j-1:j+2], second_image[i-1:i+2, j-1:j+2], third_image[i-1:i+2, j-1:j+2], threshold):
                        localisation_result=localiseExtremum(i,j,image_index,octave_index,num_intervals,dog_images,sigma,contrast_threshold,image_border_width)
                        if localisation_result is not None:
                            keypoint,localised_image_index=localisation_result
                            keypoint_and_orientations=computeKeyPointOrientation(keypoint,octave_index, gaussian_pyramid[octave_index][localised_image_index])
                            for keypoint_and_orientation in keypoint_and_orientations:
                                keypoints.append(keypoint_and_orientation)
    return keypoints

def isPixelExtremum(first_subimage,second_subimage,third_subimage,threshold):
    candidate_point=second_subimage[1,1]
    if abs(candidate_point)>threshold:
        if candidate_point>0:
            return (candidate_point >= first_subimage).all() and \
                   (candidate_point >= third_subimage).all() and \
                   (candidate_point >= second_subimage[0, :]).all() and \
                   (candidate_point >= second_subimage[2, :]).all() and \
                   candidate_point >= second_subimage[1, 0] and \
                   candidate_point >= second_subimage[1, 2]
        elif candidate_point < 0:
            return (candidate_point <= first_subimage).all() and \
                   (candidate_point <= third_subimage).all() and \
                   (candidate_point <= second_subimage[0, :]).all() and \
                   (candidate_point <= second_subimage[2, :]).all() and \
                   candidate_point <= second_subimage[1, 0] and \
                   candidate_point <= second_subimage[1, 2]
    return False
            
def computeHessian(pixel_array):
    keypoint=pixel_array[1,1,1]
    dxx=pixel_array[1,1,2]-2*keypoint+pixel_array[1,1,0]
    dyy=pixel_array[1,2,1]-2*keypoint+pixel_array[1,0,1]
    dss=pixel_array[2,1,1]-2*keypoint+pixel_array[0,1,1]
    dxy=0.25*(pixel_array[1,2,2]-pixel_array[1,2,0]-pixel_array[1,0,2]+pixel_array[1,0,0])
    dxs=0.25*(pixel_array[2,1,2]-pixel_array[2,1,0]-pixel_array[0,1,2]+pixel_array[0,1,0])
    dys=0.25*(pixel_array[2,2,1]-pixel_array[2,0,1]-pixel_array[0,2,1]+pixel_array[0,0,1])
    return np.array([[dxx,dxy,dxs],
            [dxy,dyy,dys],
            [dxs,dys,dss]])

def computeJacobian(pixel_array):
    dx=0.5*(pixel_array[1,1,2]-pixel_array[1,1,0])
    dy=0.5*(pixel_array[1,2,1]-pixel_array[1,0,1])
    ds=0.5*(pixel_array[2,1,1]-pixel_array[0,1,1])
    return np.array([dx,dy,ds])

"""def localiseExtremum(i,j,image_index,octave_index,num_intervals,dog_images,sigma,contrast_threshold,image_border_width, eigenvalue_ratio=10):
    outside_image=False
    image_shape=dog_images[0].shape
    for attempt_index in range(5):
        first_image=dog_images[image_index-1]
        second_image=dog_images[image_index]
        third_image= dog_images[image_index+1]
        pixel_array=np.stack([first_image[i-1:i+2, j-1:j+2],second_image[i-1:i+2, j-1:j+2],third_image[i-1:i+2, j-1:j+2]]).astype('float32')/255
        jacobian=computeJacobian(pixel_array)
        hessian=computeHessian(pixel_array)
        extremum_update=-lstsq(hessian,jacobian,cond=None)[0]
        if abs(extremum_update[0])<0.5 and abs(extremum_update[1])<0.5 and abs(extremum_update[2])<0.5:
              break
        j+=int(round(extremum_update[0]))
        i+=int(round(extremum_update[1]))
        image_index+=int(round(extremum_update[2]))
        if i<image_border_width or i>=image_shape[0]-image_border_width or j<image_border_width or j>=image_shape[1]-image_border_width or image_index<1 or image_index>num_intervals:
            outside_image=True
            break
    if outside_image:
        return None
    if i>=4:
        return None
    functionValueatUpdatedExtremum=pixel_array[1,1,1]+0.5*np.dot(jacobian, extremum_update)
    if abs(functionValueatUpdatedExtremum)*num_intervals>=contrast_threshold:
        xy_hessian=hessian[:2,:2]
        xy_hessian_trace=np.trace(xy_hessian)
        xy_hessian_det=det(xy_hessian)
        if xy_hessian_det > 0 and eigenvalue_ratio * (xy_hessian_trace ** 2) < ((eigenvalue_ratio + 1) ** 2) * xy_hessian_det:
            # Contrast check passed -- construct and return OpenCV KeyPoint object
            x = (j + extremum_update[0]) * (2 ** octave_index)
            y = (i + extremum_update[1]) * (2 ** octave_index)
            octave = octave_index + image_index * (2 ** 8) + int(round((extremum_update[2] + 0.5) * 255)) * (2 ** 16)
            size = sigma * (2 ** ((image_index + extremum_update[2]) / float(num_intervals))) * (2 ** (octave_index + 1))  # octave_index + 1 because the input image was doubled
            response = abs(functionValueatUpdatedExtremum)
            keypoint = Keypoint(x,y,size,-1,response,octave,-1)
            return keypoint, image_index
    return None"""

def localiseExtremum(i, j, image_index, octave_index, num_intervals, dog_images_in_octave, sigma, contrast_threshold, image_border_width, eigenvalue_ratio=10, num_attempts_until_convergence=5):
    """Iteratively refine pixel positions of scale-space extrema via quadratic fit around each extremum's neighbors
    """
    extremum_is_outside_image = False
    image_shape = dog_images_in_octave[0].shape
    for attempt_index in range(num_attempts_until_convergence):
        # need to convert from uint8 to float32 to compute derivatives and need to rescale pixel values to [0, 1] to apply Lowe's thresholds
        first_image=dog_images_in_octave[image_index-1]
        second_image=dog_images_in_octave[image_index]
        third_image= dog_images_in_octave[image_index+1]
        pixel_cube = np.stack([first_image[i-1:i+2, j-1:j+2],
                            second_image[i-1:i+2, j-1:j+2],
                            third_image[i-1:i+2, j-1:j+2]]).astype('float32') / 255.
        gradient = computeJacobian(pixel_cube)
        hessian = computeHessian(pixel_cube)
        extremum_update = -lstsq(hessian, gradient, cond=None)[0]
        if abs(extremum_update[0]) < 0.5 and abs(extremum_update[1]) < 0.5 and abs(extremum_update[2]) < 0.5:
            break
        j += int(round(extremum_update[0]))
        i += int(round(extremum_update[1]))
        image_index += int(round(extremum_update[2]))
        # make sure the new pixel_cube will lie entirely within the image
        if i < image_border_width or i >= image_shape[0] - image_border_width or j < image_border_width or j >= image_shape[1] - image_border_width or image_index < 1 or image_index > num_intervals:
            extremum_is_outside_image = True
            break
    if extremum_is_outside_image:
        return None
    if attempt_index >= num_attempts_until_convergence - 1:
        return None
    functionValueatUpdatedExtremum = pixel_cube[1, 1, 1] + 0.5 * np.dot(gradient, extremum_update)
    if abs(functionValueatUpdatedExtremum) * num_intervals >= contrast_threshold:
        xy_hessian = hessian[:2, :2]
        xy_hessian_trace = np.trace(xy_hessian)
        xy_hessian_det = det(xy_hessian)
        if xy_hessian_det > 0 and eigenvalue_ratio * (xy_hessian_trace ** 2) < ((eigenvalue_ratio + 1) ** 2) * xy_hessian_det:
            # Contrast check passed -- construct and return OpenCV KeyPoint object
            x = (j + extremum_update[0]) * (2 ** octave_index)
            y = (i + extremum_update[1]) * (2 ** octave_index)
            octave = octave_index + image_index * (2 ** 8) + int(round((extremum_update[2] + 0.5) * 255)) * (2 ** 16)
            size = sigma * (2 ** ((image_index + extremum_update[2]) / float(num_intervals))) * (2 ** (octave_index + 1))  # octave_index + 1 because the input image was doubled
            response = abs(functionValueatUpdatedExtremum)
            keypoint = Keypoint(x,y,size,-1,response,octave,-1)
            return keypoint, image_index
    return None
                
def computeKeyPointOrientation(keypoint,octave_index,image,radius_factor=3, num_bins=36, peak_ratio=0.8, scale_factor=1.5):
    keypoints_and_orientations=[]
    image_shape=image.shape
    
    scale=scale_factor*keypoint.size/float(2**(octave_index+1))
    radius=int(round(radius_factor*scale))
    weight_factor=-0.5/(scale**2)
    raw_histogram=np.zeros(num_bins)
    smooth_histogram=np.zeros(num_bins)
    
    for i in range(-radius, radius + 1):
        region_y = int(round(keypoint.y / float(2 ** octave_index))) + i
        if region_y > 0 and region_y < image_shape[0] - 1:
            for j in range(-radius, radius + 1):
                region_x = int(round(keypoint.x / float(2 ** octave_index))) + j
                if region_x > 0 and region_x < image_shape[1] - 1:
                    dx = image[region_y, region_x + 1] - image[region_y, region_x - 1]
                    dy = image[region_y - 1, region_x] - image[region_y + 1, region_x]
                    gradient_magnitude = np.sqrt(dx * dx + dy * dy)
                    gradient_orientation = np.rad2deg(np.arctan2(dy, dx))
                    weight = np.exp(weight_factor * (i ** 2 + j ** 2))  # constant in front of exponential can be dropped because we will find peaks later
                    histogram_index = int(round(gradient_orientation * num_bins / 360.))
                    raw_histogram[histogram_index % num_bins] += weight * gradient_magnitude

    for n in range(num_bins):
        smooth_histogram[n] = (6 * raw_histogram[n] + 4 * (raw_histogram[n - 1] + raw_histogram[(n + 1) % num_bins]) + raw_histogram[n - 2] + raw_histogram[(n + 2) % num_bins]) / 16.
    orientation_max = max(smooth_histogram)
    orientation_peaks = np.where(np.logical_and(smooth_histogram > np.roll(smooth_histogram, 1), smooth_histogram > np.roll(smooth_histogram, -1)))[0]
    for peak_index in orientation_peaks:
        peak_value = smooth_histogram[peak_index]
        if peak_value >= peak_ratio * orientation_max:
            # Quadratic peak interpolation
            # The interpolation update is given by equation (6.30) in https://ccrma.stanford.edu/~jos/sasp/Quadratic_Interpolation_Spectral_Peaks.html
            left_value = smooth_histogram[(peak_index - 1) % num_bins]
            right_value = smooth_histogram[(peak_index + 1) % num_bins]
            interpolated_peak_index = (peak_index + 0.5 * (left_value - right_value) / (left_value - 2 * peak_value + right_value)) % num_bins
            orientation = 360. - interpolated_peak_index * 360. / num_bins
            if abs(orientation - 360.) < float_tolerance:
                orientation = 0
            new_keypoint = Keypoint(keypoint.x,keypoint.y, keypoint.size, orientation, keypoint.response, keypoint.octave)
            keypoints_and_orientations.append(new_keypoint)
    return keypoints_and_orientations
    
    
def compareKeypoints(keypoint1,keypoint2):
    if keypoint1.x!=keypoint2.x:
        return keypoint1.x-keypoint2.x
    if keypoint1.y!=keypoint2.y:
        return keypoint1.y-keypoint2.y
    if keypoint1.size!=keypoint2.size:
        return keypoint2.size-keypoint1.size
    if keypoint1.angle!=keypoint2.angle:
        return keypoint1.angle-keypoint2.angle
    if keypoint1.response!=keypoint2.response:
        return keypoint2.response-keypoint1.response
    if keypoint1.octave!=keypoint2.octave:
        return keypoint2.octave-keypoint1.octave
    return keypoint2.class_id-keypoint1.class_id

def removeDuplicateKeypoints(keypoints):
    if len(keypoints)<2:
        return keypoints
    
    keypoints.sort(key=cmp_to_key(compareKeypoints))
    unique_keypoints=[keypoints[0]]
    
    for keypoint in keypoints[1:]:
        last_keypoint=unique_keypoints[-1]
        if (last_keypoint.x!=keypoint.x or
            last_keypoint.y!=keypoint.y or
            last_keypoint.size!=keypoint.size or
            last_keypoint.angle!=keypoint.angle):
            unique_keypoints.append(keypoint)
    return unique_keypoints

def keypoint_to_image_size(keypoints):
    image_sizes=[]
    for keypoint in keypoints:
        keypoint.x=0.5*keypoint.x
        keypoint.y=0.5*keypoint.y
        keypoint.size=0.5*keypoint.size
        keypoint.octave=(keypoint.octave&~255)|((keypoint.octave-1)&255)
        image_sizes.append(keypoint)
    return image_sizes
    
def unpackOctave(keypoint):
    octave = keypoint.octave & 255
    layer = (keypoint.octave >> 8) & 255
    if octave >= 128:
        octave = octave | -128
    scale = 1 / float(1 << octave) if octave >= 0 else float(1 << -octave)
    return octave, layer, scale

def generateDescriptors(keypoints,gaussian_pyramid,window_width=4, num_bins=8, scale_multiplier=3, descriptor_max_value=0.2):
    descriptors = []

    for keypoint in keypoints:
        octave, layer, scale = unpackOctave(keypoint)
        gaussian_image = gaussian_pyramid[octave + 1][layer]
        num_rows, num_cols = gaussian_image.shape
        pointx = round(scale * np.array(keypoint.x))
        pointy = round(scale * np.array(keypoint.y))
        bins_per_degree = num_bins / 360.
        angle = 360. - keypoint.angle
        cos_angle = np.cos(np.deg2rad(angle))
        sin_angle = np.sin(np.deg2rad(angle))
        weight_multiplier = -0.5 / ((0.5 * window_width) ** 2)
        row_bin_list = []
        col_bin_list = []
        magnitude_list = []
        orientation_bin_list = []
        histogram_tensor = np.zeros((window_width + 2, window_width + 2, num_bins))   # first two dimensions are increased by 2 to account for border effects

        # Descriptor window size (described by half_width) follows OpenCV convention
        hist_width = scale_multiplier * 0.5 * scale * keypoint.size
        half_width = int(round(hist_width * np.sqrt(2) * (window_width + 1) * 0.5))   # sqrt(2) corresponds to diagonal length of a pixel
        half_width = int(min(half_width, np.sqrt(num_rows ** 2 + num_cols ** 2)))     # ensure half_width lies within image

        for row in range(-half_width, half_width + 1):
            for col in range(-half_width, half_width + 1):
                row_rot = col * sin_angle + row * cos_angle
                col_rot = col * cos_angle - row * sin_angle
                row_bin = (row_rot / hist_width) + 0.5 * window_width - 0.5
                col_bin = (col_rot / hist_width) + 0.5 * window_width - 0.5
                if row_bin > -1 and row_bin < window_width and col_bin > -1 and col_bin < window_width:
                    window_row = int(round(pointy + row))
                    window_col = int(round(pointx + col))
                    if window_row > 0 and window_row < num_rows - 1 and window_col > 0 and window_col < num_cols - 1:
                        dx = gaussian_image[window_row, window_col + 1] - gaussian_image[window_row, window_col - 1]
                        dy = gaussian_image[window_row - 1, window_col] - gaussian_image[window_row + 1, window_col]
                        gradient_magnitude = np.sqrt(dx * dx + dy * dy)
                        gradient_orientation = np.rad2deg(np.arctan2(dy, dx)) % 360
                        weight = np.exp(weight_multiplier * ((row_rot / hist_width) ** 2 + (col_rot / hist_width) ** 2))
                        row_bin_list.append(row_bin)
                        col_bin_list.append(col_bin)
                        magnitude_list.append(weight * gradient_magnitude)
                        orientation_bin_list.append((gradient_orientation - angle) * bins_per_degree)

        for row_bin, col_bin, magnitude, orientation_bin in zip(row_bin_list, col_bin_list, magnitude_list, orientation_bin_list):
            # Smoothing via trilinear interpolation
            # Notations follows https://en.wikipedia.org/wiki/Trilinear_interpolation
            # Note that we are really doing the inverse of trilinear interpolation here (we take the center value of the cube and distribute it among its eight neighbors)
            row_bin_floor, col_bin_floor, orientation_bin_floor = np.floor([row_bin, col_bin, orientation_bin]).astype(int)
            row_fraction, col_fraction, orientation_fraction = row_bin - row_bin_floor, col_bin - col_bin_floor, orientation_bin - orientation_bin_floor
            if orientation_bin_floor < 0:
                orientation_bin_floor += num_bins
            if orientation_bin_floor >= num_bins:
                orientation_bin_floor -= num_bins

            c1 = magnitude * row_fraction
            c0 = magnitude * (1 - row_fraction)
            c11 = c1 * col_fraction
            c10 = c1 * (1 - col_fraction)
            c01 = c0 * col_fraction
            c00 = c0 * (1 - col_fraction)
            c111 = c11 * orientation_fraction
            c110 = c11 * (1 - orientation_fraction)
            c101 = c10 * orientation_fraction
            c100 = c10 * (1 - orientation_fraction)
            c011 = c01 * orientation_fraction
            c010 = c01 * (1 - orientation_fraction)
            c001 = c00 * orientation_fraction
            c000 = c00 * (1 - orientation_fraction)

            histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, orientation_bin_floor] += c000
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c001
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, orientation_bin_floor] += c010
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c011
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, orientation_bin_floor] += c100
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c101
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, orientation_bin_floor] += c110
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c111

        descriptor_vector = histogram_tensor[1:-1, 1:-1, :].flatten()  # Remove histogram borders
        # Threshold and normalize descriptor_vector
        threshold = np.linalg.norm(descriptor_vector) * descriptor_max_value
        descriptor_vector[descriptor_vector > threshold] = threshold
        descriptor_vector /= max(np.linalg.norm(descriptor_vector), float_tolerance)
        # Multiply by 512, round, and saturate between 0 and 255 to convert from float32 to unsigned char (OpenCV convention)
        descriptor_vector = np.round(512 * descriptor_vector)
        descriptor_vector[descriptor_vector < 0] = 0
        descriptor_vector[descriptor_vector > 255] = 255
        descriptors.append(descriptor_vector)
    return np.array(descriptors, dtype='float32')
