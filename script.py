#!user/bin/env python
import numpy as np
import cv2 as cv
from rasterio.plot import show
import subprocess
import rasterio as rio
import json
import os
import glob
import matplotlib.pyplot as plt

# metadata keys
keys_to_extract = [
    'CalibratedOpticalCenterX', 'CalibratedOpticalCenterY', 
    'VignettingData', 'DewarpData', 'CalibratedHMatrix', 
    'BitsPerSample', 'BlackLevel', 'SensorGain', 
    'ExposureTime', 'SensorGainAdjustment', 'Irradiance',
    'VignettingCenter', 'VignettingPolynomial'
]


# get metadata from [XMP: drone-dji]
def get_metadata(file_path):
    try:
        result = subprocess.run(
            ['exiftool', '-j', file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True  
        )
        metadata = json.loads(result.stdout)
        return metadata[0] if metadata else {}
    except subprocess.CalledProcessError as e:
        print(f"Error running exiftool: {e.stderr}")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON output: {e}")
        return {}


# Get only the data needed for NDVI calculation
def filter_metadata(metadata, keys):
    return {key: metadata.get(key) for key in keys}

# Save metadata to a json file
def save_metadata(file_path, metadata):
    base_name = os.path.basename(file_path)
    output_file = f"metadata/{os.path.splitext(base_name)[0]}.json"
    with open(output_file, 'w') as f:
        json.dump(metadata, f, indent=4)

# Read the TIF image
def read_image(file_path):
    with rio.open(file_path) as src:
        return src.read(1), src.transform
    
# get metadata from json file
def get_md_from_json(file_path, key):
    filename_j = f"metadata/{os.path.splitext(os.path.basename(file_path))[0]}.json"
    try:
        with open(filename_j) as f:
            metadata = json.load(f)
            return metadata[key]
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    

def process_image(file_path, keys):
    metadata = get_metadata(file_path)
    filtered_metadata = filter_metadata(metadata, keys)
    save_metadata(file_path, filtered_metadata)


# extract for all tif images in the DJI_202405031358_001 directory
image_files = glob.glob('DJI_202405031358_001/*.TIF') 

#this loop needs to be run only once to get the metadata into json files
for image_file in image_files:
    process_image(image_file, keys_to_extract)

###########################################
#### Correction and alignment functoins
###########################################

#For step 1: Vignette correction
def correct_vignette(image):
    #get the vignetting data from the metadata
    k = get_md_from_json(image, 'VignettingData')

    #convert string to numpy array
    k = np.array([float(i) for i in k.split(',')]).reshape(6, 1)

    #read image
    data, transf = read_image(image)
    
    center_x = get_md_from_json(image, 'CalibratedOpticalCenterX')
    center_y = get_md_from_json(image, 'CalibratedOpticalCenterY')

    corrected_img = np.zeros_like(data)    

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            r = np.sqrt((j - center_x)**2 + (i - center_y)**2)
            factor = 1 + k[0] * r + k[1] * r**2 + k[2] * r**3 + k[3] * r**4 + k[4] * r**5 + k[5] * r**6
            # factor = np.polyval(k, r)
            corrected_img[i, j] = data[i, j] * factor
    return corrected_img, transf


#For step 2: distortion correction
def undistort_image(image):
    # get the dewarp data from the metadata
    dewarp_data = get_md_from_json(image, 'DewarpData')
    center_x = get_md_from_json(image, 'CalibratedOpticalCenterX')
    center_y = get_md_from_json(image, 'CalibratedOpticalCenterY')

    # correct vignetting
    corrected, _ = correct_vignette(image)

    dewarp_data = dewarp_data.split(';')[1]
    dewarp_data = np.array([float(i) for i in dewarp_data.split(',')])

    # define the camera matrix
    fx , fy, cx, cy = dewarp_data[:4]
    mtx = np.array([
        [fx, 0, center_x + cx],
        [0, fy, center_y + cy],
        [0, 0, 1]
    ])

    # define the distortion coefficients
    k1, k2, p1, p2, k3 = dewarp_data[4:]
    dist = np.array([k1, k2, p1, p2, k3])

    # image dimension
    h, w = read_image(image)[0].shape 

    # Compute the optimal new camera matrix
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    # undistort the image
    dst = cv.undistort(corrected, mtx, dist, None, newcameramtx)

    # # crop the image
    x, y, w, h = roi
    # dst = dst[y:y+h, x:x+w]
    return dst

#step3: Align images
def align_image(image_path):
    Hmatrix = get_md_from_json(image_path, 'CalibratedHMatrix')

    #convert string to numpy array
    Hmatrix = np.array([float(i) for i in Hmatrix.split(',')]).reshape(3, 3)
    
    #dewarp image
    image_data = undistort_image(image_path)

    print(image_data.shape)
    #align image
    aligned_image = cv.warpPerspective(image_data, Hmatrix, (image_data.shape[1], image_data.shape[0]))
    return aligned_image

# step 4: align diffs due to exposure time
def align_exposure_diffs(tgt_path, src_path):
    
     #! step1 through step3 corrections
    tgt_img_data = align_image(tgt_path)
    src_img_data = align_image(src_path)


    # apply Guassian smoothing
    src_img_datas = cv.GaussianBlur(src_img_data, (5, 5), 0)
    tgt_img_datas = cv.GaussianBlur(tgt_img_data, (5, 5), 0)
 
    # apply sobel filter
    edge_img1 = cv.Sobel(src_img_datas, cv.CV_64F, 1, 0, ksize=5)
    edge_img2 = cv.Sobel(tgt_img_datas, cv.CV_64F, 1, 0, ksize=5)

    # convert to absolute values
    edge_img1 = cv.convertScaleAbs(edge_img1)
    edge_img2 = cv.convertScaleAbs(edge_img2)

    # convert to float32 for ECC
    edge_img1 = np.float32(edge_img1)
    edge_img2 = np.float32(edge_img2)
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    _, warp_matrix = cv.findTransformECC(edge_img1, edge_img2, warp_matrix, cv.MOTION_AFFINE)
    aligned_img2 = cv.warpAffine(src_img_data, warp_matrix, (tgt_img_data.shape[1], tgt_img_data.shape[0]), flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP)    

    return tgt_img_data, aligned_img2

def adjust_etime(etime):
    print("etime:",etime)
    x, y = map(float, etime.split('/'))
    etime = x/y * 1e6 # 1e6 to convert to microseconds
    return etime

# NVDI calculation
def calculate_ndvi(nir, red):
    nir_irradiance = get_md_from_json(nir, 'Irradiance') # NIR_ls * pLS_nir
    red_irradiance = get_md_from_json(red, 'Irradiance') # RED_ls * pLS_red

    pCam_nir = get_md_from_json(nir, 'SensorGainAdjustment')
    pCam_red = get_md_from_json(red, 'SensorGainAdjustment')

    #nir metadata
    NIR_etime = adjust_etime(get_md_from_json(nir, 'ExposureTime'))
    NIR_gain = get_md_from_json(nir, 'SensorGain')
    # bitnum = get_md_from_json(nir, 'BitsPerSample')16
    I_bl = 3200

    #red metadata
    RED_etime = adjust_etime(get_md_from_json(red, 'ExposureTime'))
    RED_gain = get_md_from_json(red, 'SensorGain')
    # I_bl_red = get_md_from_json(red, 'BlackLevel')


    # get processed pixel values(step1 through step4)
    I_nir, I_red = align_exposure_diffs(nir, red)

    # normalized pixel values for NIR    
    I_nir = I_nir.astype(np.float32) / 65535
    I_red = I_red.astype(np.float32) / 65535

    # nor

    NIR_camera = (I_nir - I_bl) / (NIR_gain * NIR_etime/1e6)
    RED_camera = (I_red - I_bl) / (RED_gain * RED_etime/1e6)

    RED_ref = RED_camera * pCam_red / red_irradiance
    NIR_ref = NIR_camera * pCam_nir / nir_irradiance
    
    ndvi = (NIR_ref - RED_ref) / (NIR_ref + RED_ref)
    return ndvi

if __name__ == '__main__':
    tgt_path = 'DJI_202405031358_001/DJI_20240503140325_0001_MS_NIR.TIF'
    src_path = 'DJI_202405031358_001/DJI_20240503140325_0001_MS_R.TIF'

    ndvi = calculate_ndvi(tgt_path, src_path)
    # print(ndvi)
    show(ndvi, cmap=plt.cm.summer)