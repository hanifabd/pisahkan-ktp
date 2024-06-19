import os
import cv2
import numpy as np
import math
from pythonRLSA.rlsa_fast import rlsa_fast

def add_white_block_signature(image, block_width, block_height):
    '''
    This function will cover offside signature in detail area
    '''
    # Get the dimensions of the original image
    height, width, _ = image.shape

    # Create a white block image
    white_block = np.full((block_height, block_width, 3), 255, dtype=np.uint8)

    # Place the white block in front of the original image
    image_with_block_front = np.copy(image)
    image_with_block_front[height-block_height:, width-block_width:] = white_block

    return image_with_block_front

def add_white_block_provinsi(image, block_width, block_height):
    '''
    This function will create white block in upper left, upper left, and upper center of provinsi area
    '''
    # Get the dimensions of the original image
    height, width, _ = image.shape

    # Create a white block image
    white_block = np.full((block_height, block_width, 3), 255, dtype=np.uint8)
    upper_height = 5
    white_block_upper = np.full((upper_height, width, 3), 255, dtype=np.uint8)

    image_with_block_front = np.copy(image)
    # Place the white block in the upper left corner of the original image
    image_with_block_front[:block_height, :block_width] = white_block
    # Place the white block in the upper right corner of the original image
    image_with_block_front[:block_height, width - block_width:] = white_block
    # Place the white block in the upper center of the original image
    image_with_block_front[:upper_height, :] = white_block_upper

    return image_with_block_front

def prepareRecognitionArea(image_path):
    '''
    This function will crop image to provinsi, nik, and detail area
    '''
    assert isinstance(image_path, str), "Image Path for OCR Process"
    try:
      # Open Image
      img = cv2.imread(image_path)

      # Resize Image to Standard Preprocessing Size
      imgResize = cv2.resize(img, (1000, 700))

      # Segment Identity Card (Indonesian IC) based on Image Ratio
      imgHeight, imgWidth = imgResize.shape[:2]
      # Param List Note: Matrix[[y, height], [x, width]]
      provinsi_area = imgResize[5:int(imgHeight*0.16), 5:imgWidth-10]
      provinsi_area = add_white_block_provinsi(provinsi_area, 30, 50) # This function implemented for provinsi area only - to cover border error

      nik_area = imgResize[int(imgHeight*0.135):int((imgHeight*0.135)+(imgHeight*0.12)), 0:int(imgWidth*0.75)]

      detail_area = imgResize[int(imgHeight*0.23):int((imgHeight*0.23)+(imgHeight*0.7)), 0:int(imgWidth*0.7)]
      detail_area = add_white_block_signature(detail_area, 100, 200) # This function implemented for detail area only - to cover offside signature

      return imgResize, provinsi_area, nik_area, detail_area
    except Exception as e:
      print(f"ERROR: {e}")
      return None

def prepareRecognitionAreaNdarray(image):
    '''
    This function will crop image to provinsi, nik, and detail area
    '''
    assert isinstance(image, np.ndarray), "image can be ndarray type only"
    try:
      # Resize Image to Standard Preprocessing Size
      imgResize = cv2.resize(image, (1000, 700))

      # Segment Identity Card (Indonesian IC) based on Image Ratio
      imgHeight, imgWidth = imgResize.shape[:2]
      # Param List Note: Matrix[[y, height], [x, width]]
      provinsi_area = imgResize[5:int(imgHeight*0.16), 5:imgWidth-10]
      provinsi_area = add_white_block_provinsi(provinsi_area, 30, 50) # This function implemented for provinsi area only - to cover border error

      nik_area = imgResize[int(imgHeight*0.135):int((imgHeight*0.135)+(imgHeight*0.12)), 0:int(imgWidth*0.75)]

      detail_area = imgResize[int(imgHeight*0.23):int((imgHeight*0.23)+(imgHeight*0.7)), 0:int(imgWidth*0.7)]
      detail_area = add_white_block_signature(detail_area, 100, 200) # This function implemented for detail area only - to cover offside signature

      return imgResize, provinsi_area, nik_area, detail_area
    except Exception as e:
      print(f"ERROR: {e}")
      return None

def apply_clahe_to_channels(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    '''
    This function will do clahe operation
    '''
    # Split the image into individual channels
    channels = cv2.split(image)

    # Apply CLAHE to each channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    clahe_channels = [clahe.apply(channel) for channel in channels]

    # Merge the CLAHE-enhanced channels back into an image
    clahe_image = cv2.merge(clahe_channels)

    return clahe_image

def preprocessRecognitionArea(image, contrast_factor=1.7, gamma_factor=1.0):
    '''
    This function will preprocess the image of each area
    '''
    # Adjust Image Contrast
    factor = contrast_factor # 1.7
    imgAdjustedContrast = image.astype(float)*factor
    imgAdjustedContrast = np.clip(imgAdjustedContrast, 0, 255).astype(np.uint8)

    # Gamma Correction - gamma correction can strengthen or weaken dark colors depending on the gamma value used
    gamma = gamma_factor # 1.0
    gamma_corrected = np.clip(imgAdjustedContrast ** (1 / gamma), 0, 255).astype(np.uint8)

    # Standarize Image Channel
    imageClahe = apply_clahe_to_channels(gamma_corrected, clip_limit=3.0, tile_grid_size=(8, 8))

    # Denoise Image
    imgDenoise = cv2.fastNlMeansDenoisingColored(imageClahe, None, 12, 12, 2, 20)

    # Gray Image
    imgGray = cv2.cvtColor(imgDenoise, cv2.COLOR_BGR2GRAY)

    # Thresholding Image
    imgThresh = cv2.threshold(imgGray, 220, 255, cv2.THRESH_TRUNC)[1]

    return imgThresh

def add_black_block_details(image, x_position, block_width):
    '''
    This function will create black block the left of detail area
    '''
    # Get image dimensions
    height, width = image.shape

    # Calculate the starting and ending x coordinates of the block
    start_x = x_position
    end_x = start_x + block_width

    # Ensure the block stays within image bounds
    start_x = max(start_x, 0)
    end_x = min(end_x, width)

    # Create a white block
    image[:, start_x:end_x] = 0

    return image

def add_black_block_border(image, top, bottom, left, right):
    '''
    This function will create black block in the edge of the image
    '''
    # Get image dimensions
    height, width = image.shape

    # Create a black border
    image[:top, :] = 0  # Top border
    image[-bottom:, :] = 0  # Bottom border
    image[:, :left] = 0  # Left border
    image[:, -right:] = 0  # Right border

    return image

def findTextPattern(image, typeImage):
    '''
    This function will create text line pattern for each row in the image
    '''
    # Thresholding Image
    threshImage = cv2.threshold(image, 220, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # RLSA Operation to extract out the ROI(region of interest) like block-of-text/title/content with applied heuristics.
    rlsaValue = max(math.ceil(image.shape[0]/100),math.ceil(image.shape[1]/100))+20
    rlsaImage = rlsa_fast(threshImage, True, False, rlsaValue)

    # Reverse Monochrome Color
    reverseBWImage = cv2.bitwise_not(rlsaImage.astype(np.uint8))

    # Erode - Dilate Vertically
    kernel = np.ones((8, 10), np.uint8)
    erodeImage = cv2.erode(reverseBWImage, kernel, iterations=1)

    # Add White Block Middle Separator
    if typeImage == "detail":
      erodeImage = add_black_block_details(erodeImage, 100, 20)

    # Clear Edge
    erodeImage = add_black_block_border(erodeImage, 8, 4, 40, 40)

    # Remove Tall Black Regions - Indicate Noise
    clearImage = np.zeros_like(erodeImage)
    # Find contours of the black regions
    contours, _ = cv2.findContours(erodeImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Remove tall black regions
    max_height_threshold_black_region = 50
    for contour in contours:
        # Get bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)
        # Check if the height exceeds the threshold
        if h < max_height_threshold_black_region:
            # Fill the tall black region with white color
            cv2.drawContours(clearImage, [contour], 0, (255), -1)

    # Dilate Image
    kernel = np.ones((1, 300), np.uint8)
    fullRowDilateImage = cv2.dilate(clearImage, kernel, iterations=1)

    # Erode - Dilate Vertically
    kernel = np.ones((4, 1), np.uint8)
    fullRowErodeImage = cv2.erode(fullRowDilateImage, kernel, iterations=1)

    # Thinning Black Area
    thinDilateImage = cv2.ximgproc.thinning(fullRowErodeImage)

    # Reverse Monochrome Color
    patternImage = cv2.bitwise_not(thinDilateImage)

    return patternImage

def provinsiCreateBoudingBox(image, min_area_threshold):
    '''
    This function will create bounding box for provinsi area
    '''
    # Thresholding Image
    imgThresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5)

    # Find contours
    contours, _ = cv2.findContours(imgThresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get bounding boxes
    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h >= min_area_threshold:
            bounding_boxes.append((0, max(y-20, 0), min(x+w, int(image.shape[1])), y+h+20))

    bounding_boxes.sort(key=lambda bbox: (bbox[1], bbox[0]))

    return bounding_boxes

def nikCreateBoudingBox(image, min_area_threshold):
    '''
    This function will create bounding box for nik area
    '''
    # Thresholding Image
    imgThresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5)

    # Find contours
    contours, _ = cv2.findContours(imgThresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get bounding boxes
    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h >= min_area_threshold:
            bounding_boxes.append((0, max(y-30, 0), min(x+image.shape[1], int(image.shape[1])), y+h+30))

    bounding_boxes.sort(key=lambda bbox: (bbox[1], bbox[0]))

    return bounding_boxes

def detailCreateBoudingBox(image, min_area_threshold):
    '''
    This function will create bounding box for detail area
    '''
    # Thresholding Image
    imgThresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5)

    # Find contours
    contours, _ = cv2.findContours(imgThresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get bounding boxes
    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h >= min_area_threshold:
            bounding_boxes.append((0, max(y-12, 0), min(x+image.shape[1], int(image.shape[1])), y+h+12))

    bounding_boxes.sort(key=lambda bbox: (bbox[1], bbox[0]))

    return bounding_boxes

def segmenter(image_path, contrast_factor=1.7, gamma_factor=1.0):
    '''
     This function identifies and isolates sections of Indonesian identity cards from images. 
     By detecting text and photo areas, it creates boundaries for province, nik, and personal information.
    '''
    assert isinstance(image_path, str), "image_path can be str() type only"

    try:
        # Attempt to open the file in read mode
        with open(image_path, "r"):
            # Original Image
            originalImage, provinsiArea, nikArea, detailArea = prepareRecognitionArea(image_path)

            # Preprocessed Image
            provinsiAreaPreprocessed = preprocessRecognitionArea(provinsiArea, contrast_factor=contrast_factor, gamma_factor=gamma_factor)
            nikAreaPreprocessed = preprocessRecognitionArea(nikArea, contrast_factor=contrast_factor, gamma_factor=gamma_factor)
            detailAreaPreprocessed = preprocessRecognitionArea(detailArea, contrast_factor=contrast_factor, gamma_factor=gamma_factor)

            # Pattern Image
            provinsiAreaPattern = findTextPattern(provinsiAreaPreprocessed, "provinsi")
            nikAreaPattern = findTextPattern(nikAreaPreprocessed, "nik")
            detailAreaPattern = findTextPattern(detailAreaPreprocessed, "detail")

            # Draw bounding boxes on the original image
            segmented_provinsi = []
            boundingBoxesRowsProvinsi = provinsiCreateBoudingBox(provinsiAreaPattern, 450)
            for bbox in boundingBoxesRowsProvinsi:
                x1, y1, x2, y2 = bbox
                segmented_provinsi.append(provinsiArea[y1:y2, x1:x2])

            segmented_nik = []
            boundingBoxesRowsNIK = nikCreateBoudingBox(nikAreaPattern, 450)
            for bbox in boundingBoxesRowsNIK:
                x1, y1, x2, y2 = bbox
                segmented_nik.append(nikArea[y1:y2, x1:x2])

            segmented_detail = []
            boundingBoxesRowsDetail = detailCreateBoudingBox(detailAreaPattern, 450)
            for bbox in boundingBoxesRowsDetail:
                x1, y1, x2, y2 = bbox
                segmented_detail.append(detailArea[y1:y2, x1:x2])

            return {
                "image": [originalImage],
                "provinsiArea": segmented_provinsi,
                "nikArea": segmented_nik,
                "detailArea": segmented_detail,
            }
    except FileNotFoundError:
        # If the file does not exist, FileNotFoundError is raised
        raise FileNotFoundError("File does not exist.")

def segmenter_ndarray(image, contrast_factor=1.7, gamma_factor=1.0):
    '''
     This function identifies and isolates sections of Indonesian identity cards from images. 
     By detecting text and photo areas, it creates boundaries for province, nik, and personal information.
    '''
    assert isinstance(image, np.ndarray), "image can be ndarray type only"

    try:
        # Original Image
        originalImage, provinsiArea, nikArea, detailArea = prepareRecognitionAreaNdarray(image)
        # Preprocessed Image
        provinsiAreaPreprocessed = preprocessRecognitionArea(provinsiArea, contrast_factor=contrast_factor, gamma_factor=gamma_factor)
        nikAreaPreprocessed = preprocessRecognitionArea(nikArea, contrast_factor=contrast_factor, gamma_factor=gamma_factor)
        detailAreaPreprocessed = preprocessRecognitionArea(detailArea, contrast_factor=contrast_factor, gamma_factor=gamma_factor)
        # Pattern Image
        provinsiAreaPattern = findTextPattern(provinsiAreaPreprocessed, "provinsi")
        nikAreaPattern = findTextPattern(nikAreaPreprocessed, "nik")
        detailAreaPattern = findTextPattern(detailAreaPreprocessed, "detail")
        # Draw bounding boxes on the original image
        segmented_provinsi = []
        boundingBoxesRowsProvinsi = provinsiCreateBoudingBox(provinsiAreaPattern, 450)
        for bbox in boundingBoxesRowsProvinsi:
            x1, y1, x2, y2 = bbox
            segmented_provinsi.append(provinsiArea[y1:y2, x1:x2])
        segmented_nik = []
        boundingBoxesRowsNIK = nikCreateBoudingBox(nikAreaPattern, 450)
        for bbox in boundingBoxesRowsNIK:
            x1, y1, x2, y2 = bbox
            segmented_nik.append(nikArea[y1:y2, x1:x2])
        segmented_detail = []
        boundingBoxesRowsDetail = detailCreateBoudingBox(detailAreaPattern, 450)
        for bbox in boundingBoxesRowsDetail:
            x1, y1, x2, y2 = bbox
            segmented_detail.append(detailArea[y1:y2, x1:x2])
        return {
            "image": [originalImage],
            "provinsiArea": segmented_provinsi,
            "nikArea": segmented_nik,
            "detailArea": segmented_detail,
        }
    except Exception as e:
        # If the file does not exist, FileNotFoundError is raised
        raise f"ERROR: {e}"

def detect_empty_list_in_dict(input_dict):
    '''
    Detects if any value in the given dictionary is an empty list.
    '''
    # Iterate over the values of the dictionary
    for value in input_dict.values():
        if isinstance(value, list) and not value:
            return True
    return False

def adaptive_segmenter(image_path, contrast_factor=1.7, delta_contrast=0.1, gamma_factor=1.0):
    '''
    This function will execute segmenter() with operation of contrast_factor if result contain []. contrast_factor will be reduced by delta_contrast
    '''
    contrast_factor = contrast_factor
    while (contrast_factor >= 1.0):
        result = segmenter(image_path, contrast_factor=contrast_factor, gamma_factor=gamma_factor)
        if detect_empty_list_in_dict(result):
            contrast_factor = round(contrast_factor - delta_contrast, 1)
        else:
            return result 

def adaptive_segmenter_ndarray(image, contrast_factor=1.7, delta_contrast=0.1, gamma_factor=1.0):
    '''
    This function will execute segmenter_ndarray() with operation of contrast_factor if result contain []. contrast_factor will be reduced by delta_contrast
    '''
    contrast_factor = contrast_factor
    while (contrast_factor >= 1.0):
        result = segmenter_ndarray(image, contrast_factor=contrast_factor, gamma_factor=gamma_factor)
        if detect_empty_list_in_dict(result):
            contrast_factor = round(contrast_factor - delta_contrast, 1)
        else:
            return result 

def getPassPhoto(image_path):
    '''
    This function will crop pass-photo from the image
    '''
    assert isinstance(image_path, str), "Image Path"
    try:
        # Open Image
        img = cv2.imread(image_path)

        # Resize Image to Standard Preprocessing Size
        imgResize = cv2.resize(img, (1000, 700))

        # Segment Identity Card (Indonesian IC) based on Image Ratio
        imgHeight, imgWidth = imgResize.shape[:2]
        # Param List Note: Matrix[[y, height], [x, width]]
        # passphoto = imgResize[int(imgHeight*0.16):, int(imgHeight*0.90):]
        passphoto = imgResize[int(imgHeight*0.16):int(imgHeight*0.70), int(imgHeight*0.93):]
        
        return passphoto
    except Exception as e:
        print(f"ERROR: {e}")
        return None

def getSignature(image_path):
    '''
    This function will crop signature from the image
    '''
    assert isinstance(image_path, str), "Image Path"
    try:
        # Open Image
        img = cv2.imread(image_path)

        # Resize Image to Standard Preprocessing Size
        imgResize = cv2.resize(img, (1000, 700))

        # Segment Identity Card (Indonesian IC) based on Image Ratio
        imgHeight, imgWidth = imgResize.shape[:2]
        # Param List Note: Matrix[[y, height], [x, width]]
        signature = imgResize[int(imgHeight*0.745):, int(imgHeight*0.90):]
        
        return signature
    except Exception as e:
        print(f"ERROR: {e}")
        return None

def getPassPhotoNdarray(image):
    assert isinstance(image, np.ndarray), "image can be ndarray type only"
    try:
        # Resize Image to Standard Preprocessing Size
        imgResize = cv2.resize(image, (1000, 700))

        # Segment Identity Card (Indonesian IC) based on Image Ratio
        imgHeight, imgWidth = imgResize.shape[:2]
        # Param List Note: Matrix[[y, height], [x, width]]
        # passphoto = imgResize[int(imgHeight*0.16):, int(imgHeight*0.90):]
        passphoto = imgResize[int(imgHeight*0.16):int(imgHeight*0.70), int(imgHeight*0.93):]
        
        return passphoto
    except Exception as e:
        print(f"ERROR: {e}")
        return None

def getSignatureNdarray(image):
    assert isinstance(image, np.ndarray), "image can be ndarray type only"
    try:
        # Resize Image to Standard Preprocessing Size
        imgResize = cv2.resize(image, (1000, 700))

        # Segment Identity Card (Indonesian IC) based on Image Ratio
        imgHeight, imgWidth = imgResize.shape[:2]
        # Param List Note: Matrix[[y, height], [x, width]]
        signature = imgResize[int(imgHeight*0.745):, int(imgHeight*0.90):]
        
        return signature
    except Exception as e:
        print(f"ERROR: {e}")
        return None