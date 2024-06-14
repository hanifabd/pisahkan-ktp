# **PISAHKAN KTP: Indonesian ID Card (KTP) Information Segmentation**

<center><img src="./assets/OIG1.jpg" alt="Beautiful Landscape" width="250"></center>


## **About**
`pisahkan_ktp` is a Python function that extracts province, NIK, and personal information from an image of an Indonesian National Identity Card (KTP). It utilizes image processing techniques to locate and isolate relevant sections of the KTP image, then extracts text data accurately. The extracted information is returned in a structured format, facilitating further processing or integration into other applications.

## **Requirements**
- Python 3.7 or Higher
- numpy
- opencv-python
- opencv-contrib-python
- pythonRLSA

## **Key Features**
- Extracts province, NIK, and personal information from Indonesian National Identity Card (KTP) images.
- Utilizes image processing techniques to locate and isolate relevant sections accurately.
- Returns extracted information in a structured format for easy integration and further processing.

## Usage
### Manual Installation via Github
1. Clone Repository
    ```
    git clone https://github.com/hanifabd/pisahkan-ktp
    ```
2. Installation
    ```
    cd pisahkan-ktp && pip install dist/pisahkan_ktp-0.2.11-py3-none-any.whl
    ```
### Installation Using Pip
1. Installation
    ```sh
    pip install pisahkan-ktp
    ```
### Inference
1. Usage
    - Text Area
        - Standard Segmenter
            ```py
            # Input ==> Image Path
            from pisahkan_ktp.ktp_segmenter import segmenter

            image_path = "./tests/sample.jpg"
            result = segmenter(image_path)
            print(result)

            # Input ==> Numpy Array Image ==> cv2.imread(image_path)
            from pisahkan_ktp.ktp_segmenter import segmenter_ndarray

            image_path = "./tests/sample.jpg"
            image = cv2.imread(image_path)
            result = segmenter_ndarray(image)
            print(result)
            ```
        
        - Adaptive Segmenter (Adjust contrast level in preprocessing)
            ```py
            # Input ==> Image Path
            from pisahkan_ktp.ktp_segmenter import adaptive_segmenter
    
            image_path = "./tests/sample.jpg"
            result = adaptive_segmenter(image_path, contrast_factor=1.7, delta_contrast=0.7, gamma_factor=1.0)
            print(result)
    
            # Input ==> Numpy Array Image ==> cv2.imread(image_path)
            from pisahkan_ktp.ktp_segmenter import adaptive_segmenter_ndarray
    
            image_path = "./tests/sample.jpg"
            image = cv2.imread(image_path)
            result = adaptive_segmenter_ndarray(image, contrast_factor=1.7, delta_contrast=0.7, gamma_factor=1.0)
            print(result)
            ```

    - Pass-Photo & Signature
        ```py
        # Input ==> Image Path
        from pisahkan_ktp.ktp_segmenter import getPassPhoto, getSignature

        image_path = "./tests/sample.jpg"
        result = getPassPhoto(image_path)
        # Output Image Numpy Array

        # Input ==> Numpy Array Image ==> cv2.imread(image_path)
        from pisahkan_ktp.ktp_segmenter import getPassPhotoNdarray, getSignatureNdarray

        image_path = "./tests/sample.jpg"
        image = cv2.imread(image_path)
        result = getPassPhotoNdarray(image)
        # Output Image Numpy Array
        ```
    > NOTE!!! Input image must be a clear Indonesian ID Card (KTP) no/less background noise for optimal performance

3. Result Text Area
    ```json
    {
        "image": [originalImage],
        "provinsiArea": [segmented_provinsi_img_matrix_list],
        "nikArea": [segmented_nik_img_matrix_list],
        "detailArea": [segmented_detail_img_matrix_list],
    }
    ```

4. Preview
    - Original Image

        <img src="./tests/sample.jpg" alt="Beautiful Landscape" width="500">

    - Provinsi Area Cropped
        
        <img src="./assets/8-5-provinsi.jpg" alt="Beautiful Landscape" width="500">
    
    - NIK Area Cropped
        
        <img src="./assets/8-6-nik.jpg" alt="Beautiful Landscape" width="500">

    - Detail Area Cropped
        
        <img src="./assets/8-7-detail.jpg" alt="Beautiful Landscape" width="500">

## How to Show in Matplotlib
### Input ==> Image Path
```py
from pisahkan_ktp.ktp_segmenter import segmenter
import matplotlib.pyplot as plt
import cv2

def show_result(result_dict):
    num_boxes = len(result_dict)
    fig, axes = plt.subplots(num_boxes, 1)
    if num_boxes == 1:
        axes = [axes]
    for i, bbox in enumerate(result_dict):
        ax = axes[i]
        if bbox.size:
            ax.imshow(cv2.cvtColor(bbox, cv2.COLOR_BGR2RGB))
            ax.axis('off')
    plt.tight_layout()
    plt.show()

image_path = "./tests/sample.jpg"
result = segmenter(image_path)

# Close pop up window first to see other result -> VSCODE
show_result(result["provinsiArea"])
show_result(result["nikArea"])
show_result(result["detailArea"])
```

### Input ==> Numpy Array Image ==> cv2.imread(image_path)
```py
from pisahkan_ktp.ktp_segmenter import segmenter_ndarray
import matplotlib.pyplot as plt
import cv2

def show_result(result_dict):
    num_boxes = len(result_dict)
    fig, axes = plt.subplots(num_boxes, 1)
    if num_boxes == 1:
        axes = [axes]
    for i, bbox in enumerate(result_dict):
        ax = axes[i]
        if bbox.size:
            ax.imshow(cv2.cvtColor(bbox, cv2.COLOR_BGR2RGB))
            ax.axis('off')
    plt.tight_layout()
    plt.show()

image_path = "./tests/sample.jpg"
image = cv2.imread(image_path)
result = segmenter_ndarray(image)

# Close pop up window first to see other result
show_result(result["provinsiArea"])
show_result(result["nikArea"])
show_result(result["detailArea"])
```