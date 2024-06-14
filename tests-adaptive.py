from src.pisahkan_ktp.ktp_segmenter import adaptive_segmenter
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

image_path = "./tests/sample2.jpg"
result = adaptive_segmenter(image_path, contrast_factor=1.7, delta_contrast=0.7, gamma_factor=1.0)
print(result)

# Close pop up window first to see other result
show_result(result["provinsiArea"])
show_result(result["nikArea"])
show_result(result["detailArea"])

# -----------------------------------------------------------------------------------------------------------------------------

from src.pisahkan_ktp.ktp_segmenter import adaptive_segmenter_ndarray
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

image_path = "./tests/sample2.jpg"
image = cv2.imread(image_path)
result = adaptive_segmenter_ndarray(image, contrast_factor=1.7, delta_contrast=0.7, gamma_factor=1.0)

# Close pop up window first to see other result
show_result(result["provinsiArea"])
show_result(result["nikArea"])
show_result(result["detailArea"])