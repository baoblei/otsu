import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def apply_threshold(img, threshold):
    """
    根据阈值对图像进行二值化
    """
    binary_img = (img >= threshold).astype(np.uint8) * 255
    return binary_img

def _otsu_threshold(data):
    hist, bins = np.histogram(data, bins=50) # hist 为灰度直方图，bins 为灰度值
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # OTSU算法实现
    total = hist.sum()
    sum_total = sum(bin_centers * hist)
    
    max_variance = 0
    threshold = 0
    
    sum_b = 0
    count_b = 0
    
    for i in range(len(hist)):
        count_b += hist[i]
        if count_b == 0:
            continue
            
        count_f = total - count_b
        if count_f == 0:
            break
            
        sum_b += bin_centers[i] * hist[i]
        mean_b = sum_b / count_b
        mean_f = (sum_total - sum_b) / count_f
        
        variance = count_b * count_f * (mean_b - mean_f) ** 2
        if variance > max_variance:
            max_variance = variance
            threshold = bin_centers[i]
            
    return threshold

if __name__ == "__main__":
    image = Image.open('img/org.jpg').convert("L")
    img_array = np.asarray(image)
    threshold = _otsu_threshold(img_array)
    print("Threshold:", threshold)
    binary_image = apply_threshold(img_array, threshold)

    plt.imshow(binary_image, cmap='gray')
    plt.savefig("img/otsu_result.jpg")
