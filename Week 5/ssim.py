from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2gray
from skimage.transform import resize

def compute_ssim(image1, image2, resize_second=True):
    image1_gray = rgb2gray(image1)
    if resize_second:
        image2 = resize(image2, (image1.shape[0], image1.shape[1]), anti_aliasing=True)
    image2_gray = rgb2gray(image2)
    return ssim(image1_gray, image2_gray)
