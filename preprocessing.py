import numpy as np

from PIL import Image, ImageEnhance


def random_augment_image(path, rfactor=0.17):

    '''
    Returns PIL image.
    '''

    def _adjust_random_sharpness(image, low=-3.0, high=4.0):
        factor = np.random.uniform(low, high)
        sharpness = ImageEnhance.Sharpness(image)
        result_image = sharpness.enhance(factor)
        return result_image


    def _adjust_random_brightness(image, low=0.2, high=3.8):
        factor = np.random.uniform(low, high)
        sharpness = ImageEnhance.Brightness(image)
        result_image = sharpness.enhance(factor)
        return result_image


    def _adjust_random_contrast(image, low=0.3, high=2.0):
        factor = np.random.uniform(low, high)
        sharpness = ImageEnhance.Contrast(image)
        result_image = sharpness.enhance(factor)
        return result_image


    def _adjust_random_color(image, low=-3.0, high=4.0):
        factor = np.random.uniform(low, high)
        sharpness = ImageEnhance.Color(image)
        result_image = sharpness.enhance(factor)
        return result_image

    def _adjust_random_rotation(image, max_angle=360):
        angle = np.random.randint(low=0, high=max_angle)
        return image.rotate(angle)

    image = Image.open(path).convert('RGB')

    if np.random.rand() > rfactor:
        image = _adjust_random_brightness(image, low=0.7, high=2.0)
    if np.random.rand() > rfactor:
        image = _adjust_random_color(image, low=0.7, high=2.)
    if np.random.rand() > rfactor:
        image = _adjust_random_contrast(image)
    if np.random.rand() > rfactor:
        image = _adjust_random_sharpness(image)
    if np.random.rand() > rfactor:
        image = _adjust_random_rotation(image)
    
    return image