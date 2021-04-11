import numpy as np

from PIL import Image, ImageEnhance


def image_to_numpy(image):

    '''
    Convert image to numpy array.

    Args:
        image: PIL.Image. Image for converting.

    Returns:
        Numpy array with shape (height, width, channels).
    '''

    w, h = image.size
    image_data = np.array(image.getdata())
    channels = image_data.shape[-1]
    return np.reshape(image_data, (h, w, channels))


def central_crop(image, size):

    '''
    '''

    w, h = image.size
    target_h, target_w = size
    left = int(w/2 - target_w/2)
    right = left + target_w
    upper = int(h/2 - target_h/2)
    lower = upper + target_h
    return image.crop((left, upper, right, lower))


def load_image_from_path(path, format='RGB', target_size=None):

    '''
    Load image from path with its's original size in RGB format by default.

    Args:
        path: string. Local path to image.
        format: string. Formats to attempt to load the file in. Default 'RGB'.
        target_size: tuple. If non None, resize image to target size. Default None. 

    Returns:
        PIL.Image object.
    '''

    image = Image.open(path).convert(format)
    if target_size is not None:
        image = image.resize(target_size)
    return image


def random_augment_image(image, rfactor=0.14):

    '''
    Returns PIL image.

    Args:
        image: PIL.Image. Image for augmentations.
        rfactor: float. If random values < rfactor then function returns non-augmented image.

    Returns:
        PIL.Image object.

    '''

    def _adjust_random_perspective(image):
    
        def _find_coeffs(pa, pb):
            matrix = []
            for p1, p2 in zip(pa, pb):
                matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
                matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

            A = np.matrix(matrix, dtype=np.float)
            B = np.array(pb).reshape(8)
            res = np.linalg.solve(A, B)
            return np.reshape(res, 8)

        w, h = image.size
        current_coordinates = np.array([
            [0, 0],
            [0, w],
            [h, w],
            [h, 0],
        ])
        deltas = np.random.randint(size=8, low=0, high=int(0.25*np.max([w, h])))
        target_coordinates = np.array([
            [0+deltas[0], 0+deltas[1]],
            [0+deltas[2], w-deltas[3]],
            [h-deltas[4], w-deltas[5]],
            [h-deltas[6], 0+deltas[7]],
        ])
        
        coeffs = _find_coeffs(current_coordinates, target_coordinates)
        result_image = image.transform(image.size, Image.PERSPECTIVE, coeffs, Image.BICUBIC)
        return result_image
    

    def _adjust_random_sharpness(image, low=-3.0, high=5.0):
        factor = np.random.uniform(low, high)
        sharpness = ImageEnhance.Sharpness(image)
        result_image = sharpness.enhance(factor)
        return result_image


    def _adjust_random_brightness(image, low=0.7, high=1.3):
        factor = np.random.uniform(low, high)
        sharpness = ImageEnhance.Brightness(image)
        result_image = sharpness.enhance(factor)
        return result_image


    def _adjust_random_contrast(image, low=0.8, high=1.2):
        factor = np.random.uniform(low, high)
        sharpness = ImageEnhance.Contrast(image)
        result_image = sharpness.enhance(factor)
        return result_image


    def _adjust_random_color(image, low=0.8, high=1.2):
        factor = np.random.uniform(low, high)
        sharpness = ImageEnhance.Color(image)
        result_image = sharpness.enhance(factor)
        return result_image
    

    def _adjust_random_rotation(image, max_angle=360):
        angle = np.random.randint(low=0, high=max_angle)
        return image.rotate(angle)


    if np.random.rand() > rfactor:
        if np.random.rand() > rfactor:
            image = _adjust_random_perspective(image)
        if np.random.rand() > rfactor:
            image = _adjust_random_sharpness(image)
        if np.random.rand() > rfactor:
            image = _adjust_random_brightness(image)
        if np.random.rand() > rfactor:
            image = _adjust_random_contrast(image)
        if np.random.rand() > rfactor:
            image = _adjust_random_color(image)
        if np.random.rand() > rfactor:
            image = _adjust_random_rotation(image)
    
    return image