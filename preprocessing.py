import numpy as np

from PIL import Image, ImageEnhance


def random_augment_image(path, rfactor=0.14):

    '''
    Returns PIL image.

    Params:
        path: string. Path to image. Image default converts to RGB formats automaticaly.
        rfactor: float. 

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
        deltas = np.random.randint(size=8, low=0, high=int(0.3*np.max([w, h])))
        target_coordinates = np.array([
            [0+deltas[0], 0+deltas[1]],
            [0+deltas[2], w-deltas[3]],
            [h-deltas[4], w-deltas[5]],
            [h-deltas[6], 0+deltas[7]],
        ])
        
        coeffs = _find_coeffs(current_coordinates, target_coordinates)
        result_image = image.transform(image.size, Image.PERSPECTIVE, coeffs, Image.BICUBIC)
        return result_image
    

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
    if np.random.rand() > rfactor:
        image = _adjust_random_perspective(image)
    
    return image