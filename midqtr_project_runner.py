"""
DSC 20 Mid-Quarter Project Runner (cv2)
"""
# pylint: disable = E1101

import cv2
import numpy as np
from midqtr_project import (
    RGBImage,
    ImageProcessing as IP,
    ImageKNNClassifier,
)


def img_read(path):
    """
    Read the image with given `path` to a RGBImage instance.
    """
    mat = cv2.imread(path).transpose(2, 0, 1).tolist()
    mat.reverse()  # convert BGR (cv2 default behavior) to RGB
    return RGBImage(mat)


def img_save(path, image):
    """
    Save a RGBImage instance (`image`) as a image file with given `path`.
    """
    mat = np.stack(list(reversed(image.get_pixels()))).transpose(1, 2, 0)
    cv2.imwrite(path, mat)


def create_random_pixels(low, high, nrows, ncols):
    """
    Create a random pixels matrix with dimensions of
    3 (channels) x `nrows` x `ncols`, and fill in integer
    values between `low` and `high` (both exclusive).
    """
    return np.random.randint(low, high + 1, (3, nrows, ncols)).tolist()


def pixels_example():
    """
    An example of the 3-dimensional pixels matrix (3 x 5 x 10).
    """
    return [
        [
            # channel 0: red (5 rows x 10 columns)
            [206, 138, 253, 211, 102, 194, 188, 188, 120, 231],
            [204, 208, 220, 214, 203, 165, 249, 225, 198, 185],
            [113, 196, 133, 235, 173, 179, 252, 105, 214, 238],
            [152, 156, 143, 114, 166, 132, 106, 115, 116, 177],
            [231, 193, 123, 154, 184, 242, 226, 155, 222, 223],
        ],
        [
            # channel 1: green (5 rows x 10 columns)
            [214, 190, 173, 141, 248, 189, 105, 193, 125, 122],
            [209, 136, 131, 187, 177, 186, 239, 222, 175, 152],
            [239, 236, 177, 243, 183, 192, 114, 211, 147, 192],
            [168, 119, 120, 182, 190, 108, 181, 219, 198, 127],
            [251, 222, 205, 102, 104, 217, 234, 196, 131, 127],
        ],
        [
            # channel 2: blue (5 rows x 10 columns)
            [233, 188, 214, 175, 152, 174, 235, 174, 234, 149],
            [163, 169, 131, 209, 232, 180, 238, 224, 152, 214],
            [137, 135, 181, 146, 243, 210, 236, 107, 193, 200],
            [230, 233, 206, 227, 150, 131, 177, 187, 143, 150],
            [117, 188, 127, 166, 134, 219, 241, 108, 217, 202],
        ],
    ]


def image_processing_test_examples():
    """
    Examples of image processing methods tests using real images.
    """
    # read image
    dsc20_img = img_read("img/dsc20.png")

    # negate and save
    negative_dsc20_img = IP.negate(dsc20_img)
    img_save("img/out/dsc20_negate.png", negative_dsc20_img)
    
    # tint
    tint_dsc20_img = IP.tint(dsc20_img, color=(0,0,100))
    img_save("img/out/dsc20_tint.png", tint_dsc20_img)
    
    # channel
    channel_img = IP.clear_channel(dsc20_img, 0)
    img_save("img/out/dsc20_channel.png", channel_img)
    
    #crop
    crop_image = IP.crop(dsc20_img, 50, 75, (75, 50))
    another_crop = IP.crop(dsc20_img,100, 50, (100, 150))

    img_save("img/out/dsc20_crop.png", crop_image)

    
    # chroma key with a background image and save
    bg_img = img_read("img/blue_gradient.png")
    chroma_white_dsc20_img = IP.chroma_key(dsc20_img, bg_img, (255, 205, 210))
    img_save("img/out/dsc20_chroma_white.png", chroma_white_dsc20_img)
    
    #rotate 180
    rotate_img = IP.rotate_180(dsc20_img)
    img_save("img/out/dsc20_rotate_180.png", rotate_img)


def knn_test_examples():
    """
    Examples of KNN classifier tests.
    """
    # make random training data (type: List[Tuple[RGBImage, str]])
    train = []
    # create training images with low intensity values
    train.extend(
        (RGBImage(create_random_pixels(0, 75, 300, 300)), "low")
        for _ in range(20)
    )
    # create training images with high intensity values
    train.extend(
        (RGBImage(create_random_pixels(180, 255, 300, 300)), "high")
        for _ in range(20)
    )

    # initialize and fit the classifier
    knn = ImageKNNClassifier(5)
    knn.fit(train)

    # should be "low"
    print(knn.predict(RGBImage(create_random_pixels(0, 75, 300, 300))))
    # can be either "low" or "high"
    print(knn.predict(RGBImage(create_random_pixels(75, 180, 300, 300))))
    # should be "high"
    print(knn.predict(RGBImage(create_random_pixels(180, 255, 300, 300))))
