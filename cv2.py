import numpy as np

# Basic color conversion constants
COLOR_RGB2GRAY = 0
COLOR_BGR2GRAY = 1
COLOR_RGB2LAB = 2

# Normalization constants for matchers
NORM_L2 = 4
NORM_HAMMING = 6

# Morphology constants
MORPH_ELLIPSE = 2
MORPH_CLOSE = 3
MORPH_OPEN = 4
TM_CCOEFF_NORMED = 5

# Misc constants
RANSAC = 8

# Simple KeyPoint class
class KeyPoint:
    def __init__(self, x=0, y=0, size=1):
        self.pt = (float(x), float(y))
        self.size = float(size)

# Dummy match object
class DMatch:
    def __init__(self, queryIdx=0, trainIdx=0, distance=0.0):
        self.queryIdx = queryIdx
        self.trainIdx = trainIdx
        self.distance = float(distance)

# ----- Detector factory functions (return simple objects with detectAndCompute) -----
class _Detector:
    def __init__(self):
        pass
    def detectAndCompute(self, image, mask=None):
        # Return a couple of keypoints and random descriptors
        kp = [KeyPoint(10, 10, 1), KeyPoint(20, 20, 1)]
        desc = np.random.rand(len(kp), 128).astype(np.float32)
        return kp, desc

def SIFT_create(*args, **kwargs):
    return _Detector()

def ORB_create(*args, **kwargs):
    return _Detector()

def AKAZE_create(*args, **kwargs):
    return _Detector()

def createCLAHE(*args, **kwargs):
    class CLAHE:
        def apply(self, img):
            return np.array(img)
    return CLAHE()

# ----- Matcher classes -----
class BFMatcher:
    def __init__(self, normType, crossCheck=False):
        self.normType = normType
        self.crossCheck = crossCheck
    def match(self, desc1, desc2):
        n = min(len(desc1), len(desc2))
        return [DMatch(i, i, 0.5) for i in range(n)]
    def knnMatch(self, desc1, desc2, k=2):
        n = min(len(desc1), len(desc2))
        matches = []
        for i in range(n):
            row = [DMatch(i, j, 0.5 + j) for j in range(k)]
            matches.append(row)
        return matches

class FlannBasedMatcher:
    def __init__(self, index_params=None, search_params=None):
        pass
    def knnMatch(self, desc1, desc2, k=2):
        n = min(len(desc1), len(desc2))
        matches = []
        for i in range(n):
            row = [DMatch(i, j, 0.5 + j) for j in range(k)]
            matches.append(row)
        return matches

# ----- Geometric transforms -----
def estimateAffine2D(src, dst, method=RANSAC, ransacReprojThreshold=3):
    M = np.eye(2,3)
    mask = np.ones((len(src),1), dtype=np.uint8)
    return M, mask

def estimateAffinePartial2D(src, dst, method=RANSAC, ransacReprojThreshold=3):
    M = np.eye(2,3)
    mask = np.ones((len(src),1), dtype=np.uint8)
    return M, mask

def findHomography(src, dst, method=RANSAC, ransacReprojThreshold=3):
    M = np.eye(3)
    mask = np.ones((len(src),1), dtype=np.uint8)
    return M, mask

# ----- Image processing helpers -----
def cvtColor(image, flag):
    if flag in (COLOR_RGB2GRAY, COLOR_BGR2GRAY):
        if image.ndim == 3:
            gray = image[...,0]*0.299 + image[...,1]*0.587 + image[...,2]*0.114
            return gray.astype(image.dtype)
        return image
    if flag == COLOR_RGB2LAB:
        if image.ndim == 3:
            gray = image[...,0]*0.299 + image[...,1]*0.587 + image[...,2]*0.114
            lab = np.stack([gray, gray, gray], axis=-1)
            return lab.astype(image.dtype)
        lab = np.stack([image, image, image], axis=-1)
        return lab.astype(image.dtype)
    return image

def getStructuringElement(shape, ksize):
    return np.ones(ksize, dtype=np.uint8)

def morphologyEx(image, op, kernel):
    return image

def resize(image, dsize):
    return np.zeros(dsize[::-1], dtype=image.dtype)

def matchTemplate(a, b, method):
    return np.array([[1.0]])

def connectedComponents(image):
    return 0, np.zeros_like(image)

def dct(src):
    return np.fft.fft2(src)

def GaussianBlur(image, ksize, sigma):
    return image

def line(img, pt1, pt2, color, thickness):
    pass

def circle(img, center, radius, color, thickness):
    pass

def rectangle(img, pt1, pt2, color, thickness):
    pass
