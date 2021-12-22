
import numpy as np

def centroid_histogram(clt):
    # Grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster.
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    # Normalize the histogram, such that it sums to one.
    hist = hist.astype("float")
    hist /= hist.sum()

    # Return the histogram.
    return hist


def get_color(hist, centroids):
    # Obtain the color with maximum percentage of area covered.
    maxi = 0
    dominant_color = [0, 0, 0]

    # Loop over the percentage of each cluster and the color of
    # each cluster.
    for (percent, color) in zip(hist, centroids):
        if (percent > maxi):
            dominant_color = color
            maxi = percent

    # Return the most dominant color.
    return dominant_color

def skinRange(HSV):
    H, S, V = HSV
    e8 = (H <= 25) and (H >= 0)
    e9 = (S < 174) and (S > 50)
    e10 = (V <= 255) and (V >= 50)
    return (e8 and e9 and e10)

def make_lower_upper(skin_color, Hue, Saturation, Value):
    # Hue values are in the range [0..180]
    if (skin_color[0] > Hue):
        if (skin_color[0] > (180 - Hue)):
            if (skin_color[1] > Saturation + 10):
                lower1 = np.array([skin_color[0] - Hue, skin_color[1] - Saturation, Value], dtype="uint8")
                upper1 = np.array([180, 255, 255], dtype="uint8")
                lower2 = np.array([0, skin_color[1] - Saturation, Value], dtype="uint8")
                upper2 = np.array([(skin_color[0] + Hue) % 180, 255, 255], dtype="uint8")
                return (True, lower1, upper1, lower2, upper2)
            else:
                lower1 = np.array([skin_color[0] - Hue, 10, Value], dtype="uint8")
                upper1 = np.array([180, 255, 255], dtype="uint8")
                lower2 = np.array([0, 10, Value], dtype="uint8")
                upper2 = np.array([(skin_color[0] + Hue) % 180, 255, 255], dtype="uint8")
                return (True, lower1, upper1, lower2, upper2)
        else:
            if (skin_color[1] > Saturation + 10):
                lower = np.array([skin_color[0] - Hue, skin_color[1] - Saturation, Value], dtype="uint8")
                upper = np.array([skin_color[0] + Hue, 255, 255], dtype="uint8")
                return (False, lower, upper)
            else:
                lower = np.array([skin_color[0] - Hue, 10, Value], dtype="uint8")
                upper = np.array([skin_color[0] + Hue, 255, 255], dtype="uint8")
                return (False, lower, upper)
    else:
        if (skin_color[1] > Saturation + 10):
            lower1 = np.array([0, skin_color[1] - Saturation, Value], dtype="uint8")
            upper1 = np.array([skin_color[0] + Hue, 255, 255], dtype="uint8")
            lower2 = np.array([180 - Hue + skin_color[0], skin_color[1] - Saturation, Value], dtype="uint8")
            upper2 = np.array([180, 255, 255], dtype="uint8")
            return (True, lower1, upper1, lower2, upper2)
        else:
            lower1 = np.array([0, 10, Value], dtype="uint8")
            upper1 = np.array([skin_color[0] + Hue, 255, 255], dtype="uint8")
            lower2 = np.array([180 - Hue + skin_color[0], 10, Value], dtype="uint8")
            upper2 = np.array([180, 255, 255], dtype="uint8")
            return (True, lower1, upper1, lower2, upper2)

def doDiff(img, target_color, src_color, size):
    img = img.astype(np.float32)
    diff01 = target_color[0] / src_color[0]
    diff02 = (255 - target_color[0]) / (255 - src_color[0])
    diff03 = (255 * (target_color[0] - src_color[0])) / (255 - src_color[0])
    diff11 = target_color[1] / src_color[1]
    diff12 = (255 - target_color[1]) / (255 - src_color[1])
    diff13 = (255 * (target_color[1] - src_color[1])) / (255 - src_color[1])
    diff21 = target_color[2] / src_color[2]
    diff22 = (255 - target_color[2]) / (255 - src_color[2])
    diff23 = (255 * (target_color[2] - src_color[2])) / (255 - src_color[2])
    diff1 = [diff01, diff11, diff21]
    diff2 = [diff02, diff12, diff22]
    diff3 = [diff03, diff13, diff23]

    for k in range(3):
        mask = img[:, :, k] < src_color[k]
        img[mask, k] *= diff1[k]
        img[~mask, k] *= diff2[k]
        img[~mask, k] += diff3[k]

    return (np.clip(img, 0, 255)).astype(np.uint8)