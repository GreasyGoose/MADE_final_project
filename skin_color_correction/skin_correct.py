import argparse
import os
import json
import cv2
import numpy as np
from sklearn.cluster import KMeans
from utils import *

DEFAULT_LABELS = "mask_labels.json"
VERBOSE = False

parser = argparse.ArgumentParser()
parser.add_argument("--head_img", type=str, default=r"sample_img\head", help="Path to folder with head img")
parser.add_argument("--head_mask", type=str, default=r"sample_img\head_mask_lip",
                    help="Path to folder with head segmentation mask img")

parser.add_argument("--body_img", type=str, default=r"sample_img\body", help="Path to folder with body img")
parser.add_argument("--body_mask", type=str, default=r"sample_img\body_mask_atr",
                    help="Path to folder with body segmentation mask img")
parser.add_argument("--output_dir", type=str, default=r"output", help="Output path")
args = parser.parse_args()


def get_mask(mask_img, del_labels, verbose=False, verbose_name=""):
    """
    :param mask_img: image of mask (ATR or LIP)
    :param del_labels: name of labels to delete from label dict [labels in dict should be in RGB format]
    :param verbose: save intermediate results in output folder if True
    :param verbose_name: name of saved intermediate file
    :return: mask img of 1 and 0
    """
    f = open(DEFAULT_LABELS)
    labels = json.load(f)
    f.close()
    for key in labels[del_labels]:
        current_color = np.array(labels[del_labels][key])
        current_color = current_color[::-1]
        idxs = np.where(np.all(mask_img == current_color, axis=-1))
        if len(idxs) == 2:
            mask_img[idxs[0], idxs[1]] = 0
    mask_img[np.any(mask_img != [0, 0, 0], axis=-1)] = [1, 1, 1]
    if verbose:
        cv2.imwrite(os.path.join(args.output_dir, verbose_name + "_skin_mask.png"), 255 * mask_img)
        print(f"Skin mask saved: {verbose_name}_skin_mask.png")
    return mask_img



def get_skin_color(img, mask):
    target_pixels = img[mask != [0, 0, 0]] # pixels under mask
    target_pixels = target_pixels.reshape((len(target_pixels) // 3, 3))
    clt = KMeans(n_clusters=4)
    clt.fit(target_pixels)
    hist = centroid_histogram(clt)
    skin_color = get_color(hist, clt.cluster_centers_)
    return np.int16(skin_color)


def skin_mask_refinement(skin_color, img, mask, verbose=False):
    skin_color_hsv = cv2.cvtColor(np.uint8([[skin_color]]), cv2.COLOR_BGR2HSV)[0][0]
    img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    blurred_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    blurred_mask = cv2.pyrUp(255 * blurred_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    blurred_mask = cv2.morphologyEx(blurred_mask, cv2.MORPH_DILATE, kernel)

    for i in range(0, 15):
        blurred_mask = cv2.medianBlur(blurred_mask, 7)
    blurred_mask = cv2.pyrDown(blurred_mask)
    _, blurred_mask = cv2.threshold(blurred_mask, 150, 255, cv2.THRESH_BINARY)
    blurred_mask = np.clip(blurred_mask / 255.0, 0, 1).astype(np.uint8)

    if skinRange(skin_color_hsv):
        Hue = 10
        Saturation = 65
        Value = 50
        result = make_lower_upper(skin_color_hsv, Hue, Saturation, Value)
        if (result[0]):
            # print("I'm here")
            lower1 = result[1]
            upper1 = result[2]
            lower2 = result[3]
            upper2 = result[4]
            color_skinMask1 = cv2.inRange(img_HSV, lower1, upper1)
            color_skinMask2 = cv2.inRange(img_HSV, lower2, upper2)
            color_skinMask = cv2.bitwise_or(color_skinMask1, color_skinMask2)
        else:
            lower = result[1]
            upper = result[2]
            color_skinMask = cv2.inRange(img_HSV, lower, upper)
        final_mask = blurred_mask * color_skinMask
        if verbose:
            cv2.imwrite(os.path.join(args.output_dir, "03_refined_skin_mask.png"), final_mask)
            print("Refined mask saved.")
        return final_mask
    else:
        print("Skin color out of default range. No mask refinement.")
        return mask




if __name__ == "__main__":


    os.makedirs(args.output_dir, exist_ok=True)

    # check input arguments
    for arg in vars(args):
        if not os.path.exists(getattr(args, arg)):
            print("Cannot find folder {}: {}".format(arg, getattr(args, arg)))
            exit()
        elif len(os.listdir(getattr(args, arg))) < 1:
            print("Folder is empty: {}".format(getattr(args, arg)))
            exit()

    mask_body_fname = os.path.join(args.body_mask, os.listdir(args.body_mask)[0])
    mask_body = cv2.imread(mask_body_fname, -1)
    mask_body = get_mask(mask_img=mask_body, del_labels="LIP_DEL" if args.body_mask.endswith("_lip.png") else "ATR_DEL", verbose=VERBOSE, verbose_name="01_body")

    mask_face_fname = os.path.join(args.head_mask, os.listdir(args.head_mask)[0])
    mask_face = cv2.imread(mask_face_fname, -1)
    mask_face = get_mask(mask_img=mask_face, del_labels="LIP_DEL" if args.body_mask.endswith("_lip.png") else "ATR_DEL", verbose=VERBOSE, verbose_name="02_head")

    img_face_fname = os.path.join(args.head_img, os.listdir(args.head_img)[0])
    img_face = cv2.imread(img_face_fname, -1)
    if img_face.shape[-1] > 3:
        trans_mask = img_face[:, :, 3] == 0
        img_face[trans_mask] = [255, 255, 255, 255]
        img_face = cv2.cvtColor(img_face, cv2.COLOR_BGRA2BGR)

    skin_color_face_BGR = get_skin_color(img_face, mask_face)

    img_body_fname = os.path.join(args.body_img, os.listdir(args.body_img)[0])
    img_body = cv2.imread(img_body_fname, -1)
    if img_body.shape[-1] > 3:
        trans_mask = img_body[:, :, 3] == 0
        img_body[trans_mask] = [255, 255, 255, 255]
        img_body = cv2.cvtColor(img_body, cv2.COLOR_BGRA2BGR)

    skin_color_body_BGR = get_skin_color(img_body, mask_body)
    mask_body = skin_mask_refinement(skin_color=skin_color_body_BGR, img=img_body, mask=mask_body, verbose=VERBOSE)


    # Change the color maintaining the texture.
    res_img = doDiff(img=np.copy(img_body),
                     target_color=skin_color_face_BGR,
                     src_color=skin_color_body_BGR,
                     size=img_body.shape)

    # Get the two images ie. the skin and the background.
    no_skin_img = cv2.bitwise_and(img_body, img_body, mask=cv2.bitwise_not(mask_body))
    skin_img = cv2.bitwise_and(res_img, res_img, mask=mask_body)
    if VERBOSE:
        cv2.imwrite(os.path.join(args.output_dir, "04_body_img.png"), img_body)
        cv2.imwrite(os.path.join(args.output_dir, "05_no_skin_img.png"), no_skin_img)
        cv2.imwrite(os.path.join(args.output_dir, "06_skin_img.png"), skin_img)
        cv2.imwrite(os.path.join(args.output_dir, "07_color_corrected.png"), res_img)
        print("Skin/no skin img saved")

    skin_swap = cv2.add(no_skin_img, skin_img)
    cv2.imwrite(os.path.join(args.output_dir, os.path.basename(img_body_fname)), skin_swap)
    print("Finished")

