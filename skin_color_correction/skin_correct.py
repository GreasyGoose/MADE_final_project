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
parser.add_argument("--head_img", type=str, default=r"sample_img\head\head.png", help="Path to head img")
parser.add_argument("--head_mask", type=str, default=r"sample_img\head\head_mask_lip.png",
                    help="Path to head segmentation mask img")

parser.add_argument("--body_img", type=str, default=r"sample_img\body\body.jpg", help="Path to body img")
parser.add_argument("--body_mask", type=str, default=r"sample_img\body\body_mask_atr.png",
                    help="Path to body segmentation mask img")
parser.add_argument("--output_dir", type=str, default=r"output", help="Output path")
args = parser.parse_args()


def get_mask(mask_img, del_labels, verbose=False, verbose_name=""):
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
        cv2.imwrite(os.path.join(args.output_dir, verbose_name + "_skin_mask.png"), mask_img)
        print(f"Skin mask saved: {verbose_name}_skin_mask.png")
    return mask_img



def get_skin_color(img, mask):
            target_pixels = img[mask != [0, 0, 0]] # pixels under mask
            target_pixels = target_pixels.reshape((len(target_pixels) // 3, 3))
            clt = KMeans(n_clusters=4)
            clt.fit(target_pixels)
            hist = centroid_histogram(clt)
            skin_color = get_color(hist, clt.cluster_centers_)
            # Return the color.
            return np.int16(skin_color)


def skin_mask_refinement(skin_color, img, mask, verbose=False):
    skin_color_hsv = cv2.cvtColor(np.uint8([[skin_color]]), cv2.COLOR_BGR2HSV)[0][0]
    img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
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
        final_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) * color_skinMask
        if verbose:
            cv2.imwrite(os.path.join(args.output_dir, "refined_mask.png"), final_mask)
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
            print("Cannot find {}: {}".format(arg, getattr(args, arg)))
            exit()

    mask_body = cv2.imread(args.body_mask, -1)
    mask_body = get_mask(mask_img=mask_body, del_labels="LIP_DEL" if args.body_mask.endswith("_lip.png") else "ATR_DEL", verbose=VERBOSE, verbose_name="body")

    mask_face = cv2.imread(args.head_mask, -1)
    mask_face = get_mask(mask_img=mask_face, del_labels="LIP_DEL" if args.body_mask.endswith("_lip.png") else "ATR_DEL", verbose=VERBOSE, verbose_name="head")

    img_face = cv2.imread(args.head_img, -1)
    if img_face.shape[-1] > 3:
        trans_mask = img_face[:, :, 3] == 0
        img_face[trans_mask] = [255, 255, 255, 255]
        img_face = cv2.cvtColor(img_face, cv2.COLOR_BGRA2BGR)

    skin_color_face_BGR = get_skin_color(img_face, mask_face)

    img_body = cv2.imread(args.body_img, -1)
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
        cv2.imwrite(os.path.join(args.output_dir, "body_img.png"), img_body)
        cv2.imwrite(os.path.join(args.output_dir, "res_img.png"), res_img)
        cv2.imwrite(os.path.join(args.output_dir, "no_skin_img.png"), no_skin_img)
        cv2.imwrite(os.path.join(args.output_dir, "skin_img.png"), skin_img)
        print("Skin/no skin img saved")
    skin_swap = cv2.add(no_skin_img, skin_img)
    cv2.imwrite(os.path.join(args.output_dir, os.path.basename(args.head_img)), skin_swap)
    print("Finished")

