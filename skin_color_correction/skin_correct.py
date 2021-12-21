import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--head_img", type=str, default=r"sample_img\head\head.png", help="Path to head img")
parser.add_argument("--head_mask", type=str, default=r"sample_img\head\head_mask_lip.png", help="Path to head segmentation mask img")

parser.add_argument("--body_img", type=str, default=r"sample_img\body\body.jpg", help="Path to body img")
parser.add_argument("--body_mask", type=str, default=r"sample_img\body\body_mask_atr.png", help="Path to body segmentation mask img")
parser.add_argument("--output_dir", type=str, default=r"output", help="Output path")
args = parser.parse_args()



if __name__ == "__main__":

    os.makedirs(args.output_dir, exist_ok=True)

    # check input arguments
    for arg in vars(args):
        if not os.path.exists(getattr(args, arg)):
            print("Cannot find {}: {}".format(arg, getattr(args, arg)))
            exit()
