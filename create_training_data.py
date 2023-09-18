import os
import sys
from utils import utils

if __name__ == "__main__":
    data_dir = os.path.join(os.getcwd(), "All_Data")
    out_dir = os.path.join(os.getcwd(), "Training_Data")
    num_images = 10
    method = "even"
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            sides = arg.split('=')
            if len(sides) != 2:
                print("Arg: " + arg + "needs an equals character")
            keyword = sides[0].strip()
            val = sides[1].strip()
            if keyword == "in_dirname":
                data_dir = os.path.join(os.getcwd(), val)
            elif keyword == "in_dir_absolute":
                data_dir = val
            elif keyword == "num_images":
                num_images = int(val)
            elif keyword == "out_dirname":
                out_dir = os.path.join(os.getcwd(), val)
            elif keyword == "out_dir_absolute":
                out_dir = val
            elif keyword == "method":
                method = val
            else:
                print("Invalid Keyword!: " + keyword)

    utils.get_n_images_per_video(num_images, data_dir, out_dir, method=method)
