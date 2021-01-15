from glob import glob

import os, cv2
import numpy as np


def main():
    os.chdir('./dataset_utils')
    target_dir="./0924_results(sirip)"
    index_path = os.path.join(target_dir, 'index.html')
    index = open(index_path, "w")
    index.write("<html><body><table><tr>")
    index.write("<th>name</th><th>input</th><th>only SNU</th><th>SNU+Sirip</th><th>label</th></tr>")

    fp_paths = glob("%s/01/*.png"%target_dir)
    for fp_path in fp_paths:
        cur_file = os.path.split(fp_path)[-1]
        index.write("<td>%s</td>" % os.path.basename(fp_path))
        index.write("<td><img src='./01_resize/%s' " % (cur_file) + "style=\"width:100%\"></td>")
        index.write("<td><img src='./02_resize/%s' " % (cur_file) + "style=\"width:100%\"></td>")
        index.write("<td><img src='./03_resize/%s' " % (cur_file) + "style=\"width:100%\"></td>")
        index.write("<td><img src='./04_resize/%s' " % (cur_file) + "style=\"width:100%\"></td>")

        #index.write("<td><img src='./01_resize/%s'></td>" % (cur_file))
        #index.write("<td><img src='./02_resize/%s'></td>" % (cur_file))
        #index.write("<td><img src='./03_resize/%s'></td>" % (cur_file))
        #index.write("<td><img src='./04_resize/%s'></td>" % (cur_file))
        index.write("</tr>")
    index.close()


# sirip_resize("./0924_results(sirip)/01")
# sirip_resize("./0924_results(sirip)/02", scale=1/2.5, is_label=True)
def sirip_resize(target_dir, scale=1/5, is_label=False):
    output_dir = "%s_resize" % (target_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    target_paths = glob("%s/*.png" % target_dir)
    ii, iii = 1, len(target_paths)
    for target_path in target_paths:
        print("[%d/%d] %s" % (ii, iii, target_path))
        ii += 1

        if not is_label:
            cur_img = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
        elif is_label:
            cur_img = cv2.imread(target_path)

        cur_name = os.path.split(target_path)[-1]
        if not is_label:
            new_img = cv2.resize(cur_img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        elif is_label:
            new_img = cv2.resize(cur_img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite("%s/%s" % (output_dir, cur_name), new_img)
    return



if __name__ == '__main__':
    main()