import cv2
from collections import Counter
import scipy.sparse as sps
import csv

from glob import glob
import os
import time
import numpy as np

def ttmp():
    ## 0611
    os.chdir('./dataset_utils')

    image_path='./tmp/1-1.png'
    img1 = cv2.imread(image_path)
    img2 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    img1_1024_Area = cv2.resize(img1, (1024, 1024), interpolation=cv2.INTER_AREA)
    img2_1024_Area = cv2.resize(img2, (1024, 1024), interpolation=cv2.INTER_AREA)
    img1_1024_Near = cv2.resize(img1, (1024, 1024), interpolation=cv2.INTER_NEAREST)
    img2_1024_Near = cv2.resize(img2, (1024, 1024), interpolation=cv2.INTER_NEAREST)

    cv2.imwrite("./tmp/%s.png" % 'img1_1024_Area', img1_1024_Area)
    cv2.imwrite("./tmp/%s.png" % 'img2_1024_Area', img2_1024_Area)
    cv2.imwrite("./tmp/%s.png" % 'img1_1024_Near', img1_1024_Near)
    cv2.imwrite("./tmp/%s.png" % 'img2_1024_Near', img2_1024_Near)

    img1_512_Area = cv2.resize(img1, (512, 512), interpolation=cv2.INTER_AREA)
    cv2.imwrite("./tmp/%s.png" % 'img1_512_Area', img1_512_Area)
    img1_4000_Area = cv2.resize(img1, (4000, 4000), interpolation=cv2.INTER_AREA)
    cv2.imwrite("./tmp/%s.png" % 'img1_4000_Area', img1_4000_Area)
    img1_6000_Area = cv2.resize(img1, (6000, 6000), interpolation=cv2.INTER_AREA)
    cv2.imwrite("./tmp/%s.png" % 'img1_6000_Area', img1_6000_Area)
    img1_2048_Area = cv2.resize(img1, (2048, 2048), interpolation=cv2.INTER_AREA)
    cv2.imwrite("./tmp/%s.png" % 'img1_2048_Area', img1_2048_Area)
    img1_256_Area = cv2.resize(img1, (256, 256), interpolation=cv2.INTER_AREA)
    cv2.imwrite("./tmp/%s.png" % 'img1_256_Area', img1_256_Area)

def main():
    os.chdir('./dataset_utils')
    datasets_folder = './01-2'
    pivot_scale = 2


def fp_integrate_scale(datasets_folder, pivot_scale = 2):
    fp_paths = glob("%s/trainA/*.png" % datasets_folder)
    if not os.path.exists("%s/trainA_rescaled" % (datasets_folder)):
        os.makedirs("%s/trainA_rescaled" % (datasets_folder))
    if not os.path.exists("%s/trainB_rescaled" % (datasets_folder)):
        os.makedirs("%s/trainB_rescaled" % (datasets_folder))

    f = open("%s/Info_ScaleTransform_train.csv" % datasets_folder, 'w', encoding='utf-8')
    wr = csv.writer(f)

    ii, iii = 1, len(fp_paths)
    for fp_path in fp_paths:
        print("[%d/%d] %s" % (ii, iii, fp_path))
        ii += 1

        fp = cv2.imread(fp_path, cv2.IMREAD_GRAYSCALE)
        n, m = fp.shape
        mode_gap = search_mode_gap(fp)
        ratio = pivot_scale / mode_gap
        filename = os.path.basename(fp_path)

        fp_rescaled = cv2.resize(fp, (int(m * ratio), int(n * ratio)), interpolation=cv2.INTER_AREA)
        label = cv2.imread("%s/trainB/%s" % (datasets_folder, filename))
        label_rescaled = cv2.resize(label, (int(m * ratio), int(n * ratio)), interpolation=cv2.INTER_NEAREST)

        cv2.imwrite("%s/trainA_rescaled/%s" % (datasets_folder, filename), fp_rescaled)
        cv2.imwrite("%s/trainB_rescaled/%s" % (datasets_folder, filename), label_rescaled)
        wr.writerow([filename, mode_gap, n, m, ratio, int(n * ratio), int(m * ratio)])
    f.close()


def search_mode_gap(src):
    n, m = src.shape
    src_centered = src[int(n * 0.25):int(n * 0.75), int(m * 0.25):int(m * 0.75)]
    bi_src = (src_centered < 250) * 1.0

    def gap_counts(sps_src, gaps):
        for it in range(len(sps_src.indptr) - 1):
            for jt in range(sps_src.indptr[it], sps_src.indptr[it + 1] - 1):
                gg = sps_src.indices[jt + 1] - sps_src.indices[jt] - 1
                if gg > 0:
                    gaps.append(gg)
        return gaps

    gaps = []
    tmp = sps.csr_matrix(bi_src)
    gap_counts(tmp, gaps)
    tmp = sps.csc_matrix(bi_src)
    gap_counts(tmp, gaps)

    g_counts = Counter(gaps)
    mode_gap1 = max(g_counts.keys(), key=(lambda k: g_counts[k]))
    mode_gap_count1 = g_counts[mode_gap1]
    del g_counts[mode_gap1]
    mode_gap2 = max(g_counts.keys(), key=(lambda k: g_counts[k]))
    mode_gap_count2 = g_counts[mode_gap2]

    mode_gap = max(mode_gap1, mode_gap2) if mode_gap_count1 * 0.8 < mode_gap_count2 else mode_gap1

    return mode_gap