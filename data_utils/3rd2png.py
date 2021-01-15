from glob import glob
import os
import numpy as np
import cv2
import csv

import matplotlib.pyplot as plt

def main():
    os.chdir('./dataset_utils')

    output_dir = './3rd_merged'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    src_folder = './3rd/'
    src_paths = glob('{}/*door.png'.format(src_folder))
    color_map = np.array([[0, 0, 255], [255, 232, 131], [255, 0, 0],
                                  [0, 255, 0], [131, 89, 161], [175, 89, 62]])

    for src_path in src_paths:
        output = combine_output(src_path, color_map)

        cur_file = os.path.basename(src_path).replace('door', '')
        output_path= os.path.join(output_dir, cur_file)
        cv2.imwrite(output_path, output[:,:, ::-1])


def combine_output(src_path, color_map):
    dirname = os.path.dirname(src_path)
    fp_num = os.path.basename(src_path).replace('door.png','')

    wall = cv2.imread(os.path.join(dirname, '{}wall.png'.format(fp_num)), cv2.IMREAD_GRAYSCALE)
    window = cv2.imread(os.path.join(dirname, '{}window.png'.format(fp_num)), cv2.IMREAD_GRAYSCALE)
    door = cv2.imread(os.path.join(dirname, '{}door.png'.format(fp_num)), cv2.IMREAD_GRAYSCALE)

    height, width = wall.shape
    canvas = np.zeros((height, width), dtype=np.uint8)
    canvas[wall<240] = 1
    canvas[window < 240] = 2
    canvas[door < 240] = 3

    canvas = cv2.resize(canvas, (3309, 3309), interpolation=cv2.INTER_NEAREST)
    with open(os.path.join(dirname, '{}roi.csv'.format(fp_num))) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row[0] == 'lift':
                canvas[int(row[1]):int(row[3]), int(row[2]):int(row[4])].fill(4)
            if row[0] == 'stair':
                canvas[int(row[1]):int(row[3]), int(row[2]):int(row[4])].fill(5)

    output = target_label2rgb(canvas, color_map)
    return output




def target_label2rgb(target_np, label_colors):
    width, height = target_np.shape
    target_img = np.zeros([width, height, 3], dtype=np.uint8)
    target_img[:] = label_colors[0]  # background
    for it in range(len(label_colors)-1):
        rr, cc = np.where(target_np == it+1)
        target_img[rr, cc, :] = label_colors[it+1]
    return target_img