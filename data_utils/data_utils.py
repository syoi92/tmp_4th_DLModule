from __future__ import division
import os
import json
import copy
from glob import glob
# import matplotlib.pylab as plt
import numpy as np
from skimage import io, draw
import cv2


def main():
    dataset_dir = './original (vgg_89)'
    json_label_path = glob("%s/*.json" % dataset_dir)[0]
    #  validation
    vali_annotation(dataset_dir, json_label_path)
    # boundary cropping
    bndry_cropping(dataset_dir, json_label_path)


def bndry_cropping(dataset_dir, json_label_path):
    root_dir = os.path.split(dataset_dir)[0]
    output_dir_name = os.path.basename(dataset_dir) + '_bndry'
    output_dir = os.path.join(root_dir, output_dir_name)

    A_dir = os.path.join(output_dir, 'trainA')
    B_dir = os.path.join(output_dir, 'trainB')

    if not os.path.exists(dataset_dir):
        print('There is no dataset dir')
    if not os.path.exists(A_dir):
        os.makedirs(A_dir)
    if not os.path.exists(B_dir):
        os.makedirs(B_dir)

    label_num = 8
    label_colors = [[0, 0, 255], [255, 232, 131], [255, 0, 0],
                    [0, 255, 0], [131, 89, 161], [175, 89, 62]]
    # 0=back ground, 1=wall, 2=window, 3=door, 4=lift, 5=stair, 6=del, 7=bndry
    # Blue, Yellow, Red, Green, Violet, Brown - color for label 0~5

    fp_paths = glob("%s/*.png" % dataset_dir)
    annotations = json.load(open(json_label_path))
    annotations = list(annotations.values())
    annotations = [a for a in annotations if a['regions']]

    for fp_path in fp_paths:
        filename = os.path.basename(fp_path)
        print('processing... {}'.format(filename))
        annotation = [a for a in annotations if a['filename'] == filename]
        cur_img = io.imread(fp_path)
        height, width = cur_img.shape[:2]

        if len(annotation) == 0:  # When there's no anntation
            tmp, tmp[:, :, 2] = np.zeros((height, width, 3), dtype=np.uint8), 255
            io.imsave(os.path.join(A_dir, filename), np.zeros([height, width, 3], dtype=np.uint8))
            io.imsave(os.path.join(B_dir, filename), tmp)
            print('There is no annotation for {}'.format(filename))
            continue

        polygons = [[] for _ in range(label_num - 1)]
        for r in annotation[0]['regions'].values():  # first json having the same filename
            polygon = r['shape_attributes']
            name = r['region_attributes']

            if name['name'] == "wall":  # Yellow, class_id = 1
                polygons[0].append(polygon)
            elif name['name'] == "window1":  # Green, class_id = 2
                polygons[1].append(polygon)
            elif name['name'] == "door":  # Red, class_id = 3
                polygons[2].append(polygon)
            elif name['name'] == "lift":  # Violet, class_id = 4
                polygons[3].append(polygon)
            elif name['name'] == "stair":  # Brown, class_id = 5
                polygons[4].append(polygon)
            elif name['name'] == "del":  # class_id = 6
                polygons[5].append(polygon)
            elif name['name'] == "bndry":  # class_id = 7
                polygons[6].append(polygon)

        # image after applying 'del' class
        cur_img_del = copy.deepcopy(cur_img)
        for i, p in enumerate(polygons[5]):  # 5: del-class
            if p['name'] == 'rect':
                min_y, max_y = p['x'], p['x'] + p['width']
                min_x, max_x = p['y'], p['y'] + p['height']
                rr, cc = draw.rectangle((min_x, min_y), (max_x, max_y))  # delete space
                cur_img_del[rr, cc, :] = 255
            elif p['name'] == 'polygon':
                rr, cc = draw.polygon(p['all_points_y'], p['all_points_x'])  # delete space
                cur_img_del[rr, cc, :] = 255

        # drawing annotation_map
        anno_map = np.zeros((height, width, 3), dtype=np.uint8)
        anno_map[:, :, 2] = 255  # blue background: label_colors[0]
        for j in range(5):  # j : class1~5
            for _, p in enumerate(polygons[j]):
                if p['name'] == 'rect':
                    min_y, max_y = p['x'], p['x'] + p['width']
                    min_x, max_x = p['y'], p['y'] + p['height']
                    rr, cc = draw.rectangle((min_x, min_y), (max_x, max_y))  # delete space
                    anno_map[rr, cc, :] = label_colors[j + 1][:]
                elif p['name'] == 'polygon':
                    rr, cc = draw.polygon(p['all_points_y'], p['all_points_x'])
                    anno_map[rr, cc, :] = label_colors[j + 1][:]

        for _, p in enumerate(polygons[5]):  # 5: del-class
            if p['name'] == 'rect':
                min_y, max_y = p['x'], p['x'] + p['width']
                min_x, max_x = p['y'], p['y'] + p['height']
                rr, cc = draw.rectangle((min_x, min_y), (max_x, max_y))  # delete space
                anno_map[rr, cc, :] = label_colors[0][:]
            elif p['name'] == 'polygon':
                rr, cc = draw.polygon(p['all_points_y'], p['all_points_x'])  # delete space
                anno_map[rr, cc, :] = label_colors[0][:]

        #  cropping & padding
        if len(polygons[6]) is not 0:
            p = polygons[6][-1]
            min_y, max_y = p['x'], p['x'] + p['width']
            min_x, max_x = p['y'], p['y'] + p['height']

            cx, cy = int((max_x + min_x) / 2), int((max_y + min_y) / 2)
            crop_w, crop_h = max_x - min_x, max_y - min_y
            cr, cp = int(max(crop_w, crop_h) / 2), int(abs(crop_w - crop_h) / 2)

            img_bndry = np.ones((height + 2 * cp, width + 2 * cp, 3), dtype=np.uint8) * 255
            img_bndry[cp:cp + height, cp:cp + width] = cur_img_del
            img_bndry = img_bndry[cp + cx - cr:cp + cx + cr, cp + cy - cr:cp + cy + cr]
            io.imsave(os.path.join(A_dir, filename), img_bndry)

            anno_bndry = np.zeros((height + 2 * cp, width + 2 * cp, 3), dtype=np.uint8)
            anno_bndry[:, :, 2] = 255
            anno_bndry[cp:cp + height, cp:cp + width] = anno_map
            anno_bndry = anno_bndry[cp + cx - cr:cp + cx + cr, cp + cy - cr:cp + cy + cr]
            io.imsave(os.path.join(B_dir, filename), anno_bndry)
            print('saving an cropping image')

        io.imsave(os.path.join(A_dir, filename), cur_img_del)
        io.imsave(os.path.join(B_dir, filename), anno_map)
        print('saving an image. no cropping')


def vali_annotation(dataset_dir, json_label_path):
    root_dir = os.path.split(dataset_dir)[0]
    output_dir_name = os.path.basename(dataset_dir) + '_check'
    output_dir = os.path.join(root_dir, output_dir_name)

    del_dir = os.path.join(output_dir, 'delete')
    anno_dir = os.path.join(output_dir, 'annotation')

    if not os.path.exists(dataset_dir):
        print('There is no dataset dir')
    if not os.path.exists(del_dir):
        os.makedirs(del_dir)
    if not os.path.exists(anno_dir):
        os.makedirs(anno_dir)

    label_num = 8
    label_colors = [[0, 0, 255], [255, 232, 131], [255, 0, 0],
                    [0, 255, 0], [131, 89, 161], [175, 89, 62]]
    # 0=back ground, 1=wall, 2=window, 3=door, 4=lift, 5=stair, 6=del, 7=bndry
    # Blue, Yellow, Red, Green, Violet, Brown - color for label 0~5

    fp_paths = glob("%s/*.png" % dataset_dir)
    annotations = json.load(open(json_label_path))
    annotations = list(annotations.values())
    annotations = [a for a in annotations if a['regions']]

    for fp_path in fp_paths:
        filename = os.path.basename(fp_path)
        print('processing... {}'.format(filename))
        annotation = [a for a in annotations if a['filename'] == filename]
        cur_img = io.imread(fp_path)
        height, width = cur_img.shape[:2]

        if len(annotation) == 0:  # When there's no anntation
            tmp = np.zeros((height, width, 3), dtype=np.uint8); tmp[:, :, 2] = 255
            io.imsave(os.path.join(del_dir, filename), np.zeros([height, width, 3], dtype=np.uint8))
            io.imsave(os.path.join(anno_dir, filename), tmp)
            print('There is no annotation for {}'.format(filename))
            continue

        polygons = [[] for _ in range(label_num - 1)]
        for r in annotation[0]['regions'].values():  # first json having the same filename
            polygon = r['shape_attributes']
            name = r['region_attributes']

            if name['name'] == "wall":  # Yellow, class_id = 1
                polygons[0].append(polygon)
            elif name['name'] == "window1":  # Green, class_id = 2
                polygons[1].append(polygon)
            elif name['name'] == "door":  # Red, class_id = 3
                polygons[2].append(polygon)
            elif name['name'] == "lift":  # Violet, class_id = 4
                polygons[3].append(polygon)
            elif name['name'] == "stair":  # Brown, class_id = 5
                polygons[4].append(polygon)
            elif name['name'] == "del":  # class_id = 6
                polygons[5].append(polygon)
            elif name['name'] == "bndry":  # class_id = 7
                polygons[6].append(polygon)

        # image after applying 'del' class
        cur_img_del = copy.deepcopy(cur_img)
        for i, p in enumerate(polygons[5]):  # 5: del-class
            if p['name'] == 'rect':
                min_y, max_y = p['x'], p['x'] + p['width']
                min_x, max_x = p['y'], p['y'] + p['height']
                rr, cc = draw.rectangle((min_x, min_y), (max_x, max_y))  # delete space
                cur_img_del[rr, cc, :] = 255
            elif p['name'] == 'polygon':
                rr, cc = draw.polygon(p['all_points_y'], p['all_points_x'])  # delete space
                cur_img_del[rr, cc, :] = 255
        for i, p in enumerate(polygons[6]):  # generating red boundary
            min_y, max_y = p['x'], p['x'] + p['width']
            min_x, max_x = p['y'], p['y'] + p['height']
            tkn = int(0.002 * max(width, height))
            cur_img_del[min_x - tkn:max_x + tkn, min_y - tkn:min_y + tkn, :] = [255, 0, 0]
            cur_img_del[min_x - tkn:max_x + tkn, max_y - tkn:max_y + tkn, :] = [255, 0, 0]
            cur_img_del[min_x - tkn:min_x + tkn, min_y - tkn:max_y + tkn, :] = [255, 0, 0]
            cur_img_del[max_x - tkn:max_x + tkn, min_y - tkn:max_y + tkn, :] = [255, 0, 0]

        io.imsave(os.path.join(del_dir, filename), cur_img_del)
        print('saving an image with del_class')

        # drawing annotation_map
        anno_map = np.zeros((height, width, 3), dtype=np.uint8)
        anno_map[:, :, 2] = 255  # blue background: label_colors[0]
        for j in range(5):  # j : class1~5
            for _, p in enumerate(polygons[j]):
                if p['name'] == 'rect':
                    min_y, max_y = p['x'], p['x'] + p['width']
                    min_x, max_x = p['y'], p['y'] + p['height']
                    rr, cc = draw.rectangle((min_x, min_y), (max_x, max_y))  # delete space
                    anno_map[rr, cc, :] = label_colors[j + 1][:]
                elif p['name'] == 'polygon':
                    rr, cc = draw.polygon(p['all_points_y'], p['all_points_x'])
                    anno_map[rr, cc, :] = label_colors[j + 1][:]

        for _, p in enumerate(polygons[5]):  # 5: del-class
            if p['name'] == 'rect':
                min_y, max_y = p['x'], p['x'] + p['width']
                min_x, max_x = p['y'], p['y'] + p['height']
                rr, cc = draw.rectangle((min_x, min_y), (max_x, max_y))  # delete space
                anno_map[rr, cc, :] = label_colors[0][:]
            elif p['name'] == 'polygon':
                rr, cc = draw.polygon(p['all_points_y'], p['all_points_x'])  # delete space
                anno_map[rr, cc, :] = label_colors[0][:]

        for i, p in enumerate(polygons[6]):  # generating red boundary
            min_y, max_y = p['x'], p['x'] + p['width']
            min_x, max_x = p['y'], p['y'] + p['height']
            tkn = int(0.002 * max(width, height))
            anno_map[min_x - tkn:max_x + tkn, min_y - tkn:min_y + tkn, :] = [255, 0, 0]
            anno_map[min_x - tkn:max_x + tkn, max_y - tkn:max_y + tkn, :] = [255, 0, 0]
            anno_map[min_x - tkn:min_x + tkn, min_y - tkn:max_y + tkn, :] = [255, 0, 0]
            anno_map[max_x - tkn:max_x + tkn, min_y - tkn:max_y + tkn, :] = [255, 0, 0]

        io.imsave(os.path.join(anno_dir, filename), anno_map)
        print('saving an image of visualised annotations')

    generate_inspection_html(dataset_dir, del_dir, anno_dir)


def generate_inspection_html(dataset_dir, del_dir, anno_dir):
    sample_paths = glob("%s/*.png" % dataset_dir)
    html_name = os.path.basename(dataset_dir) + '_check'

    index = open("%s.html" % html_name, "w")
    index.write("<html><body><table style=\"font-size: 100\"><tr>")
    index.write("<th>Bldg num<br>(dong)</th><th>Original</th><th>Deleted<br>(dong)</th><th>Annotations</th></tr>")

    it = 0
    for sample_path in sample_paths:
        cur_sam = os.path.basename(sample_path)
        cur_bldg = os.path.splitext(cur_sam)[0]
        index.write("<td align=\"center\">%s</td>" % cur_bldg)
        index.write("<td><img src='%s'></td>" % sample_path)
        index.write("<td><img src='%s'></td>" % os.path.join(del_dir, cur_sam))
        index.write("<td><img src='%s'></td>" % os.path.join(anno_dir, cur_sam))
        index.write("</tr>")
        it += 1
    index.close()


def dataset_patch(dataset_dir, crop_size = 512, overlapping = 0.5):
    root_dir = os.path.split(dataset_dir)[0]
    output_dir_name = os.path.basename(dataset_dir) + '_patch'
    output_dir = os.path.join(root_dir, output_dir_name)

    input_A_dir = os.path.join(dataset_dir, 'trainA')
    input_B_dir = os.path.join(dataset_dir, 'trainB')
    output_A_dir = os.path.join(output_dir, 'trainA')
    output_B_dir = os.path.join(output_dir, 'trainB')

    if not os.path.exists(output_A_dir):
        os.makedirs(output_A_dir)
    if not os.path.exists(output_B_dir):
        os.makedirs(output_B_dir)

    input_A_paths = glob("%s/*.png" % input_A_dir)
    input_B_paths = glob("%s/*.png" % input_B_dir)
    dataset_paths = list(zip(input_A_paths, input_B_paths))

    for dataset_path in dataset_paths:
        filename= os.path.splitext(os.path.basename(dataset_path[0]))[0]
        crop_name, crop_img, crop_label, crop_white = fp_cropping(dataset_path, crop_size, overlapping)
        for idx in range(len(crop_name)):
            w, h = crop_name[idx]
            if not crop_white[idx]:
                patch_name = "%s_%s_%s" % (filename, w, h) + '.png'
                cv2.imwrite(os.path.join(output_A_dir, patch_name), crop_img[idx])
                cv2.imwrite(os.path.join(output_B_dir, patch_name), crop_label[idx])



def fp_cropping(dataset_path, crop_size, overlapping = 0.5, th_white = 0.95):    # overlapping: [0, 1)
    img = cv2.imread(dataset_path[0])
    label = cv2.imread(dataset_path[1])
    height, width = img.shape[0:2]
    crop_name, crop_img, crop_label, crop_white = [], [], [], []

    w_num = int(width / crop_size / (1-overlapping))
    h_num = int(height / crop_size / (1-overlapping))
    for (h, w) in [(h_, w_) for h_ in range(h_num) for w_ in range(w_num)]:
        h1 = int(h / (h_num - 1) * (height - crop_size))
        w1 = int(w / (w_num - 1) * (width - crop_size))
        cur_img = img[h1:h1 + crop_size, w1:w1 + crop_size]
        cur_label = label[h1:h1 + crop_size, w1:w1 + crop_size]
        is_white = True if (np.sum(np.sum(cur_img, axis=-1) > 750) / crop_size ** 2) > th_white else False

        crop_name.append((w, h))
        crop_img.append(cur_img)
        crop_label.append(cur_label)
        crop_white.append(is_white)

    return crop_name, crop_img, crop_label, crop_white


if __name__ == '__main__':
    main()