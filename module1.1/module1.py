import argparse
import tensorflow as tf
import cv2
from collections import Counter
import scipy.sparse as sps
from glob import glob
import os
import time
import numpy as np
import math

parser = argparse.ArgumentParser(description='')
parser.add_argument('--src_path', dest='src_path', default='./sample.png', help='floorplan path')
parser.add_argument('--model_dir', dest='model_dir', default='./model', help='directory of trained models')
parser.add_argument('--output_dir', dest='output_dir', default='./output', help='output directory')
#
parser.add_argument('--scale_factor', dest='scale_factor', type=float, default=0,
                    help='No scaleAdjust: -1, Auto-transform: 0 (default), Manual value: positive floating number')
args = parser.parse_args()


def main():
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # path size used for DL; (img_size x img_size) and scale for model config.
    model_folder = os.listdir(args.model_dir)
    model_config = model_folder[0].split('_')
    img_size, scale = int(model_config[3]), float(model_config[6])

    # fixed parameters
    pivot_gap = 2 * scale # 2 is default gap for trained DL model
    overlapping = 0.1
    scale_ratio = 1
    use_scaleAdjust = True if not args.scale_factor < 0 else False
    use_gridTest = True

    src = cv2.imread(args.src_path, cv2.IMREAD_GRAYSCALE)
    # scaleAdjust
    if use_scaleAdjust:
        if not args.scale_factor > 0:  # scale auto-transform
            cur_mode_gap = search_mode_gap(src)  # searching the mode gap of white pixels from the src floorplan
        else:
            cur_mode_gap = args.scale_factor  # manual value

        scale_ratio = pivot_gap / cur_mode_gap
        if not scale_ratio == 1:
            src = cv2.resize(src, (0, 0), fx=scale_ratio, fy=scale_ratio, interpolation=cv2.INTER_AREA)

    # gridTest
    if not use_gridTest:
        n, m = src.shape
        if abs(n - m) > 0.05 * max(n, m):  # padding to the square shape only when w-h ratio is far from 1
            if n > m:
                pad = 255 * np.ones((n, int((n - m) / 2)))
                src = np.hstack((pad, src, pad))
            else:
                pad = 255 * np.ones((int((m - n) / 2), m))
                src = np.vstack((pad, src, pad))
        src = cv2.resize(src, (img_size, img_size), interpolation=cv2.INTER_AREA)  # resizing to DL's input size

    # Deep Learning
    model_meta = glob(os.path.join(args.model_dir, '*/*.meta'))
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        model_saver = tf.train.import_meta_graph(model_meta[0])
        model_saver.restore(sess, os.path.splitext(model_meta[0])[0])
        print("Model Restored- %s\ntime: 0.00s" % os.path.splitext(model_meta[0])[0])
        start_time = time.time()
        if not len(model_meta) == 1:
            print("One model is not specified in %s" % args.model_dir)

        g = tf.get_default_graph()
        test_img = g.get_tensor_by_name("test_img:0")
        test_logit = tf.get_collection('test_logit')[0]

        if not use_gridTest:
            src = [load_test_data(src)]
            src = np.array(src).astype(np.float32)
            logit = sess.run(test_logit, feed_dict={test_img: src})
            save_logit(logit, os.path.join(args.output_dir, os.path.basename(args.src_path)))
            print("Prediction stored at %s" % args.output_dir)
            print("Time: %5.2fs" % (time.time()-start_time))

        else:
            # when using girdTest, just padding when src size is smaller than (img_size, img_size)
            pad_h, pad_w = -1, -1
            n, m = src.shape
            if n < img_size:
                pad_h = img_size - n
                src = np.vstack((src, 255 * np.ones((pad_h, m))))
            n, m = src.shape
            if m < img_size:
                pad_w = img_size - m
                src = np.hstack((src, 255 * np.ones((n, pad_w))))

            height, width = src.shape
            print("img size to be inputted to DL: (%sx%s) | scale_factor: %s" % (height, width, (pivot_gap / scale_ratio)))
            logit = np.zeros([1, height, width, 6])
            w_num = math.ceil(width / img_size / (1 - overlapping))
            h_num = math.ceil(height / img_size / (1 - overlapping))
            print("Patch division- horizontal: %s, vertical: %s" % (w_num, h_num))

            for (h, w) in [(h_, w_) for h_ in range(h_num) for w_ in range(w_num)]:
                h1 = int(h / (h_num - 1) * (height - img_size)) if not h_num == 1 else 0
                w1 = int(w / (w_num - 1) * (width - img_size)) if not w_num == 1 else 0
                cur_img = src[h1:h1 + img_size, w1:w1 + img_size]
                cur_img = [load_test_data(cur_img)]
                cur_img = np.array(cur_img).astype(np.float32)
                cur_logit = sess.run(test_logit, feed_dict={test_img: cur_img})
                logit[0, h1:h1 + img_size, w1:w1 + img_size] += cur_logit[0]
                print(("Path processing... [%3d/%3d]" % (h*w_num + w + 1, w_num * h_num)))


            if pad_h > 0:
                logit = logit[:, :-pad_h, :, :]
            if pad_w > 0:
                logit = logit[:, :, :-pad_w, :]
            save_logit(logit, os.path.join(args.output_dir, os.path.basename(args.src_path)), scale_ratio)
            print("Prediction stored at %s" % args.output_dir)
            print("Time: %5.2fs" % (time.time() - start_time))



def search_mode_gap(src):
    n, m = src.shape
    src_centercropped = src[int(n * 0.25):int(n * 0.75), int(m * 0.25):int(m * 0.75)]
    bi_src = (src_centercropped < 250) * 1.0

    def gap_counts(sps_src, pixel_gaps):
        for it in range(len(sps_src.indptr) - 1):
            for jt in range(sps_src.indptr[it], sps_src.indptr[it + 1] - 1):
                gg = sps_src.indices[jt + 1] - sps_src.indices[jt] - 1
                if gg > 0:
                    pixel_gaps.append(gg)
        return pixel_gaps

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


def load_test_data(src):
    n, m = src.shape
    test_img = src / 127.5 - 1
    test_img = np.array(test_img).reshape([n, m, 1])
    return test_img


def save_logit(logits, image_path, scale_ratio):
    def target_label2rgb(target_np, label_colors):
        width, height = target_np.shape
        target_img = np.zeros([width, height, 3], dtype=np.uint8)
        target_img[:] = label_colors[0]  # background
        for it in range(5):
            rr, cc = np.where(target_np == it + 1)
            target_img[rr, cc, :] = label_colors[it + 1]
        return target_img

    ColorMap = np.array([[0, 0, 255], [255, 232, 131], [255, 0, 0],
                         [0, 255, 0], [131, 89, 161], [175, 89, 62]])
    logit = logits[0]
    pred_label = np.argmax(logit, axis=-1)
    img = target_label2rgb(pred_label, ColorMap)
    img = cv2.resize(img, (0, 0), fx=1/scale_ratio, fy=1/scale_ratio, interpolation=cv2.INTER_NEAREST)

    return cv2.imwrite(image_path, img[:, :, ::-1])


if __name__ == '__main__':
    main()
