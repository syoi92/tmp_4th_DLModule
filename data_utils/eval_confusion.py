from glob import glob
import numpy as np
import pandas as pd
import argparse
import cv2, os

np.set_printoptions(suppress=True)
parser = argparse.ArgumentParser(description='')
parser.add_argument('--pred_path', dest='pred_path', default='./eval_sample/020', help='prediction path')
parser.add_argument('--gt_path', dest='gt_path', default='./eval_sample/testB', help='groundtruth path')
parser.add_argument('--output_dir', dest='output_dir', help='output dir')

args = parser.parse_args()

# gt_path = './eval_sample/testB/13-2.png'
# pred_path = './eval_sample/020/13-2.png'
def main():
    pred_path, gt_path = args.pred_path, args.gt_path

    is_pred_dir = True if os.path.isdir(pred_path) else False
    is_pred_file = True if os.path.isfile(pred_path) else False
    is_gt_dir = True if os.path.isdir(gt_path) else False
    is_gt_file = True if os.path.isfile(gt_path) else False

    if not ((is_pred_dir == is_gt_dir) and (is_pred_file == is_gt_file)):
        raise NotImplementedError
    elif not is_pred_dir and not is_pred_file:
        raise NotImplementedError

    is_batch = True if is_pred_dir else False
    if args.output_dir == None: 
        output_dir = pred_path
    else:
        output_dir = args.output_dir
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

    columns = []
    for s in ["precision", "recall"]:
        for i in range(6):
            columns.append("%s%s" % (s,str(i)))

    if not is_batch:
        cm = cal_confusion_matrix(pred_path, gt_path)
        precision, recall = cal_performance(cm)
        df = pd.DataFrame([np.concatenate((precision, recall))],
                    index = [os.path.basename(pred_path)],
                    columns = columns)

        if args.output_dir == None:
            df.to_csv(os.path.splitext(pred_path) + '_eval.csv')
        else: 
            df.to_csv(os.path.join(output_dir, os.path.splitext(os.path.basename(pred_path))[0]+'_eval.csv'))
        print(pred_path)
        print(df.mean())
    
    else: 
        preds = glob(os.path.join(pred_path, '*.png'))
        df = pd.DataFrame(columns = columns)

        for pred in preds:
            print(f"evaluating.... - {pred}")
            filename = os.path.basename(pred)
            gt = os.path.join(gt_path, filename)

            cm = cal_confusion_matrix(pred, gt)
            precision, recall = cal_performance(cm)
            precision[precision < 0.00001], recall[recall < 0.00001] = None, None
            _df = pd.DataFrame([np.concatenate((precision, recall))],
                    index = [os.path.basename(pred)],
                    columns = columns)
            df = df.append(_df)

        df.to_csv(os.path.join(output_dir, os.path.split(pred_path)[-1]+'_eval.csv'))
        print(pred_path)
        print(df.mean())


def cal_confusion_matrix(pred_path, gt_path):
    pred = cv2.imread(pred_path)[:,:,[2,1,0]]
    gt = cv2.imread(gt_path)[:,:,[2,1,0]]
    if not pred.shape == gt.shape:
        pred = cv2.resize(pred, gt.shape[1::-1], interpolation=cv2.INTER_NEAREST)
    
    gt_ = target_rgb2label(gt)
    pred_ = target_rgb2label(pred)

    nc = 6 # num of class
    confusion_matrix = np.zeros([nc, nc])
    for it in range(nc):  # gt
        for jt in range(nc):  # pred
            confusion_matrix[it, jt] = np.sum(np.logical_and(gt_ == it, pred_ == jt))
    return confusion_matrix


def cal_performance(confusion_matrix):
    precision = np.diag(confusion_matrix) / (np.sum(confusion_matrix, axis=0) + 10**-10)
    recall = np.diag(confusion_matrix) / (np.sum(confusion_matrix, axis=1) + 10**-10)
    return precision, recall


def target_rgb2label(target_img):
    color_map = np.array([[0, 0, 255], [255, 232, 131], [255, 0, 0],
                        [0, 255, 0], [131, 89, 161], [175, 89, 62]])
    width, height, _ = target_img.shape
    target_label = np.zeros([width, height], dtype=np.uint8)
    for it in range(1, len(color_map)):
        rr, cc = np.where(np.all(target_img == color_map[it], axis=-1))
        target_label[rr, cc] = it
    return target_label


if __name__ == '__main__':
    main()

# pred_path = './00/testA'
# gt_path='./00/testB'
# output_dir = './00/test'
# preds = glob(os.path.join(pred_path, '*.png'))
# for pred in preds:
#     filename = os.path.basename(pred)
#     gt = os.path.join(gt_path, filename)

#     A = cv2.imread(pred)#[:,:,[2,1,0]]
#     B = cv2.imread(gt)#[:,:,[2,1,0]]
#     if not A.shape == B.shape:
#         B = cv2.resize(B, (A.shape[1], A.shape[0]), interpolation=cv2.INTER_NEAREST)
#     B[B[:,:,0] == 255, :] = [255, 255,255]
#     cv2.imwrite(os.path.join(output_dir, filename), cv2.addWeighted(A, 0.5, B, 0.5, 0))