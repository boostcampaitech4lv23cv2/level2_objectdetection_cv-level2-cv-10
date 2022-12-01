import numpy as np
import argparse
from pycocotools.coco import COCO
import pandas as pd
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import os

import mmcv
from mmcv import Config
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.apis import single_gpu_test
from mmcv.runner import load_checkpoint
import os
from mmcv.parallel import MMDataParallel

def box_iou_calc(boxes1, boxes2):
    # <https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py>
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        boxes1 (Array[N, 4])
        boxes2 (Array[M, 4])
    Returns:
        iou (Array[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    This implementation is taken from the above link and changed so that it only uses numpy..
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(boxes1.T)
    area2 = box_area(boxes2.T)

    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    inter = np.prod(np.clip(rb - lt, a_min=0, a_max=None), 2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

class ConfusionMatrix:
    def __init__(self, num_classes: int, CONF_THRESHOLD=0.3, IOU_THRESHOLD=0.5):
        self.matrix = np.zeros((num_classes + 1, num_classes + 1))
        self.num_classes = num_classes
        self.CONF_THRESHOLD = CONF_THRESHOLD
        self.IOU_THRESHOLD = IOU_THRESHOLD

    def plot(self, file_name='./', names=["General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]):
        try:
            import seaborn as sn

            array = self.matrix / (self.matrix.sum(0).reshape(1, self.num_classes + 1) + 1E-6)  # normalize
            array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

            fig = plt.figure(figsize=(12, 9), tight_layout=True)
            sn.set(font_scale=1.0 if self.num_classes < 50 else 0.8)  # for label size
            labels = (0 < len(names) < 99) and len(names) == self.num_classes  # apply names to ticklabels
            sn.heatmap(array, annot=self.num_classes < 30, annot_kws={"size": 8}, cmap='Blues', fmt='.2f', square=True,
                       xticklabels=names + ['background FP'] if labels else "auto",
                       yticklabels=names + ['background FN'] if labels else "auto").set_facecolor((1, 1, 1))
            fig.axes[0].set_xlabel('True')
            fig.axes[0].set_ylabel('Predicted')
            fig.savefig(Path('./result_analysis') / file_name, dpi=250)
        except Exception as e:
            print(e)
            pass

    def process_batch(self, detections, labels: np.ndarray):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates confusion matrix accordingly
        """
        gt_classes = labels[:, 0].astype(np.int16)

        try:
            detections = detections[detections[:, 4] > self.CONF_THRESHOLD]
        except IndexError or TypeError:
            # detections are empty, end of process
            for i, label in enumerate(labels):
                gt_class = gt_classes[i]
                self.matrix[self.num_classes, gt_class] += 1
            return

        detection_classes = detections[:, 5].astype(np.int16)

        all_ious = box_iou_calc(labels[:, 1:], detections[:, :4])
        want_idx = np.where(all_ious > self.IOU_THRESHOLD)

        all_matches = [[want_idx[0][i], want_idx[1][i], all_ious[want_idx[0][i], want_idx[1][i]]]
                       for i in range(want_idx[0].shape[0])]

        all_matches = np.array(all_matches)
        if all_matches.shape[0] > 0:  # if there is match
            all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]

            all_matches = all_matches[np.unique(all_matches[:, 1], return_index=True)[1]]

            all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]

            all_matches = all_matches[np.unique(all_matches[:, 0], return_index=True)[1]]

        for i, label in enumerate(labels):
            gt_class = gt_classes[i]
            if all_matches.shape[0] > 0 and all_matches[all_matches[:, 0] == i].shape[0] == 1:
                detection_class = detection_classes[int(all_matches[all_matches[:, 0] == i, 1][0])]
                self.matrix[detection_class, gt_class] += 1
            else:
                self.matrix[self.num_classes, gt_class] += 1

        for i, detection in enumerate(detections):
            if not all_matches.shape[0] or ( all_matches.shape[0] and all_matches[all_matches[:, 1] == i].shape[0] == 0 ):
                detection_class = detection_classes[i]
                self.matrix[detection_class, self.num_classes] += 1

    def return_matrix(self):
        return self.matrix

    def print_matrix(self):
        for i in range(self.num_classes + 1):
            print(' '.join(map(str, self.matrix[i])))

def main(args):

    conf_mat = ConfusionMatrix(num_classes = 10, CONF_THRESHOLD = 0.01, IOU_THRESHOLD = 0.5)
    gt_path = '/opt/ml/dataset/' + args.gt_json
    with open(gt_path, 'r') as outfile:
        test_anno = (json.load(outfile))
    model_config_path = os.path.join('/opt/ml/baseline/UniverseNet/configs/_trash_', args.cfg_path)
    model_checkpoint_path = os.path.join('/opt/ml/baseline/UniverseNet/work_dirs', args.chkpoint_path)
    model_file_name = '/opt/ml/baseline/UniverseNet/work_dirs/confusion_matrix.png'

    classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
            "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
    cfg = Config.fromfile(model_config_path)
    cfg.data.test.classes = classes
    cfg.data.test.ann_file = gt_path
    cfg.data.test.pipeline[1]['img_scale'] = (1024,1024) # Resize
    cfg.data.test.test_mode = True
    cfg.data.samples_per_gpu = 4
    cfg.seed= 97235
    cfg.gpu_ids = [1]
    cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
    cfg.model.train_cfg = None

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False)

    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, model_checkpoint_path, map_location = 'cpu')
    model.CLASSES = dataset.CLASSES
    model = MMDataParallel(model.cuda(), device_ids = [0])
    preds = single_gpu_test(model, data_loader, show_score_thr=0.05)

    new_pred = []
    for pred in preds:
        new_pred.append([])
        for c, bboxes in enumerate(pred):
            for bbox in bboxes:
                bbox = bbox.tolist()
                bbox.append(c)
                new_pred[-1].append(bbox)

    gt = []

    coco = COCO(gt_path)
    
    for image_id in coco.getImgIds():
        gt.append([])
        image_info = coco.loadImgs(image_id)[0]
        ann_ids = coco.getAnnIds(imgIds=image_info['id']) # 주의: 이미지ID와 AnnID는 다르다.
        anns = coco.loadAnns(ann_ids)
        
        file_name = image_info['file_name']
        
        for ann in anns:
            gt[-1].append([
                       float(ann['category_id']),
                       float(ann['bbox'][0]),
                       float(ann['bbox'][1]),
                       float(ann['bbox'][0]) + float(ann['bbox'][2]),
                       (float(ann['bbox'][1]) + float(ann['bbox'][3])),  
                       ]
                       )
    for p, g in zip(new_pred, gt):
        conf_mat.process_batch(np.array(p), np.array(g))
    conf_mat.plot(model_file_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--gt_json', type=str, default='fold_3_val.json')
    parser.add_argument('--cfg_path', type=str, default='cascade_swin_l.py')
    parser.add_argument('--chkpoint_path', type=str, default='cascade_swin_l/best_bbox_mAP_epoch_12.pth')
    # parser.add_argument('--file_name', type=str, default='/opt/ml/baseline/mmdetection/work_dirs/confusion_matrix.png',)
    args = parser.parse_args()
    main(args)