import torch
from ssd.modeling.anchors.prior_box import PriorBox
from ssd.utils import box_utils
from .transforms import *


class TrainAugmentation:
    def __init__(self, cfg):
        """
        Args:
            size: the size the of final image.
            mean: mean pixel value per channel.
        """

        self.augment = Compose([
            ConvertFromInts(),
            PhotometricDistort(),
            RandomSampleCrop_v2(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(cfg.INPUT.IMAGE_SIZE),
            SubtractMeans(cfg.INPUT.MEAN),
            lambda img, boxes=None, labels=None: (
                img / cfg.INPUT.STD, boxes, labels),
            ToTensor(),
        ])

    def __call__(self, img, boxes, labels):
        """

        Args:
            img: the output of cv.imread in RGB layout.
            boxes: boundding boxes in the form of (x1, y1, x2, y2).
            labels: labels of boxes.
        """
        return self.augment(img, boxes, labels)


class TestTransform:
    def __init__(self, cfg):
        self.transform = Compose([
            ToPercentCoords(),
            Resize(cfg.INPUT.IMAGE_SIZE),
            SubtractMeans(cfg.INPUT.MEAN),
            lambda img, boxes=None, labels=None: (
                img / cfg.INPUT.STD, boxes, labels),
            ToTensor(),
        ])

    def __call__(self, image, boxes, labels):
        return self.transform(image, boxes, labels)


class PredictionTransform:
    def __init__(self, cfg):
        self.transform = Compose([
            Resize(cfg.INPUT.IMAGE_SIZE),
            SubtractMeans(cfg.INPUT.MEAN),
            lambda img, boxes=None, labels=None: (
                img / cfg.INPUT.STD, boxes, labels),
            ToTensor()
        ])

    def __call__(self, image):
        image, _, _ = self.transform(image)
        return image


class SSDTargetTransform:
    def __init__(self, cfg):
        self.center_form_priors = PriorBox(cfg)()
        self.corner_form_priors = box_utils.center_form_to_corner_form(
            self.center_form_priors)
        self.center_variance = cfg.MODEL.CENTER_VARIANCE
        self.size_variance = cfg.MODEL.SIZE_VARIANCE
        self.iou_threshold = cfg.MODEL.THRESHOLD

    def __call__(self, gt_boxes, gt_labels):
        if type(gt_boxes) is np.ndarray:
            gt_boxes = torch.from_numpy(gt_boxes)
        if type(gt_labels) is np.ndarray:
            gt_labels = torch.from_numpy(gt_labels)
        boxes, labels = box_utils.assign_priors(gt_boxes, gt_labels,
                                                self.corner_form_priors, self.iou_threshold)
        boxes = box_utils.corner_form_to_center_form(boxes)
        locations = box_utils.convert_boxes_to_locations(
            boxes, self.center_form_priors, self.center_variance, self.size_variance)

        return locations, labels
