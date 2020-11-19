from torch.utils.data import ConcatDataset

from ssd.config.path_catlog import DatasetCatalog
from .voc import VOCDataset
from .coco import COCODataset

_DATASETS = {
    'VOC': VOCDataset,
    'COCO': COCODataset,
}


def build_dataset(cfg, transform=None, target_transform=None, is_train=True):
    dataset_list = cfg.DATASETS.TRAIN if is_train else cfg.DATASETS.TEST
    assert len(dataset_list) > 0
    datasets = []
    for split in dataset_list:
        args = dict(
            data_dir=cfg.DATASETS.DATA_DIR,
            split=split,
            class_names=cfg.MODEL.CLASSES,
            transform=transform,
            target_transform=target_transform
        )
        factory = _DATASETS[cfg.DATASETS.TYPE]

        if factory == VOCDataset:
            args['keep_difficult'] = not is_train
        elif factory == COCODataset:
            args['remove_empty'] = is_train
        dataset = factory(**args)
        datasets.append(dataset)
    # for testing, return a list of datasets
    if not is_train:
        return datasets
    dataset = datasets[0]
    if len(datasets) > 1:
        dataset = ConcatDataset(datasets)

    return [dataset]
