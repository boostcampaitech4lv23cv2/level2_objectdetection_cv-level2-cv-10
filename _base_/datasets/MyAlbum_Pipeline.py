# dataset settings
train_dataset_type = 'CocoDataset'# 'MultiImageMixDataset'
test_dataset_type = 'CocoDataset'
data_root = '../../dataset/'

# cutmix, mixup, cutout 구현
albu_train_transforms=[
    # random rotate
    dict(type='RandomRotate90',p=0.5),
    dict(type='HorizontalFlip',p=0.5),

    # random color 
    dict(type='CLAHE',p=0.5),
    dict(type='HueSaturationValue', hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=10, p=0.5),
    dict(
        type='OneOf',
        transforms=[
            dict(type='FancyPCA', p=0.6),
            dict(
                type='RandomBrightnessContrast',
                brightness_limit=[0.1, 0.3],
                contrast_limit=[0.1, 0.3],
                p=0.2),
        ],
        p=0.3),

    # random Blur #
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', p=1.0),
            dict(type='GaussianBlur', p=1.0),
            dict(type='MedianBlur', blur_limit=5, p=1.0),
            dict(type='MotionBlur', p=1.0)
        ],
        p=0.1),
    
    
    # dict(type='RandomGridShuffle', grid=(2, 2),p=0.2),
    
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.0625,
        scale_limit=0.0,
        rotate_limit=10, 
        interpolation=1,
        p=0.3),
    
    # random Noise
    dict(
        type='OneOf',
        transforms=[
            dict(type='GaussNoise', p=1.0),
            dict(type='ISONoise', p=1.0),
            dict(type='MultiplicativeNoise', p=1.0)
        ],
        p=0.2),
]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

multi_scale = [(512, 512), (1024, 1024)] # 

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
    # dict(type='Resize', img_scale=multi_scale, multiscale_mode='value', keep_ratio=True),
    dict(type='Pad', size_divisor=32),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_masks': 'masks',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True
    ),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect', 
        keys=['img', 'gt_bboxes', 'gt_labels'],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'img_norm_cfg',
                   'pad_shape', 'scale_factor'))
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        # img_scale = multi_scale,
        img_scale = (1024, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
            # dict(type='Resize', img_scale=multi_scale, multiscale_mode='value', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        type=train_dataset_type,
        ann_file=data_root + 'fold_0_train.json',
        img_prefix=data_root,
        classes=classes,
        pipeline=train_pipeline),
    val=dict(
        type=test_dataset_type,
        ann_file=data_root + 'fold_0_val.json',
        img_prefix=data_root,
        classes=classes,
        pipeline=test_pipeline),
    test=dict(
        type=test_dataset_type,
        ann_file=data_root + 'test.json',
        img_prefix=data_root,
        classes=classes,
        pipeline=test_pipeline))

evaluation = dict(interval=1, metric='bbox', save_best='bbox_mAP')
