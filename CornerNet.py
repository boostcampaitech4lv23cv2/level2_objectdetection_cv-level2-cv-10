_base_ = [
    './_base_/datasets/MyAlbum_Pipeline.py',
    './_base_/default_runtime.py',
    './_base_/schedules/schedule_1x.py'
]

data = dict(samples_per_gpu=4)
auto_scale_lr = dict(enable=False, base_batch_size=8)

# model settings
model = dict(
    type='CornerNet',
    backbone=dict(
        type='HourglassNet',
        downsample_times=5,
        num_stacks=2,
        stage_channels=[256, 256, 384, 384, 384, 512],
        stage_blocks=[2, 2, 2, 2, 2, 4],
        norm_cfg=dict(type='BN', requires_grad=True)),
    neck=None,
    bbox_head=dict(
        type='CornerHead',
        num_classes=10,
        in_channels=256,
        num_feat_levels=2,
        corner_emb_channels=1,
        loss_heatmap=dict(
            type='GaussianFocalLoss', alpha=2.0, gamma=4.0, loss_weight=1),
        loss_embedding=dict(
            type='AssociativeEmbeddingLoss',
            pull_weight=0.10,
            push_weight=0.10),
        loss_offset=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1)),
    # training and testing settings
    train_cfg=None,
    test_cfg=dict(
        corner_topk=100,
        local_maximum_kernel=3,
        distance_threshold=0.5,
        score_thr=0.05,
        max_per_img=100,
        nms=dict(type='soft_nms', iou_threshold=0.5, method='gaussian')))

# optimizer
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay= 0.01,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

lr_config = dict(
    _delete_=True,
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=1e-5)

optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

runner = dict(max_epochs=30)
checkpoint_config = dict(interval=10)

# load_from = 'https://download.openmmlab.com/mmdetection/v2.0/cornernet/cornernet_hourglass104_mstest_10x5_210e_coco/cornernet_hourglass104_mstest_10x5_210e_coco_20200824_185720-5fefbf1c.pth'
resume_from = '/opt/ml/baseline/mmdetection/work_dirs/CornerNet/best_bbox_mAP_epoch_19.pth'