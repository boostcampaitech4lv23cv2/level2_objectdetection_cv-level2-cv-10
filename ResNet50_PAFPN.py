_base_=[
    './_base_/datasets/MyAlbum_Pipeline.py',
    './_base_/models/cascade_rcnn_r50_fpn.py',
    './_base_/schedules/schedule_1x.py',
    './_base_/default_runtime.py'
]

data = dict(samples_per_gpu=4)
auto_scale_lr = dict(enable=False, base_batch_size=8)

model = dict(
    backbone=dict(
        type='DetectoRS_ResNet',
        conv_cfg=dict(type='ConvAWS'),
        sac=dict(type='SAC', use_deform=True),
        stage_with_sac=(False, True, True, True)
    ),
    
    neck=dict(
        type='PAFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    
    rpn_head=dict(
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2],
            strides=[4, 8, 16, 32, 64]))
)

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
checkpoint_config = dict(interval=2)