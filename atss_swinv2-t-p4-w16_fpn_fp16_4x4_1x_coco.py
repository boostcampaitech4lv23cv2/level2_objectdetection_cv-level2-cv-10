_base_ = [
    './_base_/models/atss_swinv2-t-p4-w16_fpn.py',
    './_base_/datasets/coco_detection.py',
    './_base_/schedules/schedule_1x.py',
    './_base_/default_runtime.py'
]

data = dict(samples_per_gpu=8)

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'cpb_mlp': dict(decay_mult=0.),
            'logit_scale': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(warmup_iters=1000)
lr_config = dict(step=[27, 33])
runner = dict(type='EpochBasedRunner', max_epochs=36)
fp16 = dict(loss_scale=dict(init_scale=512))