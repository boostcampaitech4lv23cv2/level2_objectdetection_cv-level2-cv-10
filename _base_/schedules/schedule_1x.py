# optimizer
optimizer = dict(type='AdamW', lr=0.0002, weight_decay=0.01)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=50,
    warmup_ratio=0.0001,
    step=[4, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
