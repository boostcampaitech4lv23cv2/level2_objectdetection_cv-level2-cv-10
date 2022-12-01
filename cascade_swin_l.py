_base_=[
    './_base_/datasets/albu_dataset.py',
    './_base_/models/cascade_rcnn_r50_fpn_fl.py',
    './_base_/schedules/schedule_1x.py',
    './_base_/default_runtime.py'
]

pretrained = '/opt/ml/baseline/UniverseNet/configs/_trash_/pretrained/swin_large_patch4_window7_224_22k.pth'
model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[192,384,768,1536]))
