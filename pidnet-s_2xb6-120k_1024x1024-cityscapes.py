dataset_type = 'DynamicEarthNet'
data_root = '/home/Hung_Data/HungData/mmseg_data/Datasets/DynamicEarthNet/data_monthly'
crop_size = (1024, 1024)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(1024, 1024),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(1024, 1024), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[[{
            'type': 'Resize',
            'scale_factor': 0.5,
            'keep_ratio': True
        }, {
            'type': 'Resize',
            'scale_factor': 0.75,
            'keep_ratio': True
        }, {
            'type': 'Resize',
            'scale_factor': 1.0,
            'keep_ratio': True
        }, {
            'type': 'Resize',
            'scale_factor': 1.25,
            'keep_ratio': True
        }, {
            'type': 'Resize',
            'scale_factor': 1.5,
            'keep_ratio': True
        }, {
            'type': 'Resize',
            'scale_factor': 1.75,
            'keep_ratio': True
        }],
                    [{
                        'type': 'RandomFlip',
                        'prob': 0.0,
                        'direction': 'horizontal'
                    }, {
                        'type': 'RandomFlip',
                        'prob': 1.0,
                        'direction': 'horizontal'
                    }], [{
                        'type': 'LoadAnnotations'
                    }], [{
                        'type': 'PackSegInputs'
                    }]])
]
train_dataloader = dict(
    batch_size=6,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='CityscapesDataset',
        data_root='data/cityscapes/',
        data_prefix=dict(img_path='planet', seg_map_path='labels'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                type='RandomResize',
                scale=(2048, 1024),
                ratio_range=(0.5, 2.0),
                keep_ratio=True),
            dict(
                type='RandomCrop', crop_size=(1024, 1024), cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(type='GenerateEdge', edge_width=4),
            dict(type='PackSegInputs')
        ]))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CityscapesDataset',
        data_root='data/cityscapes/',
        data_prefix=dict(
            img_path='leftImg8bit/val', seg_map_path='gtFine/val'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs')
        ]))
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CityscapesDataset',
        data_root='data/cityscapes/',
        data_prefix=dict(
            img_path='leftImg8bit/val', seg_map_path='gtFine/val'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs')
        ]))
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False
tta_model = dict(type='SegTTAModel')
class_weight = [
    0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489, 0.8786,
    1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955, 1.0865, 1.1529,
    1.0507
]
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/pidnet/pidnet-s_imagenet1k_20230306-715e6273.pth'
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=(1024, 1024))
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[
            1042.59240722656, 915.618408203125, 671.260559082031,
            2605.20922851562
        ],
        std=[
            957.958435058593, 715.548767089843, 596.943908691406,
            1059.90319824218
        ],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size=(1024, 1024)),
    backbone=dict(
        type='PIDNet',
        in_channels=4,
        channels=32,
        ppm_channels=96,
        num_stem_blocks=2,
        num_branch_blocks=3,
        align_corners=False,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        act_cfg=dict(type='ReLU', inplace=True),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/pidnet/pidnet-s_imagenet1k_20230306-715e6273.pth'
        )),
    decode_head=dict(
        type='PIDHead',
        in_channels=128,
        channels=128,
        num_classes=19,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        act_cfg=dict(type='ReLU', inplace=True),
        align_corners=True,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                class_weight=[
                    0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754,
                    1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037,
                    1.0865, 1.0955, 1.0865, 1.1529, 1.0507
                ],
                loss_weight=0.4),
            dict(
                type='OhemCrossEntropy',
                thres=0.9,
                min_kept=131072,
                class_weight=[
                    0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754,
                    1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037,
                    1.0865, 1.0955, 1.0865, 1.1529, 1.0507
                ],
                loss_weight=1.0),
            dict(type='BoundaryLoss', loss_weight=20.0),
            dict(
                type='OhemCrossEntropy',
                thres=0.9,
                min_kept=131072,
                class_weight=[
                    0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754,
                    1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037,
                    1.0865, 1.0955, 1.0865, 1.1529, 1.0507
                ],
                loss_weight=1.0)
        ]),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
iters = 120000
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005),
    clip_grad=None)
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=0,
        power=0.9,
        begin=0,
        end=120000,
        by_epoch=False)
]
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=120000, val_interval=12000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=12000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
randomness = dict(seed=304)
work_dir = '.'
