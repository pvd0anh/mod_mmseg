dataset_type = 'DynamicEarthNet'
data_root = '/home/Hung_Data/HungData/mmseg_data/Datasets/DynamicEarthNet/data_monthly'

crop_size = (512, 512)  # The crop size during training.
train_pipeline = [  # Training pipeline.
    dict(type='LoadImageFromTIF'),  # First pipeline to load images from file path.
    dict(type='LoadAnnotations'),  # Second pipeline to load annotations for current image.
    # dict(type='RandomResize',  # Augmentation pipeline that resize the images and their annotations.
    #     scale=(2048, 1024),  # The scale of image.
    #     ratio_range=(0.5, 2.0),  # The augmented scale range as ratio.
    #     keep_ratio=True),  # Whether to keep the aspect ratio when resizing the image.
    dict(type='RandomCrop',  # Augmentation pipeline that randomly crop a patch from current image.
        crop_size=crop_size,  # The crop size of patch.
        cat_max_ratio=0.75),  # The max area ratio that could be occupied by single category.
    dict(type='RandomFlip',  # Augmentation pipeline that flip the images and their annotations
        prob=0.5),  # The ratio or probability to flip
    dict(type='PhotoMetricDistortion'),  # Augmentation pipeline that distort current image with several photo metric methods.
    dict(type='PackSegInputs')  # Pack the inputs data for the semantic segmentation.
]
test_pipeline = [
    dict(type='LoadImageFromTIF'),  # First pipeline to load images from file path
    dict(type='Resize',  # Use resize augmentation
        scale=(1024, 1024),  # Images scales for resizing.
        keep_ratio=True),  # Whether to keep the aspect ratio when resizing the image.
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),  # Load annotations for semantic segmentation provided by dataset.
    dict(type='PackSegInputs')  # Pack the inputs data for the semantic segmentation.
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromTIF', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ])
]
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='planet', seg_map_path='labels'),
        img_suffix='.png', seg_map_suffix='.png',
        ann_file = 'splits/train.txt',
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='planet', seg_map_path='labels'),
        img_suffix='.tif', seg_map_suffix='.png',
        ann_file = 'splits/val.txt',
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
