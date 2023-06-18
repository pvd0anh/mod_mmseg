_base_ = [
    '../_base_/models/deeplabv3_unet_s5-d16.py', '../_base_/datasets/dynamicearthnet.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]


crop_size = (256, 256)
data_preprocessor = dict(size=crop_size)
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/unet/deeplabv3_unet_s5-d16_64x64_40k_drive/deeplabv3_unet_s5-d16_64x64_40k_drive_20201226_094047-0671ff20.pth'
# checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/unet/deeplabv3_unet_s5-d16_ce-1.0-dice-3.0_256x256_40k_hrf/deeplabv3_unet_s5-d16_ce-1.0-dice-3.0_256x256_40k_hrf_20211210_202032-59daf7a4.pth'
checkpoint_file = '/home/Hung_Data/HungData/mmseg_data/mod_mmseg/weights/deeplabv3_unet_s5-d16_ce-1.0-dice-3.0_256x256_40k_hrf_20211210_202032-59daf7a4.pth'
model = dict(
    # pretrained=checkpoint_file,
    backbone=dict(init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file)),
    decode_head=dict(num_classes=6),
    auxiliary_head=dict(num_classes=6),
    data_preprocessor=data_preprocessor,
    test_cfg=dict(crop_size=crop_size, stride=(85, 85)))


train_pipeline = [  # Training pipeline.
    dict(type='LoadImageFromTIF'),  # First pipeline to load images from file path.
    dict(type='LoadAnnotations'),  # Second pipeline to load annotations for current image.
    # dict(type='RandomResize',  # Augmentation pipeline that resize the images and their annotations.
    #     scale=(2048, 1024),  # The scale of image.
    #     ratio_range=(0.5, 2.0),  # The augmented scale range as ratio.
    #     keep_ratio=True),  # Whether to keep the aspect ratio when resizing the image.
    dict(type='RandomCrop',  # Augmentation pipeline that randomly crop a patch from current image.
        crop_size=(480, 480),  # The crop size of patch.
        cat_max_ratio=0.75),  # The max area ratio that could be occupied by single category.
    dict(type='RandomFlip',  # Augmentation pipeline that flip the images and their annotations
        prob=0.5),  # The ratio or probability to flip
    dict(type='PhotoMetricDistortion'),  # Augmentation pipeline that distort current image with several photo metric methods.
    dict(type='PackSegInputs')  # Pack the inputs data for the semantic segmentation.
]

test_pipeline = [
    dict(type='LoadImageFromTIF'),  # First pipeline to load images from file path
    dict(type='Resize',  # Use resize augmentation
        scale=(480, 480),  # Images scales for resizing.
        keep_ratio=True),  # Whether to keep the aspect ratio when resizing the image.
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),  # Load annotations for semantic segmentation provided by dataset.
    dict(type='PackSegInputs')  # Pack the inputs data for the semantic segmentation.
]


train_dataloader = dict(
    dataset=dict(
        img_suffix='.tif', seg_map_suffix='.png',
        pipeline=train_pipeline))
val_dataloader = dict(
    dataset=dict(
        img_suffix='.tif', seg_map_suffix='.png',
        pipeline=test_pipeline))
# crop_size = (64, 64)
# data_preprocessor = dict(size=crop_size)
# model = dict(
#     # data_preprocessor=data_preprocessor,
#     decode_head=dict(num_classes=6),
#     auxiliary_head=dict(num_classes=6),
#     # model training and testing settings
#     train_cfg=dict(),
#     test_cfg=dict(crop_size=(64, 64), stride=(42, 42)))
#     # test_cfg=dict(mode='whole'))
# train_dataloader = dict(batch_size=2, num_workers=4)
# val_dataloader = dict(batch_size=1, num_workers=4)
# test_dataloader = val_dataloader
log_level = 'INFO'
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='MlflowLoggerHook', exp_name='check_running_with_MlflowLoggerHook')
    ])


# _base_ = [
#     '../_base_/models/deeplabv3_unet_s5-d16.py',
#     '../_base_/datasets/chase_db1.py', '../_base_/default_runtime.py',
#     '../_base_/schedules/schedule_40k.py'
# ]
# crop_size = (128, 128)
# data_preprocessor = dict(size=crop_size)
# model = dict(
#     data_preprocessor=data_preprocessor,
#     test_cfg=dict(crop_size=(128, 128), stride=(85, 85)))

# _base_ = [
#     '../_base_/models/pspnet_unet_s5-d16.py', '../_base_/datasets/drive.py',
#     '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
# ]
# crop_size = (64, 64)
# data_preprocessor = dict(size=crop_size)
# model = dict(
#     data_preprocessor=data_preprocessor,
#     test_cfg=dict(crop_size=(64, 64), stride=(42, 42)))
