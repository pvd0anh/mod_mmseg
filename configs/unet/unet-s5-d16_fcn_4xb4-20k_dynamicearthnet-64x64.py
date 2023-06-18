_base_ = [
    '../_base_/models/fcn_unet_s5-d16.py', '../_base_/datasets/dynamicearthnet.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]


crop_size = (64, 64)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    test_cfg=dict(crop_size=(64, 64), stride=(42, 42)))


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
