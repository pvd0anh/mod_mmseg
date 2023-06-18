from mmengine import Config
from mmengine.runner import Runner


# cfg = Config.fromfile('configs/pspnet/pspnet_r50-d8_4xb2-40k_dynamicearthnet-1024x1024.py')
cfg = Config.fromfile('configs/pidnet/pidnet-s_2xb6-120k_1024x1024-dynamicearthnet.py')
# cfg = Config.fromfile('configs/unet/unet-s5-d16_fcn_4xb4-20k_dynamicearthnet-64x64.py')
cfg = Config.fromfile('configs/unet/unet_s5-d16_deeplabv3_4xb4-20k_dynamicearthnet-128x128.py')
# cfg = Config.fromfile('configs/unet/unet-s5-d16_pspnet_4xb4-20k_dynamicearthnet-64x64.py')
# print(f'Config:\n{cfg.pretty_text}')

cfg.work_dir = 'run_check/'


# cfg.dataset_type = 'DynamicEarthNet'
cfg.data_root = '/home/Hung_Data/HungData/mmseg_data/Datasets/DynamicEarthNet/data_monthly'
# cfg.data_root = '/mmsegmentation/data/Datasets/DynamicEarthNet/data_monthly'
cfg.train_dataloader.dataset.data_root= cfg.data_root
cfg.train_dataloader.batch_size = 4
cfg.train_dataloader.num_workers = 16

cfg.val_dataloader.dataset.data_root= cfg.data_root
# cfg.train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=3, max_iters=100)

cfg.train_cfg.max_iters = 300000
cfg.train_cfg.val_interval = 100
cfg.default_hooks.logger.interval = 100
cfg.default_hooks.logger.log_metric_by_epoch = False
cfg.default_hooks.checkpoint.interval = 10000

runner = Runner.from_cfg(cfg)
# start training
runner.train()

