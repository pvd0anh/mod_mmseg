from mmengine import Config
from mmengine.runner import Runner


# cfg = Config.fromfile('configs/pspnet/pspnet_r50-d8_4xb2-40k_dynamicearthnet-512x1024.py')
cfg = Config.fromfile('configs/pidnet/pidnet-s_2xb6-120k_1024x1024-dynamicearthnet.py')
print(f'Config:\n{cfg.pretty_text}')

cfg.model.data_preprocessor.mean = [1042.59240722656, 915.618408203125, 671.260559082031]
cfg.model.data_preprocessor.std = [957.958435058593, 715.548767089843, 596.943908691406]
cfg.model.data_preprocessor.size=(1024, 1024)

cfg.model.backbone.in_channels=3
cfg.work_dir = 'run_check/'

# cfg.dataset_type = 'DynamicEarthNet'
cfg.data_root = '/home/Hung_Data/HungData/mmseg_data/Datasets/DynamicEarthNet/data_monthly'
cfg.train_dataloader.dataset.data_root= cfg.data_root

cfg.val_dataloader.dataset.data_root= cfg.data_root

cfg.train_cfg.max_iters = 100
cfg.train_cfg.val_interval = 100
cfg.default_hooks.logger.interval = 10
cfg.default_hooks.checkpoint.interval = 100

runner = Runner.from_cfg(cfg)
# start training
runner.train()


""" 
+--------------------+-------+-------+
|       Class        |  IoU  |  Acc  |
+--------------------+-------+-------+
| impervious surface |  0.01 |  0.01 |
|    agriculture     |  0.0  |  0.0  |
|   forest & other   | 42.14 | 95.15 |
|      wetland       |  0.0  |  0.0  |
|        soil        |  6.55 |  6.85 |
|       water        |  1.78 |  2.4  |
+--------------------+-------+-------+


+--------------------+-------+-------+
|       Class        |  IoU  |  Acc  |
+--------------------+-------+-------+
| impervious surface |  0.5  |  0.56 |
|    agriculture     |  0.49 |  0.5  |
|   forest & other   | 53.15 | 96.56 |
|      wetland       |  0.0  |  0.0  |
|        soil        | 33.36 | 53.45 |
|       water        |  0.01 |  0.01 |
+--------------------+-------+-------+
05/25 09:11:39 - mmengine - INFO - Iter(val) [120/120]    aAcc: 51.7500  mIoU: 14.5800  mAcc: 25.1800  data_time: 0.0065  time: 0.1455

+--------------------+------+-------+
|       Class        | IoU  |  Acc  |
+--------------------+------+-------+
| impervious surface | 0.0  |  0.0  |
|    agriculture     | 0.0  |  0.0  |
|   forest & other   | 39.1 | 99.99 |
|      wetland       | 0.0  |  0.0  |
|        soil        | 0.18 |  0.18 |
|       water        | 0.0  |  0.0  |
+--------------------+------+-------+
05/25 10:06:30 - mmengine - INFO - Iter(val) [120/120]    aAcc: 39.0300  mIoU: 6.5500  mAcc: 16.7000  data_time: 0.0064  time: 0.1389

"""