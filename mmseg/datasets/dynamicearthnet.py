from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class DynamicEarthNet(BaseSegDataset):

    METAINFO = dict(
        classes=('impervious surface', 'agriculture', 'forest & other', 'wetland', 'soil', 'water'),
        palette=[[96, 96, 96], [204, 204, 0], [0, 204, 0], [0, 0, 153], [153, 76, 0], [0, 128, 255]])

    def __init__(self,
                 img_suffix='.tif',
                 seg_map_suffix='.tif',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
