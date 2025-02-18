from enum import Enum
from typing import List, NamedTuple


class LayerConfig(NamedTuple):
    indices: List[int]
    channels: List[int]


class Backbone(Enum):
    YOLO_V11_M = "yolov11m"
    VGG_16 = "vgg16"
    MOBILENET_V3_L = "mobilenetv3-large"
    EFFICIENTNET_V2_M = "efficientnetv2-m"

    @property
    def config(self) -> LayerConfig:
        """Get the layer configuration for the backbone."""
        configs = {
            Backbone.YOLO_V11_M: LayerConfig(
                indices=[9, 10],
                channels=[512, 512],
            ),
            Backbone.VGG_16: LayerConfig(
                indices=[23, 30],
                channels=[512, 512],
            ),
            Backbone.MOBILENET_V3_L: LayerConfig(
                indices=[5, 12],
                channels=[40, 112],
            ),
            Backbone.EFFICIENTNET_V2_M: LayerConfig(
                indices=[6, 7],
                channels=[304, 512],
            ),
        }
        return configs[self]
