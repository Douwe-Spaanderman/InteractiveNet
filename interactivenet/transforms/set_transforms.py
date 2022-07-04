from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    DivisiblePadd,
)

# I want something which can take the right transforms and is able to take arguments
class GenericTransform(object):
    def __init__(self) -> None:
        self.transform = []

    def compose(self):
        self.transform = Compose(
            self.transform
        )

class NIFTITransform(GenericTransform):
    def __init__(self) -> None:
        super().__init__()
        print(self.transform)

class NUMPYTransform(GenericTransform):
    def __init__(self) -> None:
        super().__init__()
        print('no')

class PreProcessTransform(NIFTITransform):
    def __init__(
        self, 
        target_spacing: Tuple[float],
        relaxed_bbox: Tuple[float] = (10, 10, 2),
        divisble_using: int = 16,
        ) -> None:
        super().__init__()

        self.transform.append(
            Resamplingd(
                keys=["image", "annotation", "mask"],
                pixdim=target_spacing,
                ),
            BoudingBoxd(
                keys=["image", "annotation", "mask"],
                on="mask",
                relaxation=relaxed_bbox,
                ),
            NormalizeIntensityd(
                keys=["image"],
                nonzero=True,
                channel_wise=False,
                ),
            EGDMapd(
                keys=["annotation"],
                image="image",
                lamb=1,
                iter=4,
                ),
            DivisiblePadd(
                keys=["image", "annotation", "mask"],
                k=divisble_using
                ),
            )

class TrainingTransform(GenericTransform):
    def __init__(self):
        super().__init__()
        print('no')

NIFTITransform()

# self.transforms = Compose(
#     [
#     LoadImaged(
#         keys=["image", "annotation", "mask"]
#         ),
#     EnsureChannelFirstd(
#         keys=["image", "annotation", "mask"]
#         ),
#     Resamplingd(
#         keys=["image", "annotation", "mask"],
#         pixdim=target_spacing,
#         ),
#     BoudingBoxd(
#         keys=["image", "annotation", "mask"],
#         on="mask",
#         relaxation=relaxed_bbox,
#         ),
#     NormalizeIntensityd(
#         keys=["image"],
#         nonzero=True,
#         channel_wise=False,
#         ),
#     EGDMapd(
#         keys=["annotation"],
#         image="image",
#         lamb=1,
#         iter=4,
#         ),
#     DivisiblePadd(
#         keys=["image", "annotation", "mask"],
#         k=divisble_using
#         ),
#     ]
# )