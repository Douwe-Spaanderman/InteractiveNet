import logging
import torch
import os

from typing import Callable, Sequence, Union, Tuple, List

from lib.transforms.transforms import (
    AnnotationToChanneld,
    TestTimeFlippingd,
    OriginalSized,
)
from monai.inferers import Inferer, SimpleInferer
from monai.data import MetaTensor
from monai.transforms import (
    AddChanneld,
    EnsureTyped,
    LoadImaged,
    CastToTyped,
    ConcatItemsd,
    ToTensord,
    ToTensord,
    MeanEnsembled,
    KeepLargestConnectedComponentd,
    FillHolesd,
    SqueezeDimd,
)

from monai.apps.deepgrow.transforms import (
    AddGuidanceFromPointsd,
)

from interactivenet.transforms.transforms import (
    Resamplingd,
    BoudingBoxd,
    NormalizeValuesd,
    EGDMapd,
)

from monailabel.interfaces.tasks.infer import InferTask, InferType

logger = logging.getLogger(__name__)

from typing import Sequence, Tuple



class InteractiveNet(InferTask):
    """ """

    def __init__(
        self,
        path,
        network=None,
        type=InferType.DEEPGROW,
        labels=None,
        dimension=3,
        ensemble: bool = False,
        tta: bool = False,
        median_shape: Tuple[float] = (128, 128, 64),
        target_spacing: Tuple[float] = (1.0, 1.0, 1.0),
        relax_bbox: Union[float, Tuple[float]] = 0.1,
        divisble_using: Union[int, Tuple[int]] = (16, 16, 8),
        clipping: List[float] = [],
        intensity_mean: float = 0,
        intensity_std: float = 0,
        ct: bool = False,
        tmp_folder: str = "/tmp/",
        description="Volumetric Interactivenet",
        **kwargs,
    ):
        super().__init__(
            path=path,
            network=network,
            type=type,
            labels=labels,
            dimension=dimension,
            description=description,
            input_key="image",
            output_label_key="label",
            output_json_key="result",
            **kwargs,
        )

        self.ensemble = ensemble
        self.tta = tta
        self.median_shape = median_shape
        self.target_spacing = target_spacing
        self.relax_bbox = relax_bbox
        self.divisble_using = divisble_using
        self.clipping = clipping
        self.intensity_mean = intensity_mean
        self.intensity_std = intensity_std
        self.ct = ct

    def pre_transforms(self, data=None) -> Sequence[Callable]:
        t = [
            LoadImaged(keys="image"),
        ]

        self.add_cache_transform(t, data)

        t.extend(
            [
                AddGuidanceFromPointsd(
                    ref_image="image",
                    guidance="interaction",
                    depth_first=False,
                    dimensions=3,
                ),
                AnnotationToChanneld(
                    ref_image="image", guidance="interaction", method="interactivenet"
                ),
                AddChanneld(keys=["image"]),
                Resamplingd(
                    keys=["image", "interaction"],
                    pixdim=self.target_spacing,
                ),
                BoudingBoxd(
                    keys=["image", "interaction"],
                    on="interaction",
                    relaxation=self.relax_bbox,
                    divisiblepadd=self.divisble_using,
                ),
                NormalizeValuesd(
                    keys=["image"],
                    clipping=self.clipping,
                    mean=self.intensity_mean,
                    std=self.intensity_std,
                ),
                EGDMapd(
                    keys=["interaction"],
                    image="image",
                    lamb=1,
                    iter=4,
                    logscale=True,
                    ct=self.ct,
                ),
                CastToTyped(
                    keys=["image", "interaction"], dtype=(torch.float32, torch.float32)
                ),
                ToTensord(keys=["image", "interaction"]),
            ]
        )

        if self.tta:
            t.extend(
                [
                    TestTimeFlippingd(keys=["image", "interaction"]),
                ]
            )
            dim = 1
        else:
            dim = 0

        t.extend(
            [
                ConcatItemsd(keys=["image", "interaction"], name="image", dim=dim),
                ToTensord(keys=["image"]),
            ]
        )
        return t

    def inferer(self, data=None) -> Inferer:
        return SimpleInferer()

    def post_transforms(self, data=None) -> Sequence[Callable]:
        t = [
            EnsureTyped(keys="label", device=data.get("device") if data else None),
        ]

        if self.tta:
            t.extend(
                [
                    TestTimeFlippingd(keys=["label"], back=True),
                ]
            )

        if self.ensemble or self.tta:
            t.extend(
                [
                    MeanEnsembled(keys="label"),
                ]
            )

        t.extend(
            [
                OriginalSized(
                    img_key="label",
                    ref_meta="image",
                    discreet=True,
                    device=data.get("device") if data else None,
                ),
                KeepLargestConnectedComponentd(keys="label"),
                FillHolesd(keys="label"),
                SqueezeDimd(keys="label"),
            ]
        )

        return t

    def run_inferer(self, data, convert_to_batch=True, device="cuda"):
        if self.tta:
            convert_to_batch = False

        if self.ensemble:
            pred = []
            models = self.path
            for model in models:
                self.model = model
                output = super().run_inferer(
                    data, convert_to_batch=convert_to_batch, device=device
                )

                pred.append(output["label"])

            output["label"] = MetaTensor(
                torch.stack(pred), affine=output["label"].affine
            )
        else:
            super().run_inferer(data, convert_to_batch=convert_to_batch, device=device)

        return data

    def _get_network(self, device):
        path = self.get_path()
        logger.info(f"Infer model path: {path}")
        if not path and not self.network:
            raise MONAILabelException(
                MONAILabelError.INFERENCE_ERROR,
                f"Model Path ({self.path}) does not exist/valid",
            )

        cached = self._networks.get(device)
        statbuf = os.stat(path) if path else None
        network = None
        if cached:
            if statbuf and statbuf.st_mtime == cached[1]:
                network = cached[0]
            elif statbuf:
                logger.warning(
                    f"Reload model from cache.  Prev ts: {cached[1]}; Current ts: {statbuf.st_mtime}"
                )

        if network is None:
            network = torch.load(path, map_location=torch.device(device))
            print(network)
            network.eval()

        return network

    def get_path(self):
        if not self.path:
            return None

        paths = self.path

        if len(paths) == 1:
            for path in reversed(paths):
                if path and os.path.exists(path):
                    return path
        else:
            if self.model in paths and os.path.exists(self.model):
                return self.model
        return None
