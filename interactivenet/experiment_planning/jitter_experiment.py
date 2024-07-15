#   Copyright 2023 Biomedical Imaging Group Rotterdam, Departments of
#   Radiology and Nuclear Medicine, Erasmus MC, Rotterdam, The Netherlands
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#   
#   http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import os
import random
import warnings
import argparse
import numpy as np
import SimpleITK as sitk
from pathlib import Path

from typing import Union, Optional, List

class JitterItem(object):
    def __init__(self, Interaction: Path, Jitter: Optional[Union[str, List[str]]] = None):
        self.InteractionName = Interaction.name
        self.Interaction = self._read_Image(Interaction)
        self.Spacing = self.Interaction.GetSpacing()
        self.Origin = self.Interaction.GetOrigin()
        self.Direction = self.Interaction.GetDirection()
        self.Interaction = self._from_simpleITK(self.Interaction)
        self.Anisotropic = self._check_Anisotropic()
        self.Dimensions = self.Interaction.shape
        self.NewInteraction = np.zeros(shape=self.Interaction.shape)

        if Jitter:
            if Jitter == "default":
                if not self.Anisotropic:
                    self.Jitter = [3, 3, 3]
                else:
                    self.Jitter = [1, 3, 3]

                print(f"Using default settings for extreme points: {Jitter}")
            else:
                self.Jitter = [int(x) for x in Jitter]

        self.run_experiment()

    def _from_simpleITK(self, img):
        if img is not None:
            return sitk.GetArrayFromImage(img)

    def _to_simpleITK(self, img):
        if img is not None:
            img = sitk.GetImageFromArray(img)
            img.SetSpacing(self.Spacing)
            img.SetOrigin(self.Origin)
            img.SetDirection(self.Direction)
            return img

    def _read_Image(self, img):
        if img:
            return sitk.ReadImage(str(img), imageIO="NiftiImageIO")

    def _save_Image(self, img, name):
        if img:
            return sitk.WriteImage(img, name)

    def _check_Anisotropic(self):
        def check(spacing):
            return np.max(spacing) / np.min(spacing) >= 3

        return check(self.Spacing)

    def perform_jitter(self, point, jitter, axis):
        new = point[axis] + jitter[axis]
        if new < self.Dimensions[axis] and new > 0:
            return new
        else:
            return point[axis]

    def run_experiment(self):
        # Get a random jitter
        for point in zip(*np.where(self.Interaction)):
            jitter = []
            for j in self.Jitter:
                jitter.append(random.randint(-j,j))

            z = self.perform_jitter(point, jitter, axis=0)
            y = self.perform_jitter(point, jitter, axis=1)
            x = self.perform_jitter(point, jitter, axis=2)

            self.NewInteraction[z, y, x] = 1

        self.NewInteraction = self._to_simpleITK(self.NewInteraction)

def jitter_experiment(
    task: str,
    experiments: int = 50,
    jitter: Optional[Union[str, List[str]]] = None,
):


    inpath = Path(os.environ["interactivenet_raw"], task)
    interactionsTs = sorted(
        [f for f in Path(inpath, "interactionsTs").glob("**/*") if f.is_file()]
    )

    experimentpath = inpath / "Experiment_Jitter"
    for idx in range(experiments):
        outpath = experimentpath / f"Jitter{idx}"
        outpath.mkdir(parents=True, exist_ok=True)
        for interaction in interactionsTs:
            interaction = JitterItem(
                Interaction=interaction,
                Jitter=jitter,
            )
            interaction._save_Image(interaction.NewInteraction, outpath / interaction.InteractionName)
        
def main():
    parser = argparse.ArgumentParser(
        description="InteractiveNet fingerprinting, experiment planning and procesing"
    )
    parser.add_argument("-t", "--task", required=True, type=str, help="Task name")
    parser.add_argument(
        "-e",
        "--experiments",
        nargs="?",
        default=50,
        type=int,
        help="How many variations do you want to create for each sample?",
    )
    parser.add_argument(
        "-j",
        "--jitter",
        nargs="+",
        default="default",
        help="How much do you want to jitter the points? If so, please provide jitter for each axes, i.e. [1, 5, 5]",
    )

    args = parser.parse_args()

    # This is stupid but whatever
    if args.jitter and len(args.jitter) != 3:
        if args.jitter != "default":
            raise KeyError(
                f"argument jitter (-e) should either be None, default or a list of 3 not: {args.jitter}"
            )

    jitter_experiment(
        task=args.task,
        experiments=args.experiments,
        jitter=args.jitter
    )

if __name__ == "__main__":
    main()