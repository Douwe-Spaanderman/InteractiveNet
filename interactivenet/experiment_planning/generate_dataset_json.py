import os
import json
import argparse
import numpy as np
import warnings
from pathlib import Path
import SimpleITK as sitk
import more_itertools
import re

def sanity_check(images, interactions, n_modalities, labels=None):
    def check(a, b, c=None):
        if c:
            return a == b == c
        else:
            return a == b

    len_images = len(images)   

    image_names_old = []

    for mod in range(n_modalities):
        image_names = list(set(["_".join(x[mod].name.split("_")[:-1]) for x in images])) 
        if image_names_old: 
            if not image_names_old == image_names:
                raise ValueError("Different subject names for modalities")
            else:
                image_names_old = image_names
        else:
            image_names_old = image_names
     
    interaction_names = list(set([x.with_suffix("").stem for x in interactions]))   

    if labels:
        label_names = list(set([x.with_suffix("").stem for x in labels]))

        if not len(labels) % len_images == len(interactions) % len_images == 0:
            raise AssertionError(
                "Length of database is not correct, e.g. more labels or interactions than images (or other way around)"
            )
        
        if (
            all(
                [
                    check(a, b, c)
                    for a, b, c in zip(image_names, label_names, interaction_names)
                ]
            )
            == False
        ):
            raise AssertionError(
                "images, labels and interactions do not have the correct names or are not ordered"
            )
    else:
        if not len(interactions) == len_images:
            raise AssertionError(
                "Length of database is not correct, e.g. more interactions than images (or other way around)"
            )

        if all([check(a, b) for a, b in zip(image_names, interaction_names)]) == False:
            raise AssertionError(
                "images and interactions do not have the correct names or are not ordered"
            )


def get_stats(inpath, n_modalities, all_subtypes=None):
    all_labels = []
    data = {}
    for mode in ["Ts", "Tr"]:
        data[mode] = {}
        images = sorted(
            [f for f in Path(inpath, "images" + mode).glob("**/*") if f.is_file()]
        )

        mod_number = np.unique([re.split(f'_|.nii',image.name)[-2] for image in images])

        for number in mod_number:
            if not len(number)==4 and number.is_integer():
                raise ValueError("Length must be 4 and must contain only integers.")

        #Check whether number of provided modalities matches the number of modalities in the images folder
        if not n_modalities == len(mod_number):
            raise ValueError("Amount of modalities does not correspond to the number of modalities in the images folder.")
        else:
            n_modalities==len(mod_number)

        images = list(more_itertools.chunked(images, n_modalities))
        labels = sorted(
            [f for f in Path(inpath, "labels" + mode).glob("**/*") if f.is_file()]
        )
        interactions = sorted(
            [f for f in Path(inpath, "interactions" + mode).glob("**/*") if f.is_file()]
        )

        if mode == "Ts":
            if not labels:
                warnings.warn("No labels present for test set")
                sanity_check(images, interactions, n_modalities)
            else:
                sanity_check(images, interactions, n_modalities, labels)
        else:
            sanity_check(images, interactions, n_modalities, labels)
    
        if all_subtypes:
            subtypes = [all_subtypes[x.stem.split(".nii")[0]] for x in labels]
            data[mode] = [
                {
                    
                    "image": [str(image[mod].relative_to(inpath)) for mod in range(n_modalities)],
                    "label": str(label.relative_to(inpath)),
                    "interaction": str(interaction.relative_to(inpath)),
                    "class": subtype,
                }
                for image, label, interaction, subtype in zip(images, labels, interactions, subtypes
                ) 
            ]
        else:
            data[mode] = [
                {
                    "image": [str(image[mod].relative_to(inpath)) for mod in range(n_modalities)],
                    "label": str(label.relative_to(inpath)),
                    "interaction": str(interaction.relative_to(inpath)),
                    "class": "",
                }
                for image, label, interaction in zip(images, labels, interactions)     
            ]

        all_labels.extend(labels)

    labels = []
    for label in all_labels:
        label = sitk.ReadImage(str(label), sitk.sitkUInt8, imageIO="NiftiImageIO")

        labels.extend(list(np.unique(sitk.GetArrayFromImage(label))))

    data["labels"] = sorted([*set(labels)])
    return data, mod_number

def main():
    parser = argparse.ArgumentParser(
        description="Create dataset.json with all metadata from dataset"
    )
    parser.add_argument("-t", "--task", required=True, type=str, help="Task name")
    parser.add_argument(
        "-m",
        "--modalities",
        nargs="+",
        required=True,
        help="What are the modalities in the dataset (consecutive order)",
    )
    parser.add_argument(
        "-l",
        "--labels",
        nargs="+",
        required=True,
        help="What are the labels in the dataset (consecutive order)",
    )
    parser.add_argument(
        "-type",
        "--subtypes",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Are there additional classes for a label",
    )
    parser.add_argument(
        "-d",
        "--description",
        nargs="?",
        default="",
        help="Dataset release",
    )
    parser.add_argument(
        "-r",
        "--release",
        nargs="?",
        default="0.0",
        help="Dataset release",
    )
    parser.add_argument(
        "-ref",
        "--reference",
        nargs="?",
        default=None,
        help="Dataset reference",
    )
    parser.add_argument(
        "-lic",
        "--licence",
        nargs="?",
        default=None,
        help="Which licence do you provide with this dataset",
    )
    args = parser.parse_args()
    n_modalities = len(args.modalities)

    inpath = Path(os.environ["interactivenet_raw"], args.task)

    if args.subtypes:
        if Path(inpath / "subtypes.json").exists():
            with open(inpath / "subtypes.json") as f:
                subtypes = json.load(f)
        else:
            KeyError(
                "args.subtypes was provided true, but cannot find subtypes.json in Raw directory."
            )
    else:
        subtypes = None

    stats, mod_number = get_stats(inpath, n_modalities, subtypes)

    if len(args.labels) == len(stats["labels"]):
        labels = {k: v for k, v in zip([*range(len(stats["labels"]))], args.labels)}
    elif len(args.labels) == len(stats["labels"]) - 1:
        labels = {"0": "background"}
        labels.update(
            {k: v for k, v in zip([*range(len(stats["labels"]) - 1)], args.labels)}
        )
    else:
        n_labels = stats["labels"]
        raise ValueError(
            f"Non matching labels, as there are {n_labels} found and {args.labels} provided"
        )
    
    modalities = {k: v for k, v in zip(mod_number, args.modalities)}


    with open(str(inpath / "dataset.json"), "w") as f:
        json.dump(
            {
                "description": args.description,
                "labels": labels,
                "modality": modalities,
                "name": args.task,
                "numTest": len(stats["Ts"]),
                "numTraining": len(stats["Tr"]),
                "reference": args.reference,
                "release": args.release,
                "test": stats["Ts"],
                "train": stats["Tr"],
            },
            f,
            indent=4,
        )


if __name__ == "__main__":
    main()

