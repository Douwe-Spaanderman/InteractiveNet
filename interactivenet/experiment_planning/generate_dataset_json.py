import os
import json
import argparse
import numpy as np
import warnings
from pathlib import Path
import SimpleITK as sitk


def sanity_check(images, labels, interactions, mode="Tr"):
    def check(a, b, c=None):
        if c:
            return a == b == c
        else:
            return a == b

    len_images = len(images)
    if mode == "Tr":
        if not len(labels) % len_images == len(interactions) % len_images == 0:
            raise AssertionError(
                "Length of database is not correct, e.g. more labels or interactions than images"
            )

    image_names = list(set(["_".join(x.name.split("_")[:-1]) for x in images]))
    interaction_names = list(set([x.with_suffix("").stem for x in interactions]))
    if mode == "Tr":
        label_names = list(set([x.with_suffix("").stem for x in labels]))
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
        if all([check(a, b) for a, b in zip(image_names, interaction_names)]) == False:
            raise AssertionError(
                "images and interactions do not have the correct names or are not ordered"
            )


def get_stats(inpath, all_subtypes=None):
    all_labels = []
    data = {}
    for mode in ["Ts", "Tr"]:
        data[mode] = {}
        images = sorted(
            [f for f in Path(inpath, "images" + mode).glob("**/*") if f.is_file()]
        )
        labels = sorted(
            [f for f in Path(inpath, "labels" + mode).glob("**/*") if f.is_file()]
        )
        interactions = sorted(
            [f for f in Path(inpath, "interactions" + mode).glob("**/*") if f.is_file()]
        )

        if mode == "Ts":
            if not labels:
                warnings.warn("No labels present for test set")
                labels = len(images) * [""]
                sanity_check(images, labels, interactions)
            else:
                sanity_check(images, labels, interactions, mode="Tr")
        else:
            sanity_check(images, labels, interactions, mode)

        if all_subtypes:
            subtypes = [all_subtypes[x.stem.split(".nii")[0]] for x in labels]
            data[mode] = [
                {
                    "image": str(image.relative_to(inpath)),
                    "label": str(label.relative_to(inpath)),
                    "interaction": str(interaction.relative_to(inpath)),
                    "class": subtype,
                }
                for image, label, interaction, subtype in zip(
                    images, labels, interactions, subtypes
                )
            ]
        else:
            data[mode] = [
                {
                    "image": str(image.relative_to(inpath)),
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
    return data


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

    stats = get_stats(inpath, subtypes)

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
            f"Non matching labels, as there are {n_labels} found and {args.labels} providedd"
        )

    modalities = {k: v for k, v in zip([*range(len(args.modalities))], args.modalities)}

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
