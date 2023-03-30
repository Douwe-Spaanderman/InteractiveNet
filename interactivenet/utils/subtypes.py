##################################
##################################
## ALL CODE HERE IS OUTDATED!!! ##
##################################
##################################
##################################

import os
import json
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Legacy code: changing subtype json to correct format"
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Do you want to overwrite the json file",
    )
    parser.add_argument("-t", "--task", required=True, help="Task name")
    args = parser.parse_args()

    inpath = Path(os.environ["interactivenet_raw"], args.task)

    with open(inpath / "subtypes.json") as f:
        subtypes = json.load(f)

    data = {}
    for key, values in subtypes.items():
        data.update({value: key for value in values})

    if args.overwrite:
        outfile = inpath / "subtypes.json"
    else:
        outfile = inpath / "new_subtypes.json"

    with open(str(outfile), "w") as f:
        json.dump(
            data,
            f,
            indent=4,
        )


if __name__ == "__main__":
    main()
