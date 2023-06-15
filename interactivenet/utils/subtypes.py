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
