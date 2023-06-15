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

def get_receptive_field(kernels, strides):
    r = [1, 1, 1]
    j = [1, 1, 1]
    for kernel, stride in zip(kernels, strides):
        for axis in range(len(kernel)):
            k = kernel[axis]
            s = stride[axis]

            # First conv
            j[axis] = j[axis] * s
            r[axis] = r[axis] + ((k - 1) * j[axis])

            # Second conv - stride always 1
            r[axis] = r[axis] + ((k - 1) * j[axis])

    return r
