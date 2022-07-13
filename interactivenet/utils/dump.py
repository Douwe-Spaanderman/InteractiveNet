def get_receptive_field(kernels, strides):
    r = [1,1,1]
    j = [1,1,1]
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