import numpy as np

image = np.array([
    [1, 2, 3, 0],
    [4, 5, 6, 1],
    [7, 8, 9, 2],
    [1, 1, 1, 1]
])


edge_kernel = np.array([
    [1, 0],
    [0, -1]
])

blur_kernel = np.array([
    [1, 1],
    [1, 1]
]) / 4

identity_kernel = np.array([
    [0, 0],
    [0, 1]
])

# The answer should be
#([
#[-4,-4,2]
#[-4,-4,4]
#[6,7,8]

def conv2d(img,kernel,stride=1):
    kh, kw = kernel.shape
    ih,iw = img.shape
    oh = ih - kh + stride
    ow = iw - kw + stride

    output = np.zeros((oh,ow))

    for i in range(oh):
        for j in range(ow):
            patch = img[i:i+kh, j:j+kw]
            output[i,j] = np.sum(patch * kernel)

    return output

result = conv2d(image,edge_kernel)
print(result)

result = conv2d(image,identity_kernel)
print(result)
result = conv2d(image,blur_kernel)
print(result)