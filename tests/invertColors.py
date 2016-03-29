# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 09:21:37 2014

@author: konkle
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def main(basename):
    image = plt.imread(basename + '.png')[:, :, :3]  # load 2D template

    v = image.sum(axis=2)

    d0 = np.diff(v, axis=0)
    y = d0.sum(axis=1)
    y /= y.max()
    ysteps = np.where(abs(y) > 0.4)[0]
    print(y.shape, y.max(), ysteps)
    assert len(ysteps) == 4

    d1 = np.diff(v, axis=1)
    x = d1.sum(axis=0)
    x /= x.max()
    xsteps = np.where(abs(x) > 0.5)[0]
    print(x.shape, x.max(), xsteps)
    assert len(xsteps) == 6

    y0s = ysteps[0], ysteps[2], ysteps[2], ysteps[2]
    y1s = ysteps[1], ysteps[3], ysteps[3], ysteps[3]
    x0s = xsteps[0], xsteps[0], xsteps[2], xsteps[4]
    x1s = xsteps[1], xsteps[1], xsteps[3], xsteps[5]
    for y0, y1, x0, x1 in zip(y0s, y1s, x0s, x1s):
        part = image[y0+2:y1, x0+2:x1, :]
        part = 1 - part
        hsv = mpl.colors.rgb_to_hsv(part)
        hsv[:, :, 0] -= 0.5
        hsv[hsv < 0] += 1
        image[y0+2:y1, x0+2:x1, :] = mpl.colors.hsv_to_rgb(hsv)

    plt.imsave(basename + '-inverted.png', image)
    print('"{0}-inverted.png" has been created'.format(basename))

if __name__ == '__main__':
    main('test-color')
    print('ready')
