# -*- coding: utf-8 -*-
"""
 
File:
    visualize.py
 
Authors: umla, soe
Date:
    26.09.20
 
Copyright:
    (c) Ibeo Automotive Systems GmbH, Hamburg, Germany
"""

import matplotlib.pyplot as plt


def plot_4_images(images):
    fig = plt.figure()
    ax = fig.add_subplot(2, 2, 1)
    imgplot = plt.imshow(images[0])
    ax.set_title('Before')

    ax = fig.add_subplot(2, 2, 2)
    imgplot = plt.imshow(images[1])

    ax = fig.add_subplot(2, 2, 3)
    imgplot = plt.imshow(images[2])

    ax = fig.add_subplot(2, 2, 4)
    imgplot = plt.imshow(images[3])

    imgplot.set_clim(0.0, 0.7)
    ax.set_title('After')

