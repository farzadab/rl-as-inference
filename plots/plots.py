# -*- coding: utf-8 -*-
'''
Ploting utility for plotting dynamically changing data with Matplotlib
'''
# Python2 Compatibility:
from __future__ import absolute_import, division, print_function
from builtins import (bytes, str, open, super, range,
                      zip, round, input, int, pow, object)

from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from matplotlib import cm
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import time

plt.ion()


class Plot(object):
    def __init__(self, nrows=1, ncols=1, parent=None):
        self.fig = None
        self.nrows = nrows
        self.ncols = ncols
        if parent is None:
            self.parent = self
            # TODO: make up a name
            self.fig = plt.figure(figsize=(4.5*ncols, 4.5*nrows))
            self.subplot_cnt = 0
            # self.fig, self.subplots = plt.subplots(nrows, ncols, figsize=(4.5*ncols, 4.5*nrows))
            # self.subplots = np.array(self.subplots).reshape(-1)[::-1].tolist()
        else:
            self.parent = parent
    
    def _get_subplot(self, projection=None):
        self.subplot_cnt += 1
        return self.fig.add_subplot(self.nrows, self.ncols, self.subplot_cnt, projection=projection)
            
    def _redraw(self):
        if self.fig:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        else:
            self.parent._redraw()
    
    # Originally from: http://www.icare.univ-lille1.fr/tutorials/convert_a_matplotlib_figure
    def get_image(self):
        """
        @brief Convert its own figure to a 3D numpy array with RGB channels and return it
        @return a numpy 3D array of RGB values
        """
        if self.fig is None:
            return self.parent.get_image() 
        # draw the renderer  .. TODO: remove?
        self.fig.canvas.draw()
    
        # Get the RGBA buffer from the figure
        w, h = self.fig.canvas.get_width_height()
        buf = np.fromstring(self.fig.canvas.tostring_rgb(), dtype=np.uint8).reshape((h, w, 3))
    
        # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
        # buf = numpy.roll ( buf, 3, axis = 2 )
        return buf / 256#.transpose((2, 0, 1))


class LinePlot(Plot):
    COLORS = ['r', 'b', 'g', 'c', 'k', 'w', 'y']
    def __init__(self, xlim=[-1,1], ylim=[-1,1], num_scatters=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.subplot = self.parent._get_subplot()
        self.sc = [self.subplot.plot([], [], self.COLORS[i] + '-')[0] for i in range(num_scatters)]
        self.subplot.set_xlim(*xlim)
        self.subplot.set_ylim(*ylim)
        self.xlim = xlim
        self.ylim = ylim
        self._redraw()

    def add_point(self, y, x, line_num=0, redraw=True):
        xs = np.append(self.sc[line_num].get_xdata(), [x])
        ys = np.append(self.sc[line_num].get_ydata(), [y])
        self.sc[line_num].set_xdata(xs)
        self.sc[line_num].set_ydata(ys)
        self.xlim = [min(self.xlim[0], x), max(self.xlim[1], x)]
        self.ylim = [min(self.ylim[0], y), max(self.ylim[1], y)]
        self.subplot.set_xlim(*self.xlim)
        self.subplot.set_ylim(*self.ylim)
        if redraw:
            self._redraw()


class ScatterPlot(Plot):
    def __init__(self, value_range=[-1,1], xlim=[-1,1], ylim=[-1,1], palette='seismic', *args, **kwargs):
        super().__init__(*args, **kwargs)
        cmap = plt.get_cmap(palette)
        norm = matplotlib.colors.Normalize(*value_range)
        # FIXME: ignoring s=scale,
        self.subplot = self.parent._get_subplot()
        self.sc = self.subplot.scatter(x=[], y=[], c=[], norm=norm, cmap=cmap, alpha=0.8, edgecolors='none')
        self.subplot.set_xlim(*xlim)
        self.subplot.set_ylim(*ylim)
        self._redraw()

    def update(self, points, values):
        '''
            points: should have the shape (N*2) and consist of x,y coordinates
            values: should have the shape (N) and consist of the values at these coordinates (i.e. points)
        '''
        # self.sc.set_offsets(np.c_[x,y])
        self.sc.set_offsets(points)
        self.sc.set_array(values)
        self._redraw()


class SurfacePlot(Plot):
    def __init__(self, value_range=[-1,1], xlim=[-1,1], ylim=[-1,1], palette='seismic', *args, **kwargs):
        super().__init__(*args, **kwargs)
        # cmap = plt.get_cmap(palette)
        # norm = matplotlib.colors.Normalize(*value_range)
        # FIXME: ignoring s=scale,
        self.subplot = self.parent._get_subplot(projection='3d')
        self.subplot.set_xlabel('X')
        self.subplot.set_ylabel('Y')
        empty = np.zeros((0,0))
        self.sc = self.subplot.plot_surface(empty, empty, empty, cmap=cm.coolwarm, alpha=0.8, edgecolors='none')
        self.subplot.set_xlim(*xlim)
        self.subplot.set_ylim(*ylim)
        self._redraw()

    def update(self, X, Y, Z):
        '''
            points: should have the shape (N*2) and consist of x,y coordinates
            values: should have the shape (N) and consist of the values at these coordinates (i.e. points)
        '''
        # self.sc.set_offsets(np.c_[x,y])
        self.sc.remove()
        self.sc = self.subplot.plot_surface(X, Y, Z, cmap=cm.coolwarm, alpha=0.8)
        # self.sc.set_offsets(points)
        # self.sc.set_array(values)
        self._redraw()


class QuiverPlot(Plot):
    def __init__(self, xlim=[-1,1], ylim=[-1,1], palette='seismic', *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.subplot = self.parent._get_subplot()
        norm = np.linalg.norm([ xlim[1]-xlim[0], ylim[1]-ylim[0]])
        self.sc = self.subplot.quiver([], [], [], [], units='xy', pivot='middle', scale=40 / norm)
        self.subplot.set_xlim(*xlim)
        self.subplot.set_ylim(*ylim)
        self._redraw()

    def update(self, points, dirs):
        '''
            points: should have the shape (N*2) and consist of x,y coordinates
            values: should have the shape (N*2) and consist of the vector values in these coordinates (i.e. points)
        '''
        self.sc.set_offsets(points)
        U, V = np.array(dirs).transpose()
        M = np.hypot(U, V)
        self.sc.set_UVC(U/M, V/M, M)
        self._redraw()
        

if __name__ == '__main__':
    plot = Plot(1,2)
    sc = ScatterPlot(parent=plot)
    qv = QuiverPlot(parent=plot)
    n = 100
    for _ in range(100):
        x = np.random.rand(n) * 2 - 1
        y = np.random.rand(n) * 2 - 1
        xy = np.c_[x, y]
        c = np.random.rand(n)
        sc.update(xy, c)
        dirs = np.random.rand(2, n)
        qv.update(xy, dirs)
        time.sleep(0.2)
