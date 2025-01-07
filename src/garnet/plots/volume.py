import numpy as np
import matplotlib.pyplot as plt

from matplotlib.transforms import Affine2D
# from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

import scipy.linalg

from garnet.plots.base import BasePlot

class SlicePlot(BasePlot):

    def __init__(self, UB, W):

        self.fig, self.ax = plt.subplots(1, 1)

        G = np.dot(UB.T, UB)

        B = scipy.linalg.cholesky(G, lower=False)
        Bp = np.dot(B, W)

        Q, R = scipy.linalg.qr(Bp)

        self.V = np.dot(R.T, R)

    def calculate_transforms(self, axes, labels, normal):

        ind = np.array(normal) != 1

        axes_ind = np.arange(3)[ind]
        slice_ind = np.arange(3)[~ind][0]

        x, y = [axes[i] for i in axes_ind]
        xlabel, ylabel = [labels[i] for i in axes_ind]

        self.z = axes[slice_ind]
        self.z_label = labels[slice_ind]
        self.slice_ind = slice_ind

        v = scipy.linalg.cholesky(self.V[ind][:,ind], lower=False)
        v /= v[0,0]

        T = np.eye(3)
        T[:2,:2] = v

        _, aspect, _ = np.diag(T).copy()
        T[1,1] = 1
        T[0,2] = -T[0,1]*y.min()

        transform = Affine2D(T)+self.ax.transData

        self.im = self.ax.pcolormesh(x,
                                     y,
                                     x+y[:,np.newaxis]+1,
                                     norm='log',
                                     shading='nearest',
                                     transform=transform,
                                     rasterized=True)

        self.ax.set_aspect(aspect)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.minorticks_on()

        self.ax.xaxis.get_major_locator().set_params(integer=True)
        self.ax.yaxis.get_major_locator().set_params(integer=True)

        self.cb = self.fig.colorbar(self.im, ax=self.ax)
        self.cb.minorticks_on()
        # self.cb.formatter.set_powerlimits((0, 0))
        # self.cb.formatter.set_useMathText(True)

    def make_slice(self, signal, value):

        i = np.argmin(np.abs(self.z-value))

        if self.slice_ind == 0:
            data = signal[i,:,:].T
        elif self.slice_ind == 1:
            data = signal[:,i,:].T
        else:
            data = signal[:,:,i].T

        self.ax.set_title(self.z_label+'$={:.3f}$'.format(self.z[i]))

        data[data <= 0] = np.nan
        data[~np.isfinite(data)] = np.nan

        vmin, vmax = np.nanmin(data), np.nanmax(data)

        if np.isclose(vmin, vmax):
            if np.isclose(vmax, 0):
                vmin, vmax = 0.01, 1.0
            else:
                vmin = vmax/100
        elif np.isnan(vmin) or np.isnan(vmax):
            vmin, vmax = 0.01, 1.0

        self.im.set_array(data.ravel())
        self.im.set_clim(vmin=vmin, vmax=vmax)

        self.cb.update_normal(self.im)
        self.cb.ax.minorticks_on()
        # self.cb.formatter.set_powerlimits((0, 0))
        # self.cb.formatter.set_useMathText(True)
