# This file is part of DmpBbo, a set of libraries and programs for the
# black-box optimization of dynamical movement primitives.
# Copyright (C) 2018 Freek Stulp
#
# DmpBbo is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# DmpBbo is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DmpBbo.  If not, see <http://www.gnu.org/licenses/>.
#
""" Module for the Trajectory class. """

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt


class Trajectory:
    def __init__(self, ts, ys, yds=None, ydds=None, misc=None):

        n_time_steps = ts.size
        if n_time_steps != ys.shape[0]:
            raise ValueError("ys.shape[0] must have size {n_time_steps}")
        _dt_mean = np.mean(np.diff(ts))

        if yds is None:
            yds = diffnc(ys, _dt_mean)
        else:
            if ys.shape != yds.shape:
                raise ValueError("yds must have same shape as ys {ys.shape}")

        if ydds is None:
            ydds = diffnc(yds, _dt_mean)
        else:
            if ys.shape != ydds.shape:
                raise ValueError("ydds must have same shape as ys {ys.shape}")

        if misc is not None:
            if n_time_steps != misc.shape[0]:
                raise ValueError("misc.shape[0] must have size {n_time_steps}")

        self._dim = 1
        if ys.ndim == 2:
            self._dim = ys.shape[1]

        self._ts = ts
        self._dt_mean = _dt_mean
        self._ys = ys
        self._yds = yds
        self._ydds = ydds
        self._misc = misc

    @classmethod
    def from_matrix(cls, matrix, n_dims_misc=0):

        (n_time_steps, n_cols) = matrix.shape
        n_dims = (n_cols - 1 - n_dims_misc) // 3

        ts = matrix[:, 0]
        ys = matrix[:, 1 : 1 * n_dims + 1]
        yds = matrix[:, 1 * n_dims + 1 : 2 * n_dims + 1]
        ydds = matrix[:, 2 * n_dims + 1 : 3 * n_dims + 1]
        misc = matrix[:, 3 * n_dims + 1 :]

        return cls(ts, ys, yds, ydds, misc)

    @property
    def ts(self):
        return self._ts

    @property
    def ys(self):
        return self._ys

    @property
    def yds(self):
        return self._yds

    @property
    def ydds(self):
        return self._ydds

    @property
    def misc(self):
        return self._misc

    @misc.setter
    def misc(self, new_misc):
        if new_misc.shape[0] != self.length:
            raise ValueError("new_misc.shape[0] must have size {self.length}")
        self._misc = new_misc

    @property
    def length(self):
        return self._ts.shape[0]

    @property
    def duration(self):
        return self._ts[-1] - self._ts[0]

    @property
    def dim(self):
        return self._dim

    @property
    def dim_misc(self):
        if self._misc is None:
            return 0
        else:
            return self._misc.shape[1]

    @property
    def y_init(self):
        return self._ys[0]

    @property
    def y_final(self):
        return self._ys[-1]

    def set_start_time_to_zero(self):
        self._ts = self._ts - self._ts[0]

    def get_range_per_dim(self):
        return self._ys.max(axis=0) - self._ys.min(axis=0)

    def crop(self, start, end, as_times=False):
        # Crop trajectory from 'start' to 'end'
        # if as_times is False, 'start' to 'end' are interpreted as indices
        # if as_times is True, 'start' to 'end' are interpreted as times

        # No need to crop empty trajectory
        if self._ts.size == 0:
            return

        if start >= end:
            raise ValueError("start >= end does not hold (not {start} >= {end})")

        if as_times:
            if start > self._ts[-1]:
                print(
                    "WARNING: Argument 'fro' out of range, because {fro} > {self._ts[-1]}. Not "
                    "cropping"
                )
                return
            if end < self._ts[0]:
                print(
                    "WARNING: Argument 'fro' out of range, because {to} < {self._ts[-1]}. Not "
                    "cropping"
                )
                return

            # Convert time 'fro' to index 'fro'
            if start <= self._ts[0]:
                # Time 'start' lies before first time in trajectory
                start = 0
            else:
                # Get first index when time is larger than 'start'
                start = np.argmax(self._ts >= start)

            if end >= self._ts[-1]:
                # Time 'end' is larger than the last time in the trajectory
                end = len(self._ts) - 1
            else:
                # Get first index when time is smaller than 'end'
                end = np.argmax(self._ts >= end)

        self._ts = self._ts[start:end]
        self._ys = self._ys[start:end, :]
        self._yds = self._yds[start:end, :]
        self._ydds = self._ydds[start:end, :]
        if self._misc is not None:
            self._misc = self._misc[start:end, :]

    @classmethod
    def from_polynomial(cls, ts, y_from, yd_from, ydd_from, y_to, yd_to, ydd_to):

        a0 = y_from
        a1 = yd_from
        a2 = ydd_from / 2

        a3 = -10 * y_from - 6 * yd_from - 2.5 * ydd_from + 10 * y_to - 4 * yd_to + 0.5 * ydd_to
        a4 = 15 * y_from + 8 * yd_from + 2 * ydd_from - 15 * y_to + 7 * yd_to - ydd_to
        a5 = -6 * y_from - 3 * yd_from - 0.5 * ydd_from + 6 * y_to - 3 * yd_to + 0.5 * ydd_to

        n_time_steps = ts.size
        n_dims = y_from.size

        ys = np.zeros([n_time_steps, n_dims])
        yds = np.zeros([n_time_steps, n_dims])
        ydds = np.zeros([n_time_steps, n_dims])

        for i in range(n_time_steps):
            t = (ts[i] - ts[0]) / (ts[n_time_steps - 1] - ts[0])
            ys[i, :] = (
                a0 + a1 * t + a2 * pow(t, 2) + a3 * pow(t, 3) + a4 * pow(t, 4) + a5 * pow(t, 5)
            )
            yds[i, :] = (
                a1 + 2 * a2 * t + 3 * a3 * pow(t, 2) + 4 * a4 * pow(t, 3) + 5 * a5 * pow(t, 4)
            )
            ydds[i, :] = 2 * a2 + 6 * a3 * t + 12 * a4 * pow(t, 2) + 20 * a5 * pow(t, 3)

        yds /= ts[n_time_steps - 1] - ts[0]
        ydds /= pow(ts[n_time_steps - 1] - ts[0], 2)

        return cls(ts, ys, yds, ydds)

    @classmethod
    def from_viapoint_polynomial(cls, ts, y_from, y_yd_ydd_viapoint, viapoint_time, y_to):

        n_time_steps = ts.size
        n_dims = y_from.size

        viapoint_time_step = 0
        while viapoint_time_step < n_time_steps and ts[viapoint_time_step] < viapoint_time:
            viapoint_time_step += 1

        yd_from = np.zeros(n_dims)
        ydd_from = np.zeros(n_dims)

        y_viapoint = y_yd_ydd_viapoint[0 * n_dims : 1 * n_dims]
        yd_viapoint = y_yd_ydd_viapoint[1 * n_dims : 2 * n_dims]
        ydd_viapoint = y_yd_ydd_viapoint[2 * n_dims : 3 * n_dims]

        yd_to = np.zeros(n_dims)
        ydd_to = np.zeros(n_dims)

        traj1 = Trajectory.from_polynomial(
            ts[:viapoint_time_step],
            y_from,
            yd_from,
            ydd_from,
            y_viapoint,
            yd_viapoint,
            ydd_viapoint,
        )

        traj2 = Trajectory.from_polynomial(
            ts[viapoint_time_step:], y_viapoint, yd_viapoint, ydd_viapoint, y_to, yd_to, ydd_to
        )

        traj1.append(traj2)

        return traj1

    @classmethod
    def from_min_jerk(cls, ts, y_from, y_to):
        n_time_steps = ts.size
        n_dims = y_from.size

        ys = np.zeros([n_time_steps, n_dims])
        yds = np.zeros([n_time_steps, n_dims])
        ydds = np.zeros([n_time_steps, n_dims])

        D = ts[n_time_steps - 1]  # noqa
        tss = ts / D

        A = y_to - y_from  # noqa

        for i_dim in range(n_dims):
            # http://noisyaccumulation.blogspot.fr/2012/02/how-to-decompose-2d-trajectory-data.html

            ys[:, i_dim] = y_from[i_dim] + A[i_dim] * (
                6 * (tss ** 5) - 15 * (tss ** 4) + 10 * (tss ** 3)
            )

            yds[:, i_dim] = (A[i_dim] / D) * (30 * (tss ** 4) - 60 * (tss ** 3) + 30 * (tss ** 2))

            ydds[:, i_dim] = (A[i_dim] / (D * D)) * (120 * (tss ** 3) - 180 * (tss ** 2) + 60 * tss)

        return cls(ts, ys, yds, ydds)

    def append(self, trajectory):
        self._ts = np.concatenate((self._ts, trajectory.ts))
        self._ys = np.concatenate((self._ys, trajectory.ys))
        self._yds = np.concatenate((self._yds, trajectory.yds))
        self._ydds = np.concatenate((self._ydds, trajectory.ydds))
        if self._misc is None or trajectory.misc is None:
            self._misc = None
        else:
            self._misc = np.concatenate((self._misc, trajectory.misc))

    def as_matrix(self):
        as_matrix = np.column_stack((self._ts, self._ys, self._yds, self._ydds))
        if self._misc is not None:
            np.column_stack((as_matrix, self._misc))
        return as_matrix

    def savetxt(self, filename):
        np.savetxt(filename, self.as_matrix(), fmt="%1.7f")

    @staticmethod
    def loadtxt(filename, n_dims_misc=0):
        data = np.loadtxt(filename)

        (n_time_steps, n_cols) = data.shape
        n_dims = (n_cols - 1 - n_dims_misc) // 3

        ts = data[:, 0]
        ys = data[:, 1 : 1 * n_dims + 1]
        yds = data[:, 1 * n_dims + 1 : 2 * n_dims + 1]
        ydds = data[:, 2 * n_dims + 1 : 3 * n_dims + 1]
        misc = data[:, 3 * n_dims + 1 :]

        return Trajectory(ts, ys, yds, ydds, misc)

    def recompute_derivatives(self):
        self._yds = diffnc(self._ys, self._dt_mean)
        self._ydds = diffnc(self._yds, self._dt_mean)

    def apply_low_pass_filter(self, cutoff, order=3):
        # Sample rate and desired cutoff frequencies (in Hz).
        _dt_mean = np.mean(np.diff(self._ts))
        sample_freq = 1.0 / _dt_mean
        self._ys = butter_low_pass_filter(self._ys, cutoff, sample_freq, order)
        self.recompute_derivatives()

    def plot(self, axs=None):
        if not axs:
            fig = plt.figure(figsize=(15, 4))
            axs = [fig.add_subplot(1, 3, i + 1) for i in range(3)]

        """Plot a trajectory"""
        all_handles = axs[0].plot(self._ts, self._ys, "-")
        axs[0].set_xlabel("time (s)")
        axs[0].set_ylabel("y")
        if len(axs) > 1:
            h = axs[1].plot(self._ts, self._yds, "-")
            all_handles.extend(h)
            axs[1].set_xlabel("time (s)")
            axs[1].set_ylabel("yd")
        if len(axs) > 2:
            h = axs[2].plot(self._ts, self._ydds, "-")
            all_handles.extend(h)
            axs[2].set_xlabel("time (s)")
            axs[2].set_ylabel("ydd")

        if self._misc and len(axs) > 3:
            h = axs[3].plot(self._ts, self._misc, "-")
            all_handles.extend(h)
            axs[3].set_xlabel("time (s)")
            axs[3].set_ylabel("misc")

        x_lim = [min(self._ts), max(self._ts)]
        for ax in axs:
            ax.set_xlim(x_lim[0], x_lim[1])

        return all_handles, axs


def diffnc(xs, dt):
    """ Do non-causal differentiation with time interval dt between data points.

    The returned vector (matrix) is of the same length as the original one.

    Stefan Schaal December 29, 1995. Converted to Python by Freek Stulp
    """

    (n_samples, n_dims) = xs.shape
    fil = np.array([1.0, 0.0, -1.0]) / 2 / dt
    xs2 = np.empty([n_samples + 2, n_dims])
    for i_dim in range(n_dims):
        xs2[:, i_dim] = np.convolve(xs[:, i_dim], fil)

    xs = xs2[1:-1, :]
    xs[0, :] = xs[1, :]
    xs[-1, :] = xs[-2, :]
    return xs


def butter_low_pass(cutoff, fs, order=3):
    # http://scipy.github.io/old-wiki/pages/Cookbook/ButterworthBandpass
    nyq = 0.5 * fs
    cut = cutoff / nyq
    b, a = butter(order, cut, btype="low", analog=False, output="ba")  # noqa 'ba' => b, a
    return b, a


def butter_low_pass_filter(data, cutoff, fs, order=3):
    # http://scipy.github.io/old-wiki/pages/Cookbook/ButterworthBandpass
    b, a = butter_low_pass(cutoff, fs, order=order)
    y = filtfilt(b, a, data, axis=0)
    return y
