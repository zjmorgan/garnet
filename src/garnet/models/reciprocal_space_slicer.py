from mantid.simpleapi import (CreatePeaksWorkspace,
                              HasUB,
                              mtd)

import numpy as np
import scipy.linalg

class ReciprocalSpaceSlicerrModel():

    def __init__(self):

        self.UB = None

    def set_md_histo_workspace(self, md_histo_ws):

        self.md_histo_ws = mtd[md_histo_ws]

        self.set_UB()

    def set_UB(self):

        if HasUB(Workspace=self.md_histo_ws):

            self.UB = self.md_histo_ws.sample().getOrientedLattice().getUB().copy()

    def get_transform(self):

        if self.UB is not None:

            UB = self.md_histo_ws.sample().getOrientedLattice().getUB()

            t = UB.copy()
            t /= np.max(t, axis=1)

            return t

    def ab_star_axes(self):

        if self.UB is not None:

            return np.dot(self.UB, [0,0,1]), np.dot(self.UB, [1,0,0])

    def bc_star_axes(self):

        if self.UB is not None:

            return np.dot(self.UB, [1,0,0]), np.dot(self.UB, [0,1,0])

    def ca_star_axes(self):

        if self.UB is not None:

            return np.dot(self.UB, [0,1,0]), np.dot(self.UB, [0,0,1])

    def ab_axes(self):

        if self.UB is not None:

            return np.cross(*self.bc_star_axes()), \
                   np.cross(*self.ca_star_axes())

    def bc_axes(self):

        if self.UB is not None:

            return np.cross(*self.ca_star_axes()), \
                   np.cross(*self.ab_star_axes())

    def ca_axes(self):

        if self.UB is not None:

            return np.cross(*self.ab_star_axes()), \
                   np.cross(*self.bc_star_axes())

    def get_vector(self, axes_type, ind):

        if self.UB is not None:

            if axes_type == '[hkl]':
                matrix = self.UB
            else:
                matrix = np.cross(np.dot(self.UB, np.roll(np.eye(3),2,1)).T,
                                  np.dot(self.UB, np.roll(np.eye(3),1,1)).T).T

            vec = np.dot(matrix, ind)

            return vec