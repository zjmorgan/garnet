from mantid.simpleapi import (HasUB,
                              mtd)

import numpy as np

class ReciprocalSpaceViewer():

    def __init__(self, peaks_ws):

        self.peaks_ws = mtd[peaks_ws]

    def get_peak_info(self):

        T = np.zeros((4,4))

        peak_dict = {}

        Qs, Is, pk_nos = [], [], []

        for j, peak in enumerate(self.peaks_ws):

            I = peak.getIntensity()

            shape = eval(peak.getPeakShape().toJSON())

            pk_no = peak.getPeakNumber()

            Q = peak.getQSampleFrame()

            radii = np.array([shape['radius0'],
                              shape['radius1'],
                              shape['radius2']])

            dir1 = np.array(shape['direction0'].split(' ')).astype(float)
            dir2 = np.array(shape['direction1'].split(' ')).astype(float)
            dir3 = np.array(shape['direction2'].split(' ')).astype(float)

            v = np.column_stack([dir1, dir2, dir3])

            P = np.dot(v, np.dot(np.diag(radii), v.T))

            T[:3,:3] = P
            T[:3,-1] = Q
            T[-1,-1] = 1
            
            Qs.append(Q)
            Is.append(I)
            pk_nos.append(pk_no)

        peak_dict['coordinates'] = Qs
        peak_dict['intensities'] = Is
        peak_dict['numbers'] = pk_nos

        return peak_dict

    def get_UB(self):

        if HasUB(Workspace=self.peaks_ws):
            return self.peaks_ws.sample().getOrientedLattice().getUB()
        else:
            return None