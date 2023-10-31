from mantid.simpleapi import (SetGoniometer,
                              FindPeaksMD,
                              PredictPeaksMD,
                              PredictSatellitePeaks,
                              CentroidPeaksMD,
                              IntegratePeaksMD,
                              IndexPeaks,
                              mtd)

import numpy as np

class Instrument():

    def __init__(self):

        self.gon_axis0 = None
        self.gon_axis1 = None
        self.gon_axis2 = None
        self.gon_axis3 = None
        self.gon_axis4 = None
        self.gon_axis5 = None

        self.rotating = None
        self.inner_gon = False
        self.right = False

        self.peak_radius = 0.1

        self.edge_pixels = 0

    def set_goniometer(self, ws):

        SetGoniometer(Workspace=ws,
                      Goniometers='None, Specify Individually',
                      Axis0=self.gon_axis0,
                      Axis1=self.gon_axis1,
                      Axis2=self.gon_axis2,
                      Axis3=self.gon_axis3,
                      Axis4=self.gon_axis4,
                      Axis5=self.gon_axis5,
                      Average=self.gon_ave)

    def find_peaks(self, mde_ws, density=1000, max_peaks=50):

        FindPeaksMD(InputWorkspace=mde_ws,
                    PeakDistanceTreshhold=0.1,
                    MaxPeaks=max_peaks,
                    PeakFindingStrategy='VolumeNormalization',
                    DensityThresholdFactor=density,
                    CalculateGoniometerForCW=self.rotating,
                    InnerGoniometer=self.inner_gon,
                    FlipX=self.right,
                    EdgePixels=self.edge_pixels,
                    OutputWorkspace='found')

    def centroid_peaks(self, mde_ws, peaks_ws):

        CentroidPeaksMD(InputWorkspace=mde_ws,
                        PeakRadius=self.peak_radius,
                        PeaksWorkspace=peaks_ws,
                        OutputWorkspace='centered')

    def integrate_peaks(self, mde_ws, peaks_ws):

        IntegratePeaksMD(InputWorkspace=mde_ws,
                         PeaksWorkspace=peaks_ws,
                         PeakRadius=self.peak_radius,
                         BackgroundOuterRadius=self.peak_radius*1.1,
                         BackgroundInnerRadius=self.peak_radius*1.2,
                         OutputWorkspace=peaks_ws)

    def predict_peaks(self, ws, refl_cond, d_min):

        ol = mtd[ws].sample().getOrientedLattice()       

        d_max = np.max([ol.a(), ol.b(), ol.c()])

        PredictPeaksMD(InputWorkspace=ws,
                       WavelengthMin=self.wl_min,
                       WavelengthMax=self.wl_max,
                       MinDSpacing=self.d_min,
                       MaxDSpacing=d_max,
                       CalculateGoniometerForCW=self.rotating,
                       InnerGoniometer=self.inner_gon,
                       FlipX=self.right,
                       EdgePixels=self.edge_pixels,
                       ReflectionCondition=refl_cond,
                       RoundHKL=True,
                       OutputWorkspace='predict')

    def predict_satellite_peaks(self, peaks_ws,
                                      sat_peaks_ws,
                                      mod_vec_1=[0,0,0],
                                      mod_vec_2=[0,0,0],
                                      mod_vec_3=[0,0,0],
                                      max_order=0,
                                      cross_terms=False):

        PredictSatellitePeaks(Peaks=peaks_ws,
                              SatellitePeaks=sat_peaks_ws,
                              ModVector1=mod_vec_1,
                              ModVector2=mod_vec_2,
                              ModVector3=mod_vec_3,
                              MaxOrder=max_order,
                              CrossTerms=cross_terms,
                              SaveModulationInfo=True)

    def index_peaks(self, peaks_ws,
                          tol=0.1,
                          sat_tol=None,
                          mod_vec_1=[0,0,0],
                          mod_vec_2=[0,0,0],
                          mod_vec_3=[0,0,0],
                          max_order=0,
                          cross_terms=False):

        tol_for_sat = sat_tol if sat_tol is not None else tol

        indexing = IndexPeaks(PeaksWorkspace=peaks_ws,
                              Tolerance=tol,
                              ToleranceForSatellite=tol_for_sat,
                              RoundHKLs=True,
                              CommonUBForAll=True,
                              ModVector1=mod_vec_1,
                              ModVector2=mod_vec_2,
                              ModVector3=mod_vec_3,
                              MaxOrder=max_order,
                              CrossTerms=cross_terms,
                              SaveModulationInfo=True)

        return indexing

class RotatingCrystalInstrument(Instrument):

    def __init__(self):

        self.rotating = True

class LaueInstrument(Instrument):

    def __init__(self):

        self.rotating = False