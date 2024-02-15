from mantid.simpleapi import (SetGoniometer,
                              FindPeaksMD,
                              PredictPeaks,
                              PredictSatellitePeaks,
                              CentroidPeaksMD,
                              IntegratePeaksMD,
                              AddPeakHKL,
                              PeakIntensityVsRadius,
                              CombinePeaksWorkspace,
                              FilterPeaks,
                              SortPeaksWorkspace,
                              DeleteWorkspace,
                              mtd)

import numpy as np

refl_cond_dict = {'P': 'Primitive', 
                  'I': 'Body centred',
                  'F': 'All-face centred',
                  'R': 'Rhombohderally centred, obverse',
                  'R(obv)': 'Rhombohderally centred, obverse',
                  'R(rev)': 'Rhombohderally centred, reverse',
                  'A': 'A-face centred',
                  'B': 'B-face centred',
                  'C': 'C-face centred'}

class PeakTools():

    def __init__(self):

        self.gon_axis = [None]*6

        self.edge_pixels = 0

    def initialize_instrument(self, inst_params):

        wl = inst_params['Wavelength']

        self.laue = True if type(wl) is list else False

        if self.laue:
            self.wl_min, self.wl_max = wl
        else:
            self.wl_min, self.wl_max = wl*0.99, wl*1.01

        gon_axis_names = inst_params.get('GoniometerAxisNames')
        gon = inst_params.get('Goniometer')

        for i, name in enumerate(gon.keys()):

            axis = gon[name]

            if gon_axis_names is not None:
                name = gon_axis_names[i]

            self.gon_axis[i] = '{},{},{},{},{},{}'.format(name,*axis)

    def set_goniometer(self, ws):

        SetGoniometer(Workspace=ws,
                      Goniometers='None, Specify Individually',
                      Axis0=self.gon_axis[0],
                      Axis1=self.gon_axis[1],
                      Axis2=self.gon_axis[2],
                      Axis3=self.gon_axis[3],
                      Axis4=self.gon_axis[4],
                      Axis5=self.gon_axis[5],
                      Average=self.laue)

    def find_peaks(self, mde_ws, 
                         peaks_ws,
                         max_d=15,
                         density=1000,
                         max_peaks=50):

        FindPeaksMD(InputWorkspace=mde_ws,
                    PeakDistanceTreshhold=2*np.pi/max_d,
                    MaxPeaks=max_peaks,
                    PeakFindingStrategy='VolumeNormalization',
                    DensityThresholdFactor=density,
                    EdgePixels=self.edge_pixels,
                    OutputWorkspace=peaks_ws)

    def centroid_peaks(self, mde_ws, peaks_ws, peak_radius):

        CentroidPeaksMD(InputWorkspace=mde_ws,
                        PeakRadius=peak_radius,
                        PeaksWorkspace=peaks_ws,
                        OutputWorkspace=peaks_ws)

    def integrate_peaks(self, mde_ws,
                              peaks_ws,
                              peak_radius,
                              background_inner_fact=0,
                              background_outer_fact=0,
                              method='sphere'):

        background_inner_radius = peak_radius*background_inner_fact
        background_outer_radius = peak_radius*background_outer_fact

        IntegratePeaksMD(InputWorkspace=mde_ws,
                         PeaksWorkspace=peaks_ws,
                         PeakRadius=peak_radius,
                         BackgroundInnerRadius=background_inner_radius,
                         BackgroundOuterRadius=background_outer_radius,
                         Ellipsoid=True if method == 'ellipsoid' else False,
                         FixQAxis=True,
                         FixMajorAxisLength=False,
                         UseCentroid=True,
                         MaxIterations=3,
                         RelaceIntensity=True,
                         IntegrateIfOnEdge=True,
                         AdaptiveQBackground=False,
                         MaskEdgeTubes=False,
                         OutputWorkspace=peaks_ws)

    def intensity_vs_radius(self, mde_ws,
                                  peaks_ws,
                                  peak_radius,
                                  background_inner_fact=0,
                                  background_outer_fact=0,
                                  steps=51):

        PeakIntensityVsRadius(InputWorkspace=mde_ws,
                              PeaksWorkspace=peaks_ws,
                              RadiusStart=0.0, 
                              RadiusEnd=peak_radius,
                              NumSteps=steps,
                              BackgroundInnerFactor=background_inner_fact,
                              BackgroundOuterFactor=background_outer_fact,
                              OutputWorkspace='peak_vs_rad')

    def get_max_d_spacing(self, ws):

        ol = mtd[ws].sample().getOrientedLattice()       

        return max([ol.a(), ol.b(), ol.c()])

    def predict_peaks(self, ws, refl_cond, d_min):

        d_max = self.get_max_d_spacing(ws)

        PredictPeaks(InputWorkspace=ws,
                     WavelengthMin=self.wl_min,
                     WavelengthMax=self.wl_max,
                     MinDSpacing=d_min,
                     MaxDSpacing=d_max,
                     ReflectionCondition=refl_cond_dict[refl_cond],
                     RoundHKL=True,
                     EdgePixels=self.edge_pixels,
                     OutputWorkspace='predict')

    def predict_satellite_peaks(self, peaks_ws,
                                      d_min,
                                      mod_vec_1=[0,0,0],
                                      mod_vec_2=[0,0,0],
                                      mod_vec_3=[0,0,0],
                                      max_order=0,
                                      cross_terms=False):

        d_max = self.get_max_d_spacing(peaks_ws)

        sat_peaks_ws = peaks_ws+'_sat'

        PredictSatellitePeaks(Peaks=peaks_ws,
                              SatellitePeaks=sat_peaks_ws,
                              ModVector1=mod_vec_1,
                              ModVector2=mod_vec_2,
                              ModVector3=mod_vec_3,
                              MaxOrder=max_order,
                              CrossTerms=cross_terms,
                              SaveModulationInfo=True,
                              IncludeIntegerHKL=False,
                              WavelengthMin=self.wl_min,
                              WavelengthMax=self.wl_max,
                              MinDSpacing=d_min,
                              MaxDSpacing=d_max)

        CombinePeaksWorkspace(LHSWorkspace=peaks_ws,
                              RHSWorkspace=sat_peaks_ws,
                              OutputWorkspace=peaks_ws)

        DeleteWorkspace(Workspace=sat_peaks_ws)

    def sort_peaks_by_d_hkl(self, peaks_ws):

        columns = ['l', 'k', 'h', 'DSpacing']

        for col in columns:

            SortPeaksWorkspace(InputWorkpace=peaks_ws,
                               ColumnNameToSortBy=col,
                               SortAscending=False,
                               OutputWorkspace=peaks_ws)

