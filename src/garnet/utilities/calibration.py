# import os
# import sys

# directory = os.path.dirname(os.path.realpath(__file__))
# sys.path.append(directory)

# directory = os.path.abspath(os.path.join(directory, "../.."))
# sys.path.append(directory)

# from mantid.simpleapi import (
#     LoadNexus,
#     FilterPeaks,
#     SortPeaksWorkspace,
#     CombinePeaksWorkspaces,
#     StatisticsOfPeaksWorkspace,
#     CreatePeaksWorkspace,
#     SaveHKL,
#     SaveReflections,
#     SaveIsawUB,
#     LoadIsawUB,
#     LoadIsawSpectrum,
#     CloneWorkspace,
#     CopySample,
#     SetGoniometer,
#     mtd,
# )


# import numpy as np
# import matplotlib.pyplot as plt

# from matplotlib.backends.backend_pdf import PdfPages
# from matplotlib.ticker import FormatStrFormatter

# from mantid.kernel import V3D

# import argparse

# from garnet.config.instruments import beamlines
# from garnet.reduction.ub import UBModel, Optimization, lattice_group
# from garnet.reduction.peaks import PeaksModel, PeakModel, centering_reflection
# from garnet.reduction.data import DataModel

# def main():
#     parser = argparse.ArgumentParser(description="Corrections for integration")

#     parser.add_argument(
#         "filename",
#         type=str,
#         help="Peaks Workspace",
#     )

#     parser.add_argument(
#         "-w", "--wobble", action="store_true", help="Off-centering correction"
#     )

#     parser.add_argument(
#         "-f",
#         "--formula",
#         type=str,
#         default="Yb3-Al5-O12",
#         help="Chemical formula",
#     )

#     parser.add_argument(
#         "-z",
#         "--zparameter",
#         type=float,
#         default="8",
#         help="Number of formula units",
#     )

#     parser.add_argument(
#         "-g",
#         "--pointgroup",
#         type=str,
#         default=None,
#         help="Point group symmetry",
#     )

#     parser.add_argument(
#         "-u",
#         "--uvector",
#         nargs="+",
#         type=float,
#         default=[0, 0, 1],
#         help="Miller indices along beam",
#     )

#     parser.add_argument(
#         "-v",
#         "--vvector",
#         nargs="+",
#         type=float,
#         default=[1, 0, 0],
#         help="Miller indices in plane",
#     )

#     parser.add_argument(
#         "-s",
#         "--shape",
#         type=str,
#         default="sphere",
#         help="Sample shape sphere, cylinder, plate",
#     )

#     parser.add_argument(
#         "-p",
#         "--parameters",
#         nargs="+",
#         type=float,
#         default=[0],
#         help="Length (diameter), height, width in millimeters",
#     )

#     parser.add_argument(
#         "-c", "--scale", type=float, default=None, help="Scale factor"
#     )

#     args = parser.parse_args()

#     peaks = Peaks("peaks", args.filename, args.scale, args.pointgroup)


# # outdir = '/SNS/TOPAZ/IPTS-31856/shared/mantid_si_cal'
# # calibration_file = '/SNS/TOPAZ/IPTS-31856/shared/garnet_reduction/calibration_eng.DetCal'#'/SNS/TOPAZ/IPTS-31856/shared/garnet_reduction/cal_eng.xml'
# # calibration_file = '/SNS/TOPAZ/IPTS-31856/shared/calibration/TOPAZ_2023B_Eng.DetCal'#'/SNS/TOPAZ/IPTS-31856/shared/garnet_reduction/cal_eng.xml'

# class calibrate

#     U = mtd['calibration_ws'].sample().getOrientedLattice().setU(np.eye(3))

#     # CalculateUMatrix(PeaksWorkspace='calibration_ws',
#     #                  a=a,
#     #                  b=b,
#     #                  c=c,
#     #                  alpha=alpha,
#     #                  beta=beta,
#     #                  gamma=gamma)

#     # SaveNexus(InputWorkspace='calibration_ws',
#     #           Filename=os.path.join(outdir, 'calibration_eng.nxs'))

#     # LoadNexus(OutputWorkspace='calibration_ws',
#     #           Filename=os.path.join(outdir, 'calibration_eng.nxs'))

#     peaks_dict = {}

#     banks = [bank.replace('bank', '') for bank in mtd['calibration_ws'].column('BankName')]
#     Q = mtd['calibration_ws'].column('QSample')
#     ol = mtd['calibration_ws'].sample().getOrientedLattice()

#     for i, peak in enumerate(mtd['calibration_ws']):
#         hkl = peak.getIntHKL()
#         peak.setIntensity(0)
#         peak.setSigmaIntensity(0)
#         tt = peak.getScattering()
#         az = peak.getAzimuthal()
#         kf_x = np.sin(tt)*np.cos(az)
#         kf_y = np.sin(tt)*np.sin(az)
#         kf_z = np.cos(tt)
#         if hkl.norm() > 0:
#             key = banks[i]
#             if peaks_dict.get(key) is None:
#                 x, y, gamma_ang, nu_ang = [], [], [], []
#             else:
#                 x, y, gamma_ang, nu_ang = peaks_dict[key]
#             d = 2*np.pi/np.linalg.norm(Q[i])
#             d0 = ol.d(*hkl)
#             x.append(d)
#             y.append(d0)
#             gamma_ang.append(np.rad2deg(np.arctan2(kf_x, kf_z)))
#             nu_ang.append(np.rad2deg(np.arcsin(kf_y)))
#             peaks_dict[key] = x, y, gamma_ang, nu_ang

#     FilterPeaks(InputWorkspace='calibration_ws',
#                 OutputWorkspace='calibration_ws',
#                 Criterion='!=',
#                 BankName='None')

#     SCDCalibratePanels(PeakWorkspace='calibration_ws',
#                        RecalculateUB=False,
#                        Tolerance=0.1,
#                        a=a,
#                        b=b,
#                        c=c,
#                        alpha=alpha,
#                        beta=beta,
#                        gamma=gamma,
#                        OutputWorkspace='calibration_table',
#                        DetCalFilename=os.path.join(outdir, 'calibration_eng.DetCal'),
#                        CSVFilename=os.path.join(outdir, 'calibration_eng.csv'),
#                        XmlFilename=os.path.join(outdir, 'calibration_eng.xml'),
#                        CalibrateT0=False,
#                        SearchRadiusT0=10,
#                        CalibrateL1=True,
#                        SearchRadiusL1=0.2,
#                        CalibrateBanks=True,
#                        SearchRadiusTransBank=0.2,
#                        SearchRadiusRotXBank=5,
#                        SearchRadiusRotYBank=5,
#                        SearchRadiusRotZBank=5,
#                        VerboseOutput=True,
#                        SearchRadiusSamplePos=0.001,
#                        TuneSamplePosition=True,
#                        CalibrateSize=False,
#                        SearchRadiusSize=0.1,
#                        FixAspectRatio=True)

#     LoadEmptyInstrument(InstrumentName='TOPAZ',
#                         OutputWorkspace='TOPAZ')

#     LoadParameterFile(Workspace='TOPAZ',
#                       Filename=os.path.join(outdir, 'calibration_eng.xml'))

#     # ApplyInstrumentToPeaks(InputWorkspace='calibration_ws',
#     #                        InstrumentWorkspace='TOPAZ',
#     #                        OutputWorkspace='calibration_ws')

#     sample_pos = mtd['TOPAZ'].getInstrument().getComponentByName('sample-position').getPos()
#     for bank in np.unique(mtd['calibration_ws'].column('BankName')):
#         MoveInstrumentComponent(Workspace='TOPAZ',
#                                 ComponentName=bank,
#                                 X=-sample_pos[0], Y=-sample_pos[1], Z=-sample_pos[2],
#                                 RelativePosition=True)

#     MoveInstrumentComponent(Workspace='TOPAZ',
#                             ComponentName='sample-position',
#                             X=-sample_pos[0], Y=-sample_pos[1], Z=-sample_pos[2],
#                             RelativePosition=True)

#     MoveInstrumentComponent(Workspace='TOPAZ',
#                             ComponentName='moderator',
#                             X=0, Y=0, Z=-sample_pos[2],
#                             RelativePosition=True)

#     ApplyInstrumentToPeaks(InputWorkspace='calibration_ws',
#                            InstrumentWorkspace='TOPAZ',
#                            OutputWorkspace='calibration_ws')

#     SCDCalibratePanels(PeakWorkspace='calibration_ws',
#                        RecalculateUB=False,
#                        Tolerance=0.1,
#                        a=a,
#                        b=b,
#                        c=c,
#                        alpha=alpha,
#                        beta=beta,
#                        gamma=gamma,
#                        OutputWorkspace='calibration_table',
#                        DetCalFilename=os.path.join(outdir, 'calibration_eng.DetCal'),
#                        CSVFilename=os.path.join(outdir, 'calibration_eng.csv'),
#                        XmlFilename=os.path.join(outdir, 'calibration_eng.xml'),
#                        CalibrateT0=False,
#                        SearchRadiusT0=10,
#                        CalibrateL1=False,
#                        SearchRadiusL1=0.2,
#                        CalibrateBanks=False,
#                        SearchRadiusTransBank=0.2,
#                        SearchRadiusRotXBank=15,
#                        SearchRadiusRotYBank=15,
#                        SearchRadiusRotZBank=15,
#                        VerboseOutput=True,
#                        SearchRadiusSamplePos=0.001,
#                        TuneSamplePosition=False,
#                        CalibrateSize=False,
#                        SearchRadiusSize=0.1,
#                        FixAspectRatio=True)

#     cal_dict = {}

#     banks = [bank.replace('bank', '') for bank in mtd['calibration_ws'].column('BankName')]
#     Q = mtd['calibration_ws'].column('QSample')
#     ol = mtd['calibration_ws'].sample().getOrientedLattice()

#     for i, peak in enumerate(mtd['calibration_ws']):
#         hkl = peak.getIntHKL()
#         tt = peak.getScattering()
#         az = peak.getAzimuthal()
#         kf_x = np.sin(tt)*np.cos(az)
#         kf_y = np.sin(tt)*np.sin(az)
#         kf_z = np.cos(tt)
#         if hkl.norm() > 0:
#             key = banks[i]
#             if cal_dict.get(key) is None:
#                 x, y, gamma_ang, nu_ang = [], [], [], []
#             else:
#                 x, y, gamma_ang, nu_ang = cal_dict[key]
#             d = 2*np.pi/np.linalg.norm(Q[i])
#             d0 = ol.d(*hkl)
#             x.append(d)
#             y.append(d0)
#             gamma_ang.append(np.rad2deg(np.arctan2(kf_x, kf_z)))
#             nu_ang.append(np.rad2deg(np.arcsin(kf_y)))
#             cal_dict[key] = x, y, gamma_ang, nu_ang

#     with PdfPages(os.path.join(outdir, 'calibration_eng_{}.pdf'.format(iter))) as pdf:

#         for key in cal_dict.keys():

#             fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, layout='constrained')

#             x, y, gamma_ang, nu_ang = peaks_dict[key]

#             x = np.array(x)
#             y = np.array(y)

#             delta_0 = x/y-1
#             gamma_0 = np.array(gamma_ang)
#             nu_0 = np.array(nu_ang)

#             ax[0].plot(x, 100*delta_0, '.', color='C0')

#             x, y, gamma_ang, nu_ang = cal_dict[key]

#             x = np.array(x)
#             y = np.array(y)

#             delta_1 = x/y-1
#             gamma_1 = np.array(gamma_ang)
#             nu_1 = np.array(nu_ang)

#             ax[1].plot(x, 100*delta_1, '.', color='C1')

#             ax[0].set_title('Bank #{} original'.format(key))
#             ax[1].set_title('Bank #{} refined'.format(key))
#             ax[0].minorticks_on()
#             ax[1].minorticks_on()
#             ax[0].set_ylim(-2,2)
#             ax[1].set_ylim(-2,2)
#             ax[0].axhline(y=0, color='k', linestyle='--')
#             ax[1].axhline(y=0, color='k', linestyle='--')
#             ax[0].set_xlabel(r'$d$-spacing [$\AA$]')
#             ax[1].set_xlabel(r'$d$-spacing [$\AA$]')
#             ax[0].set_ylabel(r'$d/d_0-1$ [%]')
#             pdf.savefig(fig)
#             plt.close()

#             fmt_str_form = FormatStrFormatter(r"$%d^\circ$")

#             fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, layout='constrained')

#             im = ax[0].scatter(gamma_0, nu_0, c=100*delta_0, cmap='coolwarm', vmin=-0.5, vmax=0.5)
#             im = ax[1].scatter(gamma_1, nu_1, c=100*delta_1, cmap='coolwarm', vmin=-0.5, vmax=0.5)

#             ax[0].xaxis.set_major_formatter(fmt_str_form)
#             ax[0].yaxis.set_major_formatter(fmt_str_form)
#             ax[1].yaxis.set_major_formatter(fmt_str_form)
#             ax[1].xaxis.set_major_formatter(fmt_str_form)

#             ax[0].minorticks_on()
#             ax[0].minorticks_on()

#             ax[0].set_title('Bank #{} original'.format(key))
#             ax[1].set_title('Bank #{} refined'.format(key))
#             ax[0].set_xlabel(r'$\gamma$-horizontal')
#             ax[1].set_xlabel(r'$\gamma$-horizontal')
#             ax[0].set_ylabel(r'$\nu$-vertical')
#             ax[0].set_aspect(1)
#             ax[1].set_aspect(1)

#             cb = fig.colorbar(im, ax=ax, orientation='horizontal')
#             cb.ax.minorticks_on()
#             cb.ax.set_xlabel(r'$d/d_0-1$ [%]')

#             pdf.savefig(fig)
#             plt.close()

#         plt.close('all')

#     calibration_file = os.path.join(outdir, 'calibration_eng.DetCal')
