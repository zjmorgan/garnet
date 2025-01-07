from garnet.reduction.plan import ReductionPlan, save_YAML

corelli = ReductionPlan()
corelli.generate_plan("CORELLI")
corelli.plan["IPTS"] = 31429
corelli.plan["Runs"] = "324246:324422"
corelli.plan[
    "UBFile"
] = "/SNS/CORELLI/IPTS-31429/shared/garnet/calibration/Yb3Al5O12_300K_2023_0703/YAG.mat"
corelli.plan[
    "DetectorCalibration"
] = "/SNS/CORELLI/shared/calibration/2022A/calibration.xml"
corelli.plan[
    "MaskFile"
] = "/SNS/CORELLI/shared/Vanadium/2022B_0725_CCR_5x7/integration.xml"
corelli.plan[
    "VanadiumFile"
] = "/SNS/CORELLI/shared/Vanadium/2022B_0725_CCR_5x7/solid_angle_2p5-8.nxs"
corelli.plan[
    "FluxFile"
] = "/SNS/CORELLI/shared/Vanadium/2022B_0725_CCR_5x7/flux_2p5-8.nxs"
corelli.plan[
    "BackgroundFile"
] = "/SNS/CORELLI/shared/Background/2022B_0725_CCR_5x7/background_2p5-8.nxs"
corelli.plan["Grouping"] = "1x4"
corelli.plan["Integration"]["Cell"] = "Cubic"
corelli.plan["Integration"]["Centering"] = "I"
corelli.plan["Integration"]["MinD"] = 0.7
corelli.plan["Integration"]["Radius"] = 0.25
corelli.plan["Normalization"]["Symmetry"] = "m-3m"

topaz = ReductionPlan()
topaz.generate_plan("TOPAZ")
topaz.plan["IPTS"] = 31189
topaz.plan["Runs"] = "46759:46769"
topaz.plan[
    "UBFile"
] = "/SNS/TOPAZ/IPTS-31189/shared/scolecite/find_peaks_46759-46769/SC_Monoclinic_C.mat"
topaz.plan[
    "DetectorCalibration"
] = "/SNS/TOPAZ/shared/calibration/2022C/TOPAZ_2022C_YAG.DetCal"
topaz.plan["MaskFile"] = "/SNS/TOPAZ/shared/Vanadium/2022C_1202_AG/mask.xml"
topaz.plan[
    "VanadiumFile"
] = "/SNS/TOPAZ/shared/Vanadium/2022C_1202_AG/solid_angle_1p8-18.nxs"
topaz.plan[
    "FluxFile"
] = "/SNS/TOPAZ/shared/Vanadium/2022C_1202_AG/flux_1p8-18.nxs"
topaz.plan[
    "BackgroundFile"
] = "/SNS/TOPAZ/shared/Background/2022C_1202_AG/background_1p8-18.nxs"
topaz.plan["Grouping"] = "2x2"
topaz.plan["Integration"]["Cell"] = "Monoclinic"
topaz.plan["Integration"]["Centering"] = "C"
topaz.plan["Integration"]["MinD"] = 0.6
topaz.plan["Integration"]["Radius"] = 0.25
topaz.plan["Normalization"]["Symmetry"] = "2/m"

mandi = ReductionPlan()
mandi.generate_plan("MANDI")
mandi.plan["IPTS"] = 31429
mandi.plan["Runs"] = "324246"
mandi.plan["UBFile"] = None
mandi.plan[
    "DetectorCalibration"
] = "/SNS/MANDI/shared/calibration/2022C/calibration.DetCal"
mandi.plan[
    "MaskFile"
] = "/SNS/CORELLI/shared/Vanadium/2022B_0725_CCR_5x7/integration.xml"
mandi.plan[
    "VanadiumFile"
] = "/SNS/CORELLI/shared/Vanadium/2022B_0725_CCR_5x7/solid_angle_2p5-8.nxs"
mandi.plan[
    "FluxFile"
] = "/SNS/CORELLI/shared/Vanadium/2022B_0725_CCR_5x7/flux_2p5-8.nxs"
mandi.plan[
    "BackgroundFile"
] = "/SNS/MANDI/shared/Background/2022C_1202/background_1p7-3p1.nxs"
mandi.plan["Grouping"] = "2x2"
mandi.plan["Integration"]["Cell"] = "Cubic"
mandi.plan["Integration"]["Centering"] = "I"
mandi.plan["Integration"]["MinD"] = 1.0
mandi.plan["Integration"]["Radius"] = 0.25
mandi.plan["Normalization"]["Symmetry"] = "m-3m"

snap = ReductionPlan()
snap.generate_plan("SNAP")
snap.plan["IPTS"] = 25836
snap.plan["Runs"] = "51455:51599"
snap.plan["UBFile"] = None
snap.plan["DetectorCalibration"] = None
snap.plan["MaskFile"] = None
snap.plan["VanadiumFile"] = ""
snap.plan["FluxFile"] = ""
snap.plan["Grouping"] = "4x4"
snap.plan["Integration"]["Cell"] = "Hexagonal"
snap.plan["Integration"]["Centering"] = "R(obv)"
snap.plan["Integration"]["MinD"] = 0.6
snap.plan["Integration"]["Radius"] = 0.25
snap.plan["Normalization"]["Symmetry"] = "-3m"

demand = ReductionPlan()
demand.generate_plan("DEMAND")
demand.plan["IPTS"] = 9884
demand.plan["Runs"] = "2"
demand.plan["Experiment"] = 817
demand.plan["UBFile"] = "/HFIR/HB3A/shared/benchmark/LiFePO4.mat"
demand.plan[
    "VanadiumFile"
] = "/HFIR/HB3A/shared/Vanadium/Vanadium_cycle493/HB3A_exp0814_scan0001.nxs"
demand.plan["Grouping"] = "4x4"
demand.plan["Integration"]["Cell"] = "Orthorhombic"
demand.plan["Integration"]["Centering"] = "P"
demand.plan["Integration"]["MinD"] = 0.7
demand.plan["Integration"]["Radius"] = 0.25
demand.plan["Normalization"]["Symmetry"] = "mmm"
demand.plan["Normalization"]["Extents"] = [[-5, 5], [-9, 9], [-5, 5]]
demand.plan["Normalization"]["Bins"] = [201, 361, 201]

wand2 = ReductionPlan()
wand2.generate_plan("WAND²")
wand2.plan["IPTS"] = 7776
wand2.plan["Runs"] = "26640:27944"
wand2.plan["UBFile"] = "/HFIR/HB2C/shared/benchmark/NaCl.mat"
wand2.plan["VanadiumFile"] = "/HFIR/HB2C/IPTS-7776/nexus/HB2C_27969.nxs.h5"
wand2.plan["BackgroundFile"] = "/HFIR/HB2C/IPTS-7776/nexus/HB2C_27968.nxs.h5"
wand2.plan["Grouping"] = "4x4"
wand2.plan["Integration"]["Cell"] = "Cubic"
wand2.plan["Integration"]["Centering"] = "F"
wand2.plan["Integration"]["MinD"] = 0.7
wand2.plan["Integration"]["Radius"] = 0.25
wand2.plan["Normalization"]["Symmetry"] = "m-3m"
wand2.plan["Normalization"]["Extents"] = [[-7, 7], [-7, 7], [-7, 7]]
wand2.plan["Normalization"]["Bins"] = [281, 281, 281]

plans = [corelli, topaz, mandi, snap, demand, wand2]
names = [k for k, v in locals().items() if v in plans]

for name, plan in zip(names, plans):
    save_YAML(plan.plan, "{}_reduction_plan.yaml".format(name))
