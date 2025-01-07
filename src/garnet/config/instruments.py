beamlines = {
    "SNAP": {
        "Facility": "SNS",
        "Name": "SNAP",
        "InstrumentName": "SNAP",
        "FancyName": "SNAP",
        "Wavelength": [1, 4],
        "Goniometer": {"BL3:Mot:omega": [0, 1, 0, 1]},
        "GoniometerEntry": "metadata.entry.daslogs",
        "Scale": "metadata.entry.proton_charge",
        "Title": "metadata.entry.title",
        "RunNumber": "metadata.entry.run_number",
        "Extension": ".nxs.h5",
        "DetectorCalibration": True,
        "TubeCalibration": False,
        "Groupings": ["1x1", "2x2", "4x4"],
        "RawFile": "nexus/SNAP_{}.nxs.h5",
        "BankPixels": [256, 256],
        "DeltaTheta": 0.0375,
    },
    "CORELLI": {
        "Facility": "SNS",
        "Name": "CORELLI",
        "InstrumentName": "CORELLI",
        "FancyName": "CORELLI",
        "Wavelength": [0.6, 2.5],
        "Goniometer": {
            "BL9:Mot:Sample:Axis1": [0, 1, 0, 1],
            "BL9:Mot:Sample:Axis2": [0, 1, 0, 1],
            "BL9:Mot:Sample:Axis3": [0, 1, 0, 1],
        },
        "GoniometerEntry": "metadata.entry.daslogs",
        "Scale": "metadata.entry.proton_charge",
        "Title": "metadata.entry.title",
        "RunNumber": "metadata.entry.run_number",
        "Extension": ".nxs.h5",
        "DetectorCalibration": True,
        "TubeCalibration": True,
        "Groupings": ["1x1", "1x2", "1x4"],
        "RawFile": "nexus/CORELLI_{}.nxs.h5",
        "BankPixels": [16, 256],
        "DeltaTheta": 0.16,
    },
    "TOPAZ": {
        "Facility": "SNS",
        "Name": "TOPAZ",
        "InstrumentName": "TOPAZ",
        "FancyName": "TOPAZ",
        "Wavelength": [0.4, 3.5],
        "Goniometer": {
            "BL12:Mot:omega": [0, 1, 0, 1],
            "BL12:Mot:chi": [0, 0, 1, 1],
            "BL12:Mot:phi": [0, 1, 0, 1],
        },
        "GoniometerEntry": "metadata.entry.daslogs",
        "Scale": "metadata.entry.proton_charge",
        "Title": "metadata.entry.title",
        "RunNumber": "metadata.entry.run_number",
        "Extension": ".nxs.h5",
        "DetectorCalibration": True,
        "TubeCalibration": False,
        "Groupings": ["1x1", "2x2", "4x4"],
        "RawFile": "nexus/TOPAZ_{}.nxs.h5",
        "BankPixels": [256, 256],
        "DeltaTheta": 0.044,
    },
    "MANDI": {
        "Facility": "SNS",
        "Name": "MANDI",
        "InstrumentName": "MANDI",
        "FancyName": "MANDI",
        "Wavelength": [2, 4],
        "Goniometer": {
            "BL11B:Mot:omega": [0, 1, 0, 1],
            "BL11B:Mot:chi": [0, 0, 1, 1],
            "BL11B:Mot:phi": [0, 1, 0, 1],
        },
        "GoniometerAxisNames": [
            "BL11B:Mot:omega",
            "BL11B:Mot:chi",
            "BL11B:Mot:phi",
        ],
        "GoniometerEntry": "metadata.entry.daslogs",
        "Scale": "metadata.entry.proton_charge",
        "Title": "metadata.entry.title",
        "RunNumber": "metadata.entry.run_number",
        "Extension": ".nxs.h5",
        "DetectorCalibration": True,
        "TubeCalibration": False,
        "Groupings": ["1x1", "2x2", "4x4"],
        "RawFile": "nexus/MANDI_{}.nxs.h5",
        "BankPixels": [256, 256],
        "DeltaTheta": 0.0420,
    },
    "WAND²": {
        "Facility": "HFIR",
        "Name": "HB2C",
        "InstrumentName": "WAND",
        "FancyName": "WAND²",
        "Wavelength": 1.486,
        "Goniometer": {
            "HB2C:Mot:sgl": [1, 0, 0, -1],
            "HB2C:Mot:sgu": [0, 0, 1, -1],
            "HB2C:Mot:s1": [0, 1, 0, 1],
        },
        "GoniometerAxisNames": [None, None, "s1"],
        "GoniometerEntry": "metadata.entry.daslogs",
        "Scale": "metadata.entry.duration",
        "Title": "metadata.entry.title",
        "RunNumber": "metadata.entry.run_number",
        "Extension": ".nxs.h5",
        "DetectorCalibration": False,
        "TubeCalibration": False,
        "Groupings": ["1x1", "2x2", "4x4"],
        "RawFile": "nexus/HB2C_{}.nxs.h5",
        "BankPixels": [480, 512],
        "DeltaTheta": 0.15625,
    },
    "DEMAND": {
        "Facility": "HFIR",
        "Name": "HB3A",
        "InstrumentName": "HB3A",
        "FancyName": "DEMAND",
        "Wavelength": 1.546,
        "Goniometer": {
            "omega": [0, 1, 0, -1],
            "chi": [0, 0, 1, -1],
            "phi": [0, 1, 0, -1],
        },
        "GoniometerEntry": "statistics",
        "Scale": "statistics.time",
        "Title": "metadata.scan_title",
        "RunNumber": "metadata.scan",
        "Extension": ".dat",
        "DetectorCalibration": False,
        "TubeCalibration": False,
        "Groupings": ["1x1"],
        "RawFile": "shared/autoreduce/HB3A_exp{:04}_scan{:04}.nxs",
        "BankPixels": [512, 512],
        "DeltaTheta": 0.0158,
    },
}
