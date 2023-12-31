isotopes = {'H': [1, 2, 3],
            'He': [3, 4],
            'Li': [6, 7],
            'Be': [],
            'B': [10, 11],
            'C': [12, 13],
            'N': [14, 15],
            'O': [16, 17, 18],
            'F': [],
            'Ne': [20, 21, 22],
            'Na': [],
            'Mg': [24, 25, 26],
            'Al': [],
            'Si': [28, 29, 30],
            'P': [],
            'S': [32, 33, 34, 36],
            'Cl': [35, 37],
            'Ar': [36, 38, 40],
            'K': [39, 40, 41],
            'Ca': [40, 42, 43, 44, 46, 48],
            'Sc': [],
            'Ti': [46, 47, 48, 49, 50],
            'V': [50, 51],
            'Cr': [50, 52, 53, 54],
            'Mn': [],
            'Fe': [54, 56, 57, 58],
            'Co': [],
            'Ni': [58, 60, 61, 62, 64],
            'Cu': [63, 65],
            'Zn': [64, 66, 67, 68, 70],
            'Ga': [69, 71],
            'Ge': [70, 72, 73, 74, 76],
            'As': [],
            'Se': [74, 76, 77, 78, 80, 82],
            'Br': [79, 81],
            'Kr': [78, 80, 82, 83, 84, 86],
            'Rb': [85, 87],
            'Sr': [84, 86, 87, 88],
            'Y': [],
            'Zr': [90, 91, 92, 94, 96],
            'Nb': [],
            'Mo': [92, 94, 95, 96, 97, 98, 100],
            'Tc': [],
            'Ru': [96, 98, 99, 100, 101, 102, 104],
            'Rh': [],
            'Pd': [102, 104, 105, 106, 108, 110],
            'Ag': [107, 109],
            'Cd': [106, 108, 110, 111, 112, 113, 114, 116],
            'In': [113, 115],
            'Sn': [112, 114, 115, 116, 117, 118, 119, 120, 122, 124],
            'Sb': [121, 123],
            'Te': [120, 122, 123, 124, 125, 126, 128, 130],
            'I': [],
            'Xe': [124, 126, 128, 129, 130, 131, 132, 134, 136],
            'Cs': [],
            'Ba': [130, 132, 134, 135, 136, 137, 138],
            'La': [138, 139],
            'Ce': [136, 138, 140, 142],
            'Pr': [],
            'Nd': [142, 143, 144, 145, 146, 148, 150],
            'Pm': [],
            'Sm': [144, 147, 148, 149, 150, 152, 154],
            'Eu': [151, 153],
            'Gd': [152, 154, 155, 156, 157, 158, 160],
            'Tb': [],
            'Dy': [156, 158, 160, 161, 162, 163, 164],
            'Ho': [],
            'Er': [162, 164, 166, 167, 168, 170],
            'Tm': [],
            'Yb': [168, 170, 171, 172, 173, 174, 176],
            'Lu': [175, 176],
            'Hf': [174, 176, 177, 178, 179, 180],
            'Ta': [180, 181],
            'W': [180, 182, 183, 184, 186],
            'Re': [185, 187],
            'Os': [184, 186, 187, 188, 189, 190, 192],
            'Ir': [191, 193],
            'Pt': [190, 192, 194, 195, 196, 198],
            'Au': [],
            'Hg': [196, 198, 199, 200, 201, 202, 204],
            'Tl': [203, 205],
            'Pb': [204, 206, 207, 208],
            'Bi': [],
            'Po': [],
            'At': [],
            'Rn': [],
            'Fr': [],
            'Ra': [],
            'Ac': [],
            'Th': [],
            'Pa': [],
            'U': [233, 234, 235, 238],
            'Np': [],
            'Pu': [238, 239, 240, 242],
            'Am': [],
            'Cm': [244, 246, 248]}

colors = {'H': (1.0, 1.0, 1.0),
          'D': (0.8, 0.8, 1.0),
          'He': (0.851, 1.0, 1.0),
          'Li': (0.8, 0.502, 1.0),
          'Be': (0.761, 1.0, 0.0),
          'B': (1.0, 0.71, 0.71),
          'C': (0.565, 0.565, 0.565),
          'N': (0.188, 0.314, 0.973),
          'O': (1.0, 0.051, 0.051),
          'F': (0.565, 0.878, 0.314),
          'Ne': (0.702, 0.89, 0.961),
          'Na': (0.671, 0.361, 0.949),
          'Mg': (0.541, 1.0, 0.0),
          'Al': (0.749, 0.651, 0.651),
          'Si': (0.941, 0.784, 0.627),
          'P': (1.0, 0.502, 0.0),
          'S': (1.0, 1.0, 0.188),
          'Cl': (0.122, 0.941, 0.122),
          'Ar': (0.502, 0.82, 0.89),
          'K': (0.561, 0.251, 0.831),
          'Ca': (0.239, 1.0, 0.0),
          'Sc': (0.902, 0.902, 0.902),
          'Ti': (0.749, 0.761, 0.78),
          'V': (0.651, 0.651, 0.671),
          'Cr': (0.541, 0.6, 0.78),
          'Mn': (0.612, 0.478, 0.78),
          'Fe': (0.878, 0.4, 0.2),
          'Co': (0.941, 0.565, 0.627),
          'Ni': (0.314, 0.816, 0.314),
          'Cu': (0.784, 0.502, 0.2),
          'Zn': (0.49, 0.502, 0.69),
          'Ga': (0.761, 0.561, 0.561),
          'Ge': (0.4, 0.561, 0.561),
          'As': (0.741, 0.502, 0.89),
          'Se': (1.0, 0.631, 0.0),
          'Br': (0.651, 0.161, 0.161),
          'Kr': (0.361, 0.722, 0.82),
          'Rb': (0.439, 0.18, 0.69),
          'Sr': (0.0, 1.0, 0.0),
          'Y': (0.58, 1.0, 1.0),
          'Zr': (0.58, 0.878, 0.878),
          'Nb': (0.451, 0.761, 0.788),
          'Mo': (0.329, 0.71, 0.71),
          'Tc': (0.231, 0.62, 0.62),
          'Ru': (0.141, 0.561, 0.561),
          'Rh': (0.039, 0.49, 0.549),
          'Pd': (0.0, 0.412, 0.522),
          'Ag': (0.753, 0.753, 0.753),
          'Cd': (1.0, 0.851, 0.561),
          'In': (0.651, 0.459, 0.451),
          'Sn': (0.4, 0.502, 0.502),
          'Sb': (0.62, 0.388, 0.71),
          'Te': (0.831, 0.478, 0.0),
          'I': (0.58, 0.0, 0.58),
          'Xe': (0.259, 0.62, 0.69),
          'Cs': (0.341, 0.09, 0.561),
          'Ba': (0.0, 0.788, 0.0),
          'La': (0.439, 0.831, 1.0),
          'Ce': (1.0, 1.0, 0.78),
          'Pr': (0.851, 1.0, 0.78),
          'Nd': (0.78, 1.0, 0.78),
          'Pm': (0.639, 1.0, 0.78),
          'Sm': (0.561, 1.0, 0.78),
          'Eu': (0.38, 1.0, 0.78),
          'Gd': (0.271, 1.0, 0.78),
          'Tb': (0.188, 1.0, 0.78),
          'Dy': (0.122, 1.0, 0.78),
          'Ho': (0.0, 1.0, 0.612),
          'Er': (0.0, 0.902, 0.459),
          'Tm': (0.0, 0.831, 0.322),
          'Yb': (0.0, 0.749, 0.22),
          'Lu': (0.0, 0.671, 0.141),
          'Hf': (0.302, 0.761, 1.0),
          'Ta': (0.302, 0.651, 1.0),
          'W': (0.129, 0.58, 0.839),
          'Re': (0.149, 0.49, 0.671),
          'Os': (0.149, 0.4, 0.588),
          'Ir': (0.09, 0.329, 0.529),
          'Pt': (0.816, 0.816, 0.878),
          'Au': (1.0, 0.82, 0.137),
          'Hg': (0.722, 0.722, 0.816),
          'Tl': (0.651, 0.329, 0.302),
          'Pb': (0.341, 0.349, 0.38),
          'Bi': (0.62, 0.31, 0.71),
          'Po': (0.671, 0.361, 0.0),
          'At': (0.459, 0.31, 0.271),
          'Rn': (0.259, 0.51, 0.588),
          'Fr': (0.259, 0.0, 0.4),
          'Ra': (0.0, 0.49, 0.0),
          'Ac': (0.439, 0.671, 0.98),
          'Th': (0.0, 0.729, 1.0),
          'Pa': (0.0, 0.631, 1.0),
          'U': (0.0, 0.561, 1.0),
          'Np': (0.0, 0.502, 1.0),
          'Pu': (0.0, 0.42, 1.0),
          'Am': (0.329, 0.361, 0.949),
          'XX': (0.471, 0.361, 0.89)}

radii = {'H': (0.25, 0.53, 1.15),
         'D': (0.25, 0.53, 1.15),
         'He': (1.2, 0.31, 1.4),
         'Li': (1.45, 1.67, 1.815),
         'Be': (1.05, 1.12, 1.53),
         'B': (0.85, 0.87, 1.92),
         'C': (0.7, 0.67, 1.7),
         'N': (0.65, 0.56, 1.55),
         'O': (0.6, 0.48, 1.52),
         'F': (0.5, 0.42, 1.47),
         'Ne': (1.6, 0.38, 1.54),
         'Na': (1.8, 1.9, 2.27),
         'Mg': (1.5, 1.45, 1.73),
         'Al': (1.25, 1.18, 1.84),
         'Si': (1.1, 1.11, 2.1),
         'P': (1.0, 0.98, 1.8),
         'S': (1.0, 0.88, 1.8),
         'Cl': (1.0, 0.79, 1.75),
         'Ar': (0.71, 0.71, 1.88),
         'K': (2.2, 2.43, 2.75),
         'Ca': (1.8, 1.94, 2.31),
         'Sc': (1.6, 1.84, 2.11),
         'Ti': (1.4, 1.76, 0.0),
         'V': (1.35, 1.71, 0.0),
         'Cr': (1.4, 1.66, 0.0),
         'Mn': (1.4, 1.61, 0.0),
         'Fe': (1.4, 1.56, 0.0),
         'Co': (1.35, 1.52, 0.0),
         'Ni': (1.35, 1.49, 1.63),
         'Cu': (1.35, 1.45, 1.4),
         'Zn': (1.35, 1.42, 1.39),
         'Ga': (1.3, 1.36, 1.87),
         'Ge': (1.25, 1.25, 2.11),
         'As': (1.15, 1.14, 1.85),
         'Se': (1.15, 1.03, 1.9),
         'Br': (1.15, 0.94, 1.84),
         'Kr': (0.0, 0.88, 2.02),
         'Rb': (2.35, 2.65, 3.03),
         'Sr': (2.0, 2.19, 2.49),
         'Y': (1.8, 2.12, 0.0),
         'Zr': (1.55, 2.06, 0.0),
         'Nb': (1.45, 1.98, 0.0),
         'Mo': (1.45, 1.9, 0.0),
         'Tc': (1.35, 1.83, 0.0),
         'Ru': (1.3, 1.78, 0.0),
         'Rh': (1.35, 1.73, 0.0),
         'Pd': (1.4, 1.69, 1.63),
         'Ag': (1.6, 1.65, 1.72),
         'Cd': (1.55, 1.61, 1.58),
         'In': (1.55, 1.56, 1.93),
         'Sn': (1.45, 1.45, 2.17),
         'Sb': (1.45, 1.33, 2.06),
         'Te': (1.4, 1.23, 2.06),
         'I': (1.4, 1.15, 1.98),
         'Xe': (0.0, 1.08, 2.16),
         'Cs': (2.6, 2.98, 3.43),
         'Ba': (2.15, 2.53, 2.68),
         'La': (1.95, 2.26, 0.0),
         'Ce': (1.85, 2.1, 0.0),
         'Pr': (1.85, 2.47, 0.0),
         'Nd': (1.85, 2.06, 0.0),
         'Pm': (1.85, 2.05, 0.0),
         'Sm': (1.85, 2.38, 0.0),
         'Eu': (1.85, 2.31, 0.0),
         'Gd': (1.8, 2.33, 0.0),
         'Tb': (1.75, 2.25, 0.0),
         'Dy': (1.75, 2.28, 0.0),
         'Ho': (1.75, 2.26, 0.0),
         'Er': (1.75, 2.26, 0.0),
         'Tm': (1.75, 2.22, 0.0),
         'Yb': (1.75, 2.22, 0.0),
         'Lu': (1.75, 2.17, 0.0),
         'Hf': (1.55, 2.08, 0.0),
         'Ta': (1.45, 2.0, 0.0),
         'W': (1.35, 1.93, 0.0),
         'Re': (1.35, 1.88, 0.0),
         'Os': (1.3, 1.85, 0.0),
         'Ir': (1.35, 1.8, 0.0),
         'Pt': (1.35, 1.77, 1.75),
         'Au': (1.35, 1.74, 1.66),
         'Hg': (1.5, 1.71, 1.55),
         'Tl': (1.9, 1.56, 1.96),
         'Pb': (1.8, 1.54, 2.02),
         'Bi': (1.6, 1.43, 2.07),
         'Po': (1.9, 1.35, 1.97),
         'At': (0.0, 1.27, 2.02),
         'Rn': (0.0, 1.2, 2.2),
         'Fr': (0.0, 0.0, 3.48),
         'Ra': (2.15, 0.0, 2.83),
         'Ac': (1.95, 0.0, 0.0),
         'Th': (1.8, 0.0, 0.0),
         'Pa': (1.8, 0.0, 0.0),
         'U': (1.75, 0.0, 1.86),
         'Np': (1.75, 0.0, 0.0),
         'Pu': (1.75, 0.0, 0.0),
         'Am': (1.75, 0.0, 0.0),
         'Cm': (1.76, 0.0, 0.0)}
