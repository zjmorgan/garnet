# The Single Crystal *GARNET* project
Single Crystal Graphical Advanced Reduction Neutron Event Toolkit

Garnets are a group of minerals with high symmetry cubic crystal system with space group *Ia-3d* (#230). 
Although they come in many colors, the word comes from a 14th-century Middle English word that has the meaning *dark red* due to the color of many naturally occuring silicate minerals.
Some rare-earth synthetic garnets has recently served as a useful calibration standard used across several beamlines.

The goal of this project is to combine several amorphous tools from many of the instruments into a user-friendly environment for data reduction.
The scope of this project only covers reduction post-data collection.

Future development may incorporate live data reduction or analysis, but that is not the focus in this effort.

Scope of covered instruments
- TOPAZ
- MANDI
- CORELLI
- DEMAND
- WAND2
- SNAP

The garnet tool will allow users to select single crystal diffraction data from one (minimally white beam) or more (minimally monochromatic beam) orientations, and transform it into a meaningful form.
There exists essential steps of a single crystal data reduction.
These include:
- UB matrix determination and refinement for data reduction and experiment planning
- Peak integration and corrections for structure refinement
- Reciprocal space reconstruction for visualization and analysis
- Order parameter tracking and event filtering analysis

Data processing will be based on Mantid and use PyQt5 for the application.

This project will be broken down into phases, milestones, or components
- [Milestone 0: Deployment](https://github.com/users/zjmorgan/projects/1?pane=issue&itemId=38115185)
- [Milestone 1: UB-matrix determination and refinement](https://github.com/users/zjmorgan/projects/1?pane=issue&itemId=38115119)

### UB determination
1. Load data
2. Apply calibration
3. Convert to Q-space
4. Find strong peaks
5. Integrate peaks
6. Find intitial UB
7. Transform UB
8. Optimize UB
9. Plan experiment
10. Setup auto-reduction
11. Launch reduction workflow

### Peak integration 
1. Load data
2. Apply calibration
3. Convert to Q-space
4. Predict peak positions
5. Integrate peaks
6. Combine peaks from all runs
7. Apply spectrum, efficiency, and absorption corrections
8. Save HKL lists

### Order parameter tracking
1. Load data
2. Apply calibration
3. Define detector region of interest near a peak
4. Filter events by log value
5. Convert to target dimension
6. Fit peak intensity
7. Plot intensity vs log value 
   
### Reciprocal space reconstruction
1. Load data
2. Apply calibration
3. Convert to Q-space
4. Normalize with vanadium
5. Apply symmetry operators
6. Subtract background
