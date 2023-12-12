# GARNET requirements document

## Instruments
Time-of-flight
- TOPAZ (SNS|BL-2)
- MANDI (SNS|BL11B)
- CORELLI (SNS|BL9)
- SNAP (SNS|BL3)
Constant wavelength
- DEMAND (HFIR|HB3A)
- WAND2 (HFIR|HB2C)
Quasi-Laue
- IMAGINE-X (HFIR|CG4D future)

## Reduction plan
Common information required of all reduction workflows
- Instrument
- Experiment number
- Run numbers
- UB-matrix
- Detector calibration files
- Vanadium files
- Mask file

## Reduction workflows
1. UB determination
2. Peak integration
3. Reciprocal space reconstruction/normalization
4. Order parameter tracking
5. Experiment planning

## Intermediate workflows
1. Selecting data
2. Defining sample

## Visualization components
1. Peaks viewer
2. Satellite peaks viewer
3. Reciprocal space viewer

### Instrument
Common information required of all instruments
- Goniometer axes
  - Name, axes, rotation sense, fixed/moveable, average/rotating
- Detector positions
  - Instrument definition file
  - Optional logs
Common information required of time-of-flight instruments
- Wavelength (momentum) band (min/max)
Common information required of constant wavelength instruments
- Incident wavelength

### Experiment number
Common information required of all experiments
- IPTS number
  - Data location `/FACILITY/INSTRUMENT/IPTS-XXXXX/`
  - Facility (SNS/HFIR)
Required of DEMAND instruments SPICE
- Experiment number

### Run numbers
All instruments except DEMAND SPICE
- List of runs (single,range,or mutl)
