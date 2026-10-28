# Network Analyser Based Anomaly Detection for Early Cancer Screening

This repository contains the software developed for the EEE4022S final-year Electrical and Computer Engineering project at the University of Cape Town (2026).

The project aims to develop a Vector Network Analyser (VNA)-based system for detecting anomalies in tissue-mimicking materials using microwave imaging. It builds upon previous work by integrating automated scanning, dielectric property measurement, image reconstruction, and visualisation into a single graphical interface.

## Getting Started

**The main entry point for the software is `GUI.py`.** The other Python files provide supporting functionality and do not need to be run individually for normal use.

Running the `GUI.py` file will run the system's main output. This will require several packages:
- `numpy`
- `scipy`
- `matplotlib`
- `scikit-rf`
- `pyserial`
- `PyOECP`
- `plotly`
- `kaleido`
- `scikit-image`
You can install them using:

```bash
pip install numpy scipy matplotlib scikit-rf pyserial PyOECP plotly kaleido scikit-image
```

The GUI also uses `tkinter`, which is included with many Python installations but may need to be installed separately on some operating systems.

The graphical interface provides access to the system's main functions. Some functionality requires the VNA and/or positioning rig to be connected and configured. The rest of the functionality will need output files.

## Main Functionality

The software integrates the following features:

- **VNA interfacing:** Configure the VNA and acquire reflection coefficient (S11) measurements.
- **Dielectric measurement:** Calculate complex relative permittivity from open-ended coaxial probe measurements using the Komarov model.
- **Automated scanning:** Control a two-dimensional positioning rig to perform raster scans of a target.
- **Image reconstruction:** Process acquired measurements and generate three-dimensional reconstructions using Delay-Multiply-And-Sum (DMAS) beamforming.
- **Signal processing:** Apply techniques including Singular Value Decomposition (SVD), coherence weighting, and distance weighting.
- **Visualisation:** View and analyse acquired measurements and reconstructed images.

## Repository Structure

- `LaptopCode/` — Main graphical interface, VNA communication, dielectric calculations, scan control, and image reconstruction software.
- `JetsonCode/` — Software for controlling the positioning rig through the NVIDIA Jetson platform. This is already loaded onto the rig.
- `PhysicalModels/` — Physical model files associated with the project.

## Hardware

The complete experimental system uses:

- A LiteVNA.
- An open-ended coaxial probe for dielectric measurements.
- A horn antenna for microwave imaging.
- A two-dimensional positioning rig controlled through an NVIDIA Jetson.

To interface with the NVIDIA, an Ethernet cable will need to connect the two. The IP addresses in the connection may need to be reassigned to match the code:
The laptop should be: `192.168.180.1`
The Jetson should be: `192.168.180.2`

If the Jetson server refuses to connect, you may need to `ssh` into it:
`ssh jetson@192.168.180.2`
The password can be supplied upon request.

Once in, the server script can be run:
`python3 Documents/jetson_motor_server.py`

## Author

Justin Heathcote-Marks  
University of Cape Town, Department of Electrical Engineering  
EEE4022S Research Project, 2026

Supervisor: Dr. Stephen Paine
