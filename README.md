# Understanding European Heatwaves with Variational Autoencoders

This repository contains the code and models used in the paper "Understanding European Heatwaves with Variational Autoencoders" submitted to Earth System Dynamics. 

> Paçal, A., Hassler, B., Weigel, K., Fernández-Torres, M.-Á., Camps-Valls, G., & Eyring, V. (2025). Understanding European Heatwaves with Variational Autoencoders, *Earth System Dynamics*. (Submitted)

**Corresponding Author**: Aytaç Paçal ([aytac.pacal@dlr.de](mailto:aytac.pacal@dlr.de))

<!-- [![DOI](https://zenodo.org/badge/DOI/zenodo.X.X.X.svg)](https://doi.org/zenodo.X.X.X) -->

## Description

This study applies variational autoencoders (VAEs) to analyze and understand European heatwave patterns. We use a 3D convolutional VAE architecture to capture both spatial and temporal features of heatwave events. The model is trained on heatwave samples extracted form ERA5 reanalysis data, which are avaiable on the [Copernicus Climate Data Store (CDS)](https://cds.climate.copernicus.eu/datasets). We used the ERA5 reanalysis data from DKRZ (Deutscher Wetterdienst) data pool. The data is provided in GRIB format and can be converted to NetCDF using the `grib_file_converter_cdo.py` script.

The VAE enables dimensionality reduction and feature extraction from complex meteorological data, allowing for the identification of key patterns and drivers behind European heatwave events. This approach provides new insights into the complex dynamics of extreme temperature events in Europe and their potential changes under climate change.

## Usage

Configure the model and training parameters in the `main.yaml` file. The model can be trained and tested using the provided scripts. The main entry point for the application is `main.py`, which orchestrates the training and evaluation process. To train the model, set the 'mode' parameter to 'train' in the `main.yaml` file. For testing, set it to 'test'. The training and testing scripts are `train.py` and `test.py`, respectively. `main.py` will call the appropriate script based on the mode specified, loads the configuration, initializes the model, and starts the training or testing process.
To run `main.py`, use the following command:

```bash
python main.py --config main.yaml
```

## Repository Structure

```bash
├── main.py                # Main entry point for the application
├── main.yaml              # Main configuration file
├── train.py               # Training script for VAE models
├── val.py                 # Validation script
├── test.py                # Testing and plotting script
├── data/                  # Directory containing input data
│   └── unique_hot_grid_clusters.csv  
├── models/                
│   └── VAEConv3D.py       # 3D Convolutional VAE 
├── saved_models/          # Trained model weights
├── scripts/               
│   ├── deaseasonalize.py  # Script to remove seasonal patterns from data
│   ├── era5_stream.py     # Script to calculate streamfunction from ERA5 data
│   ├── era5_t2m_hot_grid_points.py  # Script to identify hot grid points
│   ├── grib_file_converter_cdo.py   # Script to convert GRIB files
│   └── grib_file_converter_cdo.yaml # Configuration for GRIB conversion
└── utils/                 
    ├── misc.py            # Helper functions
    └── data_loaders/      # Data loading utilities
        ├── load_data_from_nc.py      # NetCDF data loader
        ├── load_data_from_npy.py     # NumPy data loader
        └── load_heatwave_samples.py  # Loader for heatwave samples
```

## Acknowledgements

Funding for this study was provided by the European Research Council (ERC) Synergy Grant 'Understanding and Modelling the Earth System with Machine Learning' ([USMILE](https://www.usmile-erc.eu/)) under the Horizon 2020 research and innovation programme (Grant Agreement No. 855187). M.-A.F-T. and G.C-V also acknowledge funding from the project ``eXtreme events: Artificial Intelligence for Detection and Attribution'' ([XAIDA](https://xaida.eu/)) under the H2020 programme (Grant Agreement No. 101003469). K.W. acknowledges funding by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) through the Gottfried Wilhelm Leibniz Prize awarded to Veronika Eyring (Reference number EY 22/2-1).
This work used the Deutsches Klimarechenzentrum (DKRZ) resources granted by its Scientific Steering Committee (WLA) under project ID bd1083. The [ERA5 reanalysis data](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-complete?tab=overview) was accessed from [DKRZ](https://docs.dkrz.de/doc/dataservices/finding_and_accessing_data/era_data/index.html#). The results contain modified Copernicus Climate Change Service information for 2020. Neither the European Commission nor ECMWF is responsible for any use that may be made of the Copernicus information or data it contains. 

## License

This code is released under Apache 2.0. See [LICENSE](LICENSE) for details.
