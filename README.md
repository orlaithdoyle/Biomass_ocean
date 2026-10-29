# Biomass P-band SAR over the ocean

Code for my ESA ESTEC internship (May to October 2026) looking at whether Biomass, ESA's P-band SAR, can be used over the ocean: sea surface salinity, temperature, sea state and surface features.

Orlaith Doyle (Ní Dhúill), supervised by Björn Rommen (ESA ESTEC).

The full write-up is the internship report, *Investigation of Biomass P-band SAR for Ocean Applications*. The notebooks follow the order of its Section 4. The results are early and preliminary, and so is some of this code.

## What's here

| Notebook | What it does | Report |
|---|---|---|
| `01_catalogue_download.ipynb` | MAAP login, search and download of Level-1a products, reference data downloads | 3.1, 3.2 |
| `01b_swath_scanner.ipynb` | QuickSwathScanner: scans the whole Level-1a catalogue, one record per unique swath location, ocean / coastal / land classification, world and region maps. Resumable and checkpointed | 3.1 |
| `02_processing.ipynb` | Band order check, range-Doppler geocoding, WorldCover land mask, multilooking, sigma nought, Pauli, CPD and coherence, H and alpha, instrument checks, pass strips | 4.1 |
| `03_model_reproduction.ipynb` | Klein and Swift permittivity, the Taillade et al. (2023) Table I sensitivities with pure and tilted Bragg, the size of the salinity signal, the Elfouhaily spectrum | 4.2 |
| `04_salinity.ipynb` | Repeat-pass stability, matching to SMOS, within-pass anomalies, the controlled leave-one-pass-out regression, the radar-only index | 4.3, 5.1, 5.2 |
| `05_wave_chain.ipynb` | Tiled image spectra and sub-look cross-spectra, whitening, swell wavelength and direction of travel, checks against crests, the wave model, ERA5 and the Faraday map. Starts with a test on simulated swell | 4.4, 5.4 |
| `06_sea_features.ipynb` | Detecting dark and bright patches, comparing them with the surrounding sea and with rain, wind, SST and salinity, classifying them, and the 9 x 9 block test | 4.5, 5.5 |
| `bm_tools.py` | Shared functions: reading products, geocoding, land mask, multilooking, circular stats, seawater permittivity, Bragg models, Elfouhaily spectrum, readers for the reference data | |

Each notebook starts with one cell holding the paths and the scene. To run it on a new scene, edit that cell only. Notebook 02 saves one `.npz` per frame, which 04 and 06 read. Every figure is saved to disk as well as shown.

## Setting up

```
conda create -n biomass python=3.11
conda activate biomass
pip install -r requirements.txt
```

Then copy `eo_credentials_template.txt` to `eo_credentials.txt` (somewhere outside the repo is safest), fill it in, and point notebook 01 and the swath scanner at it. Keep the quotes around each value, the swath scanner reads the file as Python. The real file is in `.gitignore` and should never be committed. The MAAP offline token expires after about 90 days, or sooner if it isn't used, and then everything fails with a 401 and no other warning.

Accounts needed:

- **ESA MAAP** for Biomass (explorer.maap.eo.esa.int)
- **Copernicus Marine** for SMOS, the global ocean analysis, OSTIA, ODYSSEA and the wave model (`copernicusmarine login` once)
- **Copernicus Climate Data Store** for ERA5 (`~/.cdsapirc`)
- **NASA Earthdata** for IMERG and SMAP

## Things to know

- **Band order.** The SCS TIFFs store the bands as VV, VH, HV, HH, the reverse of the annotation XML. The code reads the order from the band metadata and falls back to VV, VH, HV, HH with a warning. Reading it the wrong way round swaps HH and VV, which flips the sign of the CPD and inverts HH/VV.
- **Look side.** Read from the annotation if it's there, otherwise assumed right looking. A wrong look side mirrors the geolocation, so check the corners printed in notebook 02 against the KML footprint the first time.
- **Annotation tag names.** `bm_tools.read_annotation` looks for a few spellings of each tag. If a product uses another it stops and says which tag it couldn't find, add the name to the list there.
- **Swath scanner run time.** A full scan of the catalogue takes something like 8 to 11 hours and the API drops out now and then. It checkpoints as it goes, so just rerun the scan cell until it says it reached the end.
- **Cross-spectrum sign.** In the first version every tile pointed exactly opposite to the wave model, so for Biomass the sign is reversed (`CROSS_SIGN = -1` in notebook 05). This rests on a few scenes and needs confirming on more.
- **CPD sign convention.** With permittivity written as eps' - i eps'' the Bragg CPD over the sea is about +1 deg. With +i eps'' it flips. Check which one a paper uses before comparing signs.
- **Wave height is not measured.** Where it appears it's an estimate from an assumed steepness, and labelled that way.
- **Dataset IDs.** Copernicus Marine and Earthdata IDs change now and then. If a download fails, check the ID first.

## Credits

The processing chain in 02 and the wave chain in 05 follow the structure of the ocean framework of P. López-Dekker and Y. Yuan for early Biomass data, written separately for this work. Their code is not included here, as it was shared privately. The Bragg modelling was checked against the open source [oceansar](https://github.com/pakodekker/oceansar) library. The baseline polarimetric processing builds on the ESA-DLR PolSAR / Pol-InSAR training course material.

AI tools were used for programming support and editing. All analysis choices and conclusions are my own.

Biomass data: ESA MAAP. SMOS, global ocean analysis, OSTIA, ODYSSEA and wave model: Copernicus Marine Service. SMAP and IMERG: NASA Earthdata. ERA5: Copernicus Climate Change Service. Land mask: ESA WorldCover 2021.
