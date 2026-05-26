# Data

This project uses the EEGMAT dataset from PhysioNet.

## Dataset

Name: EEG During Mental Arithmetic Tasks  
Authors: Zyma et al. (2019)  
Source: https://physionet.org/content/eegmat/1.0.0/

## Download Instructions

1. Go to the dataset page:
   https://physionet.org/content/eegmat/1.0.0/

2. Download all EDF files and the metadata file (`subject-info.csv`)

3. Place the files in the following directory:

data/raw/

## Expected Structure

data/raw/
├── Subject00_1.edf   # resting state
├── Subject00_2.edf   # mental arithmetic
├── Subject01_1.edf
├── Subject01_2.edf
├── ...
├── subject-info.csv

## Data Description

Each subject has two recordings:
- *_1.edf → baseline (resting state)
- *_2.edf → mental arithmetic task

Each recording:
- duration: 60 seconds (mental math), 180 seconds (rest)
- channels: 23 (10–20 system)
- format: EDF

## Notes

- The dataset is not included in this repository due to size and licensing
- Ensure consistent file naming when loading data
- Data is already preprocessed (artifact removal using ICA)

## Citation

If you use this dataset, please cite:

Zyma I, et al.  
"Electroencephalograms during Mental Arithmetic Task Performance"  
Data, 2019  
https://doi.org/10.3390/data4010014