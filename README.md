# Airtest

This repository contains exploratory material for the AirfRANS dataset. A Jupyter notebook located in `airfoil-project/notebooks/data_exploration.ipynb` analyses the dataset structure and provides recommendations for modelling.

## Dataset exploration script

A small CLI script is provided in `airfoil-project/scripts/explore_dataset.py` to replicate basic statistics outside of Jupyter:

```bash
python airfoil-project/scripts/explore_dataset.py --path DATASET_DIR --task scarce --outfile stats.csv
```

The script uses the [`lips`](https://github.com/Extrality/lips) package to load the AirfRANS dataset and saves descriptive statistics to the specified CSV file.
