# IIMT2641 - Introduction to Business Analytics

## Group Project: A5

### Predictive Modelling for Enhanced Police Response: Forecasting Monthly Accident Rates Using Transport, Economic, and Weather Data

#### Group Members

| Name           | Student ID   |
|----------------|--------------|
| Lam Yeuk Yu    | 3036234009   |
| Cheng Jeffrey  | 3035801489   |
| Wong Ching Lam | 3036080109   |
| Li Sze Ki      | 3036080496   |
| Tang Chi Hong  | 3036234176   |

---

## Project Strucutre

```bash
├── csv
│   ├── accidents_by_month_data.csv
│   ├── cpi_data_monthly.csv
│   ├── daily_HKO_RF_ALL.csv
│   ├── hko_rf_monthly.csv
│   ├── hsi_index_data.csv
│   ├── traffic_data_monthly.csv
├── .gitignore
├── requirements.txt # Python dependencies
├── Rplots.pdf # Plot of the model
├── summary.pdf # Summary of the project
├── summary.md # Summary of the project
├── README.md
├── analysis.R # R script for running the model
└── rf_convert.py # Python script for converting daily rainfall data to monthly
```


## Instructions to running the model

First install python dependencies by running:

`pip3 install -r requirements.txt`

Then if necessary to update and calculate the data of monthly rainfall of Hong Kong, run:

`python3 rf_convert.py`

To run the R model, run:

`Rscript analysis.R`
