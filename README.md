# ConflictNet

This project aims to develop and compare deep learning models for conflict forecasting. The models here scrutenized are of the ConflictNet architecture based on recurrent U-net architecture.

## Folder Structure

```
|-README.md # The file you are reading
|-requirements.txt #
|-log.txt # Log of what have been tried, and some todo
|-data
|   |-raw # Raw data files, e.g. tabular data from viewser
|   |-processed # Processed data, e.g. tensor transformations of the viewser data
|   |-generated # Generated data, e.g. posterior distributions of forecasts and metrics
|
|-models # Trained models
|-notebooks # Jupyter notebooks for development
|-reports # Reports, figures, plots, outputsummeries
|   |-plots # Plots and figures 
|   |-timelapse # Timeslapse of conflict developments
|
|-src # Sources code for ConflictNet
    |-dataloaders # Scripts to get data and transform it
    |-networks # Pytorch network scripts
    |-utils # General functions
    |-configs # configuration files for hyperparameters and WandB
    |-training # Traing and validation scripts
    |-evaulaiton # Test and evaluation scripts
    |-visualization # Scripts to generate plots, figures and timelapse
```

## Dependencies
....



## Running the Code
....



## References
...
