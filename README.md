
# Continual Learning on financial time series

Code repo for thesis in Continual Learning @ Axyon AI 

Model is customizable, for further information refer to
```
python ./main.py -h
```
Input data can also be pre-processed with `--processing`:
* `--difference`
* `--indicators`

An example of BOCD is available at `notebook/chp_analysis.ipynb` 
and running `python ./main.py --split`


## Package and dependencies:

First, install the BOCD package via

```
pip install -e .
```
in ```detection ``` repo

then install dependencies via 

```
pip install -r requirements.txt
```
 in main project repo

[PyTorch](https://pytorch.org/) and [TA-Lib](https://github.com/mrjbq7/ta-lib) are also required, please visit the official pages

## Online learning:

Model can be executed for both regression and classification (regression not working at the moment)

Online training and testing:
```
python ./main.py --online 
```

## Continual learning with ER:

Continual training and testing:
```
python ./main.py --er
```

### TO DO:
* Test with different sequence timestep (actually 4 weeks of observation and prediction 4 weeks later)
* Implement regularization (EWC,LwF) and architectural (PNN) methods, if possible in Domain-IL scenario
* Save results(loss, accuracy) in a file, actually in a .txt but in the future .csv is better 
