
# Continual Learning on financial time series

Code repo for thesis in Continual Learning @ Axyon AI 

Model is customizable, for further information refer to
```
python ./main.py -h
```
Input data can also be pre-processed with `--processing`:
* `--none`
* `--difference`
* `--indicators` (Project is tested with this option)

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
## Model:
+ By default, the DL backbone used is a simple MLP with Dropout
+ In addition a CNN with 1-D Convolution with L1 Norm regulariation might be used (to be fixed and tested) with the argument 
```--cnn ```
## Online learning:

Model can be executed for both regression and classification (regression not working at the moment)

Online training:
```
python ./main.py  
```
or
```
python ./main.py  --model online
```

## Other CL methods
With the option ```--model ``` can be trained (and tested) with the following models:
+ Averaged GEM
+ Avergaed GEM with Reservoir
+ Dark Experience Replay
+ Elastic Weight Consolidation
+ Experience Replay
+ Gradient Episodic Memory
+ Synaptic Intelligence

## Evaluation
Wiht the argument ```--evaluate ``` each model can be tested each epoch for both current and previous tasks with a final recap .csv file.


### TO DO:
* Finish regularization for CNN
* Classification on % of outscore/outperform and not on price
* Test with different sequence timestep (actually 30 days of observation and prediction 30 days later)
