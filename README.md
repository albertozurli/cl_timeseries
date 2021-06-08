
# Continual Learning on financial time series

Code repo for thesis in Continual Learning @ Axyon AI 

Model can be customized, for further information refer to
```
python ./main.py -h
```
Input data can also be pre-processed with `--processing`:
* `--difference`
* To update

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

## Online learning:

Model can be executed for both regression and classification (regression not working at this moment)

Online training and testing:
```
python ./main.py --online 
```

## Continual learning with ER:

Continual training and testing:
```
python ./main.py --continual
```

### TO DO:
* Add indicators( from TA-LIB) as a feature
* Add cl metrics(forward/backward transfer and forgetting) 
* Fix Tensorboard
* Test with different buffer size [50,100,200,500] (1000 is imposible with oil monthly dataset,maybe it works with others)
* Test with different sequence timestep (actually 4 weeks of observation and prediction next month)


