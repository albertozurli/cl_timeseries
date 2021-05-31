
# Continual Learning on financial time series

Code repo for thesis in Continual Learning @ Axyon AI 

Model can be customized, for further information refer to
```
python ./main.py -h
```
Input data can also be pre-processed with `--processing`:
* `--difference`
* To update

An example of BOCD heatmap is available at `notebook/chp_analysis.ipynb`
(As discussed with J)

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

## Training:

Model can be executed for both regression and classification (regression not working at this moment)

Online training:
```
python ./main.py --online --train
```


## Testing:

Online testing:
```
python ./main.py --online --test
```

### TO DO:
* Add indicators( from TA-LIB) as a feature
* Implement buffer for ER
* Fix Tensorboard

