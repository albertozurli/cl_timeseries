
# Continual Learning on financial time series

Code repo for thesis in Continual Learning @ Axyon AI 

Model can be customized, for further information refer to
```
python ./main.py -h
```
Input data can also be pre-processed with different transformations:
* `--normalize` 
* `--standardize`
* `--difference`
##Package and dependencies:

First, install the BOCP package via

```
pip install -e .
```
in ```detection ``` repo

then install dependencies via 

```
pip install -r requirements.txt
```
 in main project repo

##Training:

Model can be executed for both regression and classification (regression not working at this moment)

Joint training:
```
python ./main.py --joint --train
```
Online training:
```
python ./main.py --online --train
```


##Testing:
Joint testing:
```
python ./main.py --joint --test
```
Online testing:
```
python ./main.py --online --test
```

###TO DO:
* Try different changepoint algorithm params
* Find best pre-processing technique (Discuss with J)
* Implement buffer for ER

