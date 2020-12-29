# basic requirements
- python 3.6
- torch 1.7.1
- numpy 1.19.4
# installation
```
pip install -r requirements.txt
```
# run 
1. Configure settings in `settings.py`, espeically for _mode_ and _session_ parameters
2. Prepare data
```
python prepare_data/pipeline.py
```
3. Train models
```
python CNN_train.py
```
4. Predict scores

```
python aggregate_scores.py
```
