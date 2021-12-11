
import sys
import subprocess

subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'xgboost'])

import pandas as pd
import tarfile
import xgboost as xgb

if __name__=='__main__':
    input_file = '/opt/ml/processing/input/boston.csv'
    input = pd.read_csv(input_file)
    
    model_artifacts = '/opt/ml/processing/model/model.tar.gz'
    with tarfile.open(model_artifacts) as model:
        model.extractall('/opt/ml/processing/model')
    
    loaded_model = xgb.Booster()
    loaded_model.load_model('/opt/ml/processing/model/model.json')
    
    predictions = loaded_model.inplace_predict(input.loc[:, input.columns != 'PRICE'])
    input['PREDICTED'] = predictions
    
    input.to_csv('/opt/ml/processing/results/results.csv', index=False)
