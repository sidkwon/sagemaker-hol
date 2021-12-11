
import xgboost as xgb

import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston

import pickle as pkl
import argparse
import os

parser = argparse.ArgumentParser()

# Hyperparameters are described here.
parser.add_argument("--max_depth", type=int, default=5)
parser.add_argument("--eta", type=float, default=0.2)
parser.add_argument("--gamma", type=int, default=4)
parser.add_argument("--min_child_weight", type=int, default=6)
parser.add_argument("--subsample", type=float, default=0.7)
parser.add_argument("--verbosity", type=int, default=2)
parser.add_argument("--objective", type=str, default='reg:squarederror')
parser.add_argument("--num_round", type=int, default=50)
parser.add_argument("--tree_method", type=str, default="auto")
parser.add_argument("--predictor", type=str, default="auto")

# SageMaker specific arguments. Defaults are set in the environment variables.
parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
# parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])

args = parser.parse_args()

data = pd.read_csv(f'{args.train}/boston.csv')
X, y = data.iloc[:,:-1], data.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

train = xgb.DMatrix(X_train, y_train)
test = xgb.DMatrix(X_test, y_test)

train_hp = {
    "max_depth": args.max_depth,
    "eta": args.eta,
    "gamma": args.gamma,
    "min_child_weight": args.min_child_weight,
    "subsample": args.subsample,
    "verbosity": args.verbosity,
    "objective": args.objective,
    "tree_method": args.tree_method,
    "predictor": args.predictor,
}

model_xgb = xgb.train(params=train_hp, 
                      dtrain=train,
                      evals=[(train, "train"), (test, "validation")], 
                      num_boost_round=100,
                      early_stopping_rounds=20)

preds = model_xgb.predict(test)

rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))

model_xgb.save_model(f'{args.model_dir}/model.json')
