import os
import yaml
import pandas as pd
import numpy as np

params = yaml.safe_load(open("params.yaml"))['preprocess']

def preprocess(input_path, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.read_csv(input_path)
    
    y = df.iloc[:, -1].values
    
    label2index = {}
    for i, label in enumerate(np.unique(y)):
        label2index[label] = i
    y = np.array([label2index[label] for label in y])
    
    df.iloc[:, -1] = y
    df = df.iloc[:, 1:]
    
    df.to_csv(output_path, index=False)
    

if __name__ == '__main__':
    preprocess(params['input'], params['output'])