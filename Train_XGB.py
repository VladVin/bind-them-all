import yaml
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from Ben.Train_XGB_functions import Do_XGradientBoost_regression


# read in config.yml
def load_parameters_from_yaml(file_path='config.yml'):
    with open(file_path, 'r') as file:
        parameters = yaml.safe_load(file)
    return parameters

def featurize_ecfp4(smiles_list, radius=2, n_bits=2048):
    features = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            features.append(np.array(fp))
        else:
            features.append(np.zeros((n_bits,)))
    return np.array(features)

# Example usage
pm = load_parameters_from_yaml()
score = {}
seed = 42
name = 'test1'

#load /home/ubuntu/data/splits/hi/train.tsv
train = pd.read_csv('../data/splits/hi/train.tsv', sep='\t')
valid = pd.read_csv('../data/splits/hi/valid.tsv', sep='\t')

#Train/valid of form:
#smiles	logS	S	scaffold
#O=C(NCCc1ccc2c(c1)OCO2)C1CCCC1	1.624178926	42.09000002442061	C1CCC(CCCCC2CCC3CCCC3C2)C1
#CCc1nc(NC(=O)c2ccccc2F)sc1C	1.361916619	23.0100000175561	C1CCC(CCC2CCCC2)CC1

# Featurize SMILES using ECFP4
X_train = featurize_ecfp4(train['smiles'])
y_train = train['logS']
X_valid = featurize_ecfp4(valid['smiles'])
y_valid = valid['logS']

Do_XGradientBoost_regression(X_train, y_train, X_valid, y_valid, name, pm, score, seed)

