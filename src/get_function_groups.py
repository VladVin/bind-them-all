import polaris as po
import datamol as dm
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from matplotlib import pyplot as plt
import pandas as pd
import argparse

def get_wt(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Descriptors.MolWt(mol)

def get_nH(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Descriptors.NumHDonors(mol)

def get_logp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Descriptors.MolLogP(mol)

functional_groups = {
    'Hydroxyl (-OH)': Chem.MolFromSmarts('[OH]'),
    'Methyl (-CH3)': Chem.MolFromSmarts('[CH3]'),
    'Amine (-NH2)': Chem.MolFromSmarts('[NH2]'),
    #'Carbonyl (C=O)': Chem.MolFromSmarts('[C=O]'),
    #'Carboxyl (-COOH)': Chem.MolFromSmarts('[C](=O)[O][H]'),
    #'Benzoyl (Ph-C=O)': Chem.MolFromSmarts('[c]1[cH][cH][cH][cH][cH]1C(=O)'),
    'Nitro (-NO2)': Chem.MolFromSmarts('[N+](=O)[O-]'),
    'Ether (R-O-R\')': Chem.MolFromSmarts('[OD2]([#6])[#6]'),
    'Aldehyde (-CHO)': Chem.MolFromSmarts('[CH]=O'),
    'Ketone (-C=O)': Chem.MolFromSmarts('[C](=O)[C]'),
    'Sulfonyl (R-SO2-R\')': Chem.MolFromSmarts('S(=O)(=O)[#6]'),
    'Phenyl (Ph)': Chem.MolFromSmarts('c1ccccc1'),
    #'Thiol (-SH)': Chem.MolFromSmarts('[SH]'),
    'Amide (-CONH2)': Chem.MolFromSmarts('[C](=O)[NH2]'),
    'Alcohol (-OH)': Chem.MolFromSmarts('[OH]')
}

def find_functional_groups(smiles, functional_groups):
    mol = Chem.MolFromSmiles(smiles)
    groups_present = {group_name: len(mol.GetSubstructMatches(smarts)) for group_name, smarts in functional_groups.items()}
    return groups_present

def get_features(df):
    df_train_fgroup = pd.DataFrame([find_functional_groups(s, functional_groups) for s in df['smiles']])
    df_train_fgroup['wt'] = [get_wt(i) for i in df['smiles']]
    df_train_fgroup['nH'] = [get_nH(i) for i in df['smiles']]
    df_train_fgroup['logp'] = [get_logp(i) for i in df['smiles']]
    df = pd.concat([df, df_train_fgroup], axis=1)

    return df

train_hi = pd.read_csv("~/data/splits/hi/train.tsv", sep='\t')
valid_hi = pd.read_csv("~/data/splits/hi/valid.tsv", sep='\t')
train = pd.read_csv("~/data/train.tsv", sep='\t')
test = pd.read_csv("~/data/test.tsv", sep='\t')

train_hi = get_features(train_hi)
valid_hi = get_features(valid_hi)
train = get_features(train)
test = get_features(test)

train_hi.to_csv("~/data/splits/hi/train_func_groups.tsv", sep='\t', index=False)
valid_hi.to_csv("~/data/splits/hi/valid_func_groups.tsv", sep='\t', index=False)
train.to_csv("~/data/train_func_groups.tsv", sep='\t', index=False)
test.to_csv("~/data/test_func_groups.tsv", sep='\t', index=False)
