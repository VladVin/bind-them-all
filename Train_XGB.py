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


from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, Fragments
import numpy as np


def featurize_ecfp4_2(smiles_list, radius=2, n_bits=2048):
    features = []

    # List of molecular descriptors
    descriptor_functions = [
        Descriptors.TPSA,  # Topological Polar Surface Area
        Descriptors.MolLogP,  # LogP
        Descriptors.MolWt,  # Molecular Weight
        Descriptors.NumHAcceptors,  # Number of H-bond Acceptors
        Descriptors.NumHDonors,  # Number of H-bond Donors
        Descriptors.FractionCSP3,  # Fraction of sp3 Carbons
        rdMolDescriptors.CalcNumRotatableBonds,  # Number of Rotatable Bonds
        Descriptors.BalabanJ,  # Balaban J index
        Descriptors.BertzCT,  # Bertz CT
        Descriptors.Chi0,  # Chi0
        Descriptors.Chi0v,  # Chi0v
        Descriptors.Chi1,  # Chi1
        Descriptors.Chi1v,  # Chi1v
        Descriptors.Chi2n,  # Chi2n
        Descriptors.Chi2v  # Chi2v
    ]

    # List of fragment-based features
    fragment_functions = [
        Fragments.fr_Al_COO,
        Fragments.fr_ArN,
        Fragments.fr_Ar_COO,
        Fragments.fr_Ar_N,
        Fragments.fr_Ar_NH,
        Fragments.fr_Ar_OH,
        Fragments.fr_COO,
        Fragments.fr_COO2,
        Fragments.fr_NH0,
        Fragments.fr_NH1,
        Fragments.fr_NH2,
        Fragments.fr_N_O,
        Fragments.fr_Ndealkylation1,
        Fragments.fr_Ndealkylation2,
        Fragments.fr_alkyl_carbamate,
        Fragments.fr_dihydropyridine,
        Fragments.fr_epoxide,
        Fragments.fr_ester,
        Fragments.fr_ether,
        Fragments.fr_furan,
        Fragments.fr_guanido,
        Fragments.fr_halogen
    ]

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            mol_features = list(fp)

            # Compute molecular descriptors
            for func in descriptor_functions:
                mol_features.append(func(mol))

            # Compute fragment-based features
            for func in fragment_functions:
                mol_features.append(func(mol))

            features.append(np.array(mol_features))
        else:
            # Length of zero vector is n_bits + len(descriptor_functions) + len(fragment_functions)
            features.append(np.zeros((n_bits + len(descriptor_functions) + len(fragment_functions),)))

    return np.array(features)


# Example usage
smiles_list = ["CCO", "CCN", "CCC"]
feature_array = featurize_ecfp4(smiles_list)
print(feature_array)

# Example usage
pm = load_parameters_from_yaml()
score = {}
seed = 42
name = 'test1'

mode = 'ecpf4_result'
if mode == 'ecpf4':
    #load /home/ubuntu/data/splits/hi/train.tsv
    train = pd.read_csv('../../data/splits/hi/train_with_external.tsv', sep='\t')
    valid = pd.read_csv('../../data/splits/hi/valid.tsv', sep='\t')

    #Train/valid of form:
    #smiles	logS	S	scaffold
    #O=C(NCCc1ccc2c(c1)OCO2)C1CCCC1	1.624178926	42.09000002442061	C1CCC(CCCCC2CCC3CCCC3C2)C1
    #CCc1nc(NC(=O)c2ccccc2F)sc1C	1.361916619	23.0100000175561	C1CCC(CCC2CCCC2)CC1

    # Featurize SMILES using ECFP4
    X_train = featurize_ecfp4(train['smiles'])
    y_train = train['logS']

    X_valid = featurize_ecfp4(valid['smiles'])
    y_valid = valid['logS']

    # Apply the condition to set logS values less than -1 to -1
    y_train = train['logS'].apply(lambda x: x if x >= -1 else -1)
    y_valid = valid['logS'].apply(lambda x: x if x >= -1 else -1)

    Do_XGradientBoost_regression(X_train, y_train, X_valid, y_valid, name, pm, score, seed)

if mode == 'embeddings':
    train = pd.read_csv('../../data/splits/hi/train.tsv', sep='\t')
    valid = pd.read_csv('../../data/splits/hi/valid.tsv', sep='\t')
    train_full = pd.read_csv('../../data/internal_external.tsv', sep='\t')




    y_train = train['logS']
    y_valid = valid['logS']
    y_train_full = train_full['logS']

    X_train = np.load('../../data/splits/hi/train_molformer_xl_both_10pct_embeds.npy')
    X_train_full = np.load('../../data/internal_external_molformer_xl_both_10pct_embeds.npy')
    X_valid = np.load('../../data/splits/hi/valid_molformer_xl_both_10pct_embeds.npy')
    X_test = np.load('../../data/test_molformer_xl_both_10pct_embeds.npy')
    y_test = None

    Do_XGradientBoost_regression(X_train, y_train, X_test, y_test, name, pm, score, seed, X_valid, y_valid, X_train_full, y_train_full)

if mode == 'ecpf4_result':
    #load /home/ubuntu/data/splits/hi/train.tsv
    train = pd.read_csv('../../data/splits/hi/train.tsv', sep='\t')
    valid = pd.read_csv('../../data/splits/hi/valid.tsv', sep='\t')
    full_train = pd.read_csv('../../data/train.tsv', sep='\t')
    test = pd.read_csv('../../data/test.tsv', sep='\t')

    # Featurize SMILES using ECFP4
    X_train = featurize_ecfp4(train['smiles'])
    y_train = train['logS']

    X_train_full = featurize_ecfp4(full_train['smiles'])
    y_train_full = full_train['logS']

    X_valid = featurize_ecfp4(valid['smiles'])
    y_valid = valid['logS']

    X_test = featurize_ecfp4(test['smiles'])
    y_test = None


    Do_XGradientBoost_regression(X_train, y_train, X_test, y_test, name, pm, score, seed, X_valid, y_valid, X_train_full, y_train_full)

if mode == 'both':
    if mode == 'embeddings':
        train = pd.read_csv('../../data/splits/hi/train.tsv', sep='\t')
        valid = pd.read_csv('../../data/splits/hi/valid.tsv', sep='\t')
        train_full = pd.read_csv('../../data/internal_external.tsv', sep='\t')

        y_train = train['logS']
        y_valid = valid['logS']
        y_train_full = train_full['logS']

        X_train = np.load('../../data/splits/hi/train_molformer_xl_both_10pct_embeds.npy')
        X_train_full = np.load('../../data/internal_external_molformer_xl_both_10pct_embeds.npy')
        X_valid = np.load('../../data/splits/hi/valid_molformer_xl_both_10pct_embeds.npy')
        X_test = np.load('../../data/test_molformer_xl_both_10pct_embeds.npy')
        y_test = None

        Do_XGradientBoost_regression(X_train, y_train, X_test, y_test, name, pm, score, seed, X_valid, y_valid,
                                     X_train_full, y_train_full)




