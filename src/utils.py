from chembl_structure_pipeline.standardizer import standardize_mol
import numpy as np
from rdkit.Chem import AllChem


def ECFP_from_smiles(
    smiles,
    R = 2,
    L = 2**10,
    use_features = False,
    use_chirality = False,
    as_array = True):
    
    molecule = AllChem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    
    feature_list = AllChem.GetMorganFingerprintAsBitVect(
        molecule,
        radius = R,
        nBits = L,
        useFeatures = use_features,
        useChirality = use_chirality
    )

    if as_array:
        return np.array(feature_list, dtype=bool)
    else:
        return feature_list


def tanimoto_similarity(x, y):
    x, y = x.astype(np.float32), y.astype(np.float32)
    intersection = x @ y.T
    union = x.sum(axis=1)[:, None] + y.sum(axis=1) - intersection
    sim = intersection / union

    return sim


def standardize_smiles(smiles):
    mol = AllChem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    mol_std = standardize_mol(mol)

    return AllChem.MolToSmiles(mol_std)

