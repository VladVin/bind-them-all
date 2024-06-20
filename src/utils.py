from chembl_structure_pipeline.standardizer import standardize_mol, exclude_flag
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize


LARGEST_FRAGMENT_CHOOSER = rdMolStandardize.LargestFragmentChooser()


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
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
    except:
        return None
    
    exclude = exclude_flag(mol, includeRDKitSanitization=False)
    if exclude:
        return None

    # Standardize with ChEMBL data curation pipeline. During standardization, the molecule may be broken
    try:
        # Choose molecule with largest component
        mol = LARGEST_FRAGMENT_CHOOSER.choose(mol)
        # Standardize with ChEMBL data curation pipeline. During standardization, the molecule may be broken
        mol = standardize_mol(mol)
        smiles = Chem.MolToSmiles(mol)
    except:
        return None

    # Check if molecule can be parsed by RDKit (in rare cases, the molecule may be broken during standardization)
    if Chem.MolFromSmiles(smiles) is None:
        return None
    
    return smiles

