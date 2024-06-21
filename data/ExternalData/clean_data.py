import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdmolops

#get csv from ExternalData/curated-solubility-dataset.csv
data = pd.read_csv('curated-solubility-dataset.csv')

# Function to clean SMILES and remove salts/ions, keep largest fragment, and ensure neutrality
def clean_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            # Get fragments
            fragments = Chem.GetMolFrags(mol, asMols=True)
            # Check if more than two fragments
            if len(fragments) > 2:
                return None
            # Keep only the largest fragment
            largest_frag = max(fragments, key=lambda m: m.GetNumAtoms(), default=None)
            if largest_frag:
                # Ensure the molecule is neutral (no charges)
                largest_frag = rdmolops.RemoveHs(largest_frag)
                if Chem.GetFormalCharge(largest_frag) == 0:
                    return Chem.MolToSmiles(largest_frag, isomericSmiles=True)
        return None
    except:
        return None

# Function to check if a molecule is valid
def is_valid_molecule(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False

# Apply the cleaning function to the SMILES column
data['Cleaned_SMILES'] = data['SMILES'].apply(clean_smiles)

# Count the number of deleted entries
deleted_entries = data['Cleaned_SMILES'].isnull().sum()

# Filter out invalid molecules
valid_data = data.dropna(subset=['Cleaned_SMILES'])
valid_data = valid_data[valid_data['Cleaned_SMILES'].apply(is_valid_molecule)]

# Save the cleaned dataset
valid_data.to_csv('cleaned_solubility_dataset.csv', index=False)

# Identify and save invalid SMILES
invalid_smiles = data[data['Cleaned_SMILES'].isnull()]
invalid_smiles.to_csv('invalid_smiles.csv', index=False)

# Display the invalid SMILES and count of deleted entries
print("Invalid SMILES after cleaning:")
print(invalid_smiles[['SMILES', 'Cleaned_SMILES']])
print(f"Number of deleted entries: {deleted_entries}")
