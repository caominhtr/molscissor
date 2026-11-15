import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, DataStructs
from rdkit.Chem import AllChem
from os import listdir
import sklearn
from sklearn.metrics import roc_curve, roc_auc_score
from . import sascorer
# import sascorer
from tqdm import tqdm
from collections import defaultdict


def calc_charges(mol):
    positive_charge, negative_charge = 0, 0
    for atom in mol.GetAtoms():
        charge = float(atom.GetFormalCharge())
        positive_charge += max(charge, 0)
        negative_charge -= min(charge, 0)

    return positive_charge, negative_charge


def calc_props_all(mol):
    try:
        # mol = Chem.MolFromSmiles(smiles)

        props = []

        ### MUV properties ###
        # num atoms (incl. H)
        props.append(mol.GetNumAtoms(onlyExplicit=False))
        # num heavy atoms
        props.append(mol.GetNumHeavyAtoms())
        # num boron
        props.append(len(mol.GetSubstructMatches(Chem.MolFromSmarts("[#5]"), maxMatches=mol.GetNumAtoms())))
        # num carbons
        props.append(len(mol.GetSubstructMatches(Chem.MolFromSmarts("[#6]"), maxMatches=mol.GetNumAtoms())))
        # num nitrogen
        props.append(len(mol.GetSubstructMatches(Chem.MolFromSmarts("[#7]"), maxMatches=mol.GetNumAtoms())))
        # num oxygen
        props.append(len(mol.GetSubstructMatches(Chem.MolFromSmarts("[#8]"), maxMatches=mol.GetNumAtoms())))
        # num fluorine
        props.append(len(mol.GetSubstructMatches(Chem.MolFromSmarts("[#9]"), maxMatches=mol.GetNumAtoms())))
        # num phosphorus
        props.append(len(mol.GetSubstructMatches(Chem.MolFromSmarts("[#15]"), maxMatches=mol.GetNumAtoms())))
        # num sulfur
        props.append(len(mol.GetSubstructMatches(Chem.MolFromSmarts("[#16]"), maxMatches=mol.GetNumAtoms())))
        # num chlorine
        props.append(len(mol.GetSubstructMatches(Chem.MolFromSmarts("[#17]"), maxMatches=mol.GetNumAtoms())))
        # num bromine
        props.append(len(mol.GetSubstructMatches(Chem.MolFromSmarts("[#35]"), maxMatches=mol.GetNumAtoms())))
        # num iodine
        props.append(len(mol.GetSubstructMatches(Chem.MolFromSmarts("[#53]"), maxMatches=mol.GetNumAtoms())))

        # logP
        props.append(Chem.Crippen.MolLogP(mol))
        # HBA
        props.append(Chem.rdMolDescriptors.CalcNumLipinskiHBA(mol))
        # HBD
        props.append(Chem.rdMolDescriptors.CalcNumLipinskiHBD(mol))
        # ring count
        props.append(Chem.rdMolDescriptors.CalcNumRings(mol))
        # Stereo centers
        props.append(len(Chem.FindMolChiralCenters(mol, force=True, includeUnassigned=True)))

        ### DEKOIS properties (additional) ###
        # molweight
        props.append(Descriptors.MolWt(mol))
        # aromatic ring count
        props.append(Chem.rdMolDescriptors.CalcNumAromaticRings(mol))
        # rotatable bonds
        props.append(Chem.rdMolDescriptors.CalcNumRotatableBonds(mol))
        # Pos, neg charges
        pos, neg = calc_charges(mol)
        props.append(pos)
        props.append(neg)

        ### DUD-E extended (additional) ###
        # Formal (net) charge
        props.append(Chem.rdmolops.GetFormalCharge(mol))
        # Topological polar surface area
        props.append(Chem.rdMolDescriptors.CalcTPSA(mol))

        ### Additional
        # QED
        props.append(Chem.QED.qed(mol))
        # SA score
        props.append(sascorer.calculateScore(mol))

        return props

    except:
        return [0] * 26


def calc_dataset_props_all(dataset, verbose=False):
    # Calculate mol props for an entire dataset of smiles strings
    results = []
    # for i, smiles in tqdm(enumerate(dataset)):
    for i in range(dataset.shape[0]):
        mol = dataset["Structure"][i]
        props = calc_props_all(mol)
        if props is not None:
            results.append(props)
        # if verbose and i % 1000 == 0:
        #     print("\rProcessed smiles: " + str(i), end='')
    # print("\nDone calculating properties")
    return np.array(results)


def doe_score(actives, decoys):
    all_feat = list(actives) + list(decoys)
    up_p = np.percentile(all_feat, 95, axis=0)
    low_p = np.percentile(all_feat, 5, axis=0)
    norms = up_p - low_p
    for i in range(len(norms)):
        if norms[i] == 0:
            norms[i] = 1.0

    active_norm = [act / norms for act in actives]
    decoy_norm = [dec / norms for dec in decoys]
    all_norm = active_norm + decoy_norm

    active_embed = []
    labels = [1] * (len(active_norm) - 1) + [0] * len(decoy_norm)
    for i, act in enumerate(active_norm):
        comp = list(all_norm)
        del comp[i]
        dists = [
            100 - np.linalg.norm(c - act) for c in comp
        ]  # arbitrary large number to get scores in reverse order
        fpr, tpr, _ = roc_curve(labels, dists)
        fpr = fpr[::]
        tpr = tpr[::]
        a_score = 0
        for i in range(len(fpr) - 1):
            a_score += abs(
                0.5
                * (
                    (tpr[i + 1] + tpr[i]) * (fpr[i + 1] - fpr[i])
                    - (fpr[i + 1] + fpr[i]) * (fpr[i + 1] - fpr[i])
                )
            )
        active_embed.append(a_score)

    # print(np.average(active_embed))
    return np.average(active_embed)





