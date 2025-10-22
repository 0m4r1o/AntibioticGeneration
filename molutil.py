from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Descriptors import ExactMolWt
import pickle
from rdkit.Chem import Descriptors
import math
from collections import defaultdict
from rdkit.Chem import AllChem, DataStructs

import os.path as op

_fscores = None


def readFragmentScores(name='fpscores'):
    import gzip
    global _fscores
    # generate the full path filename:
    if name == "fpscores":
        name = op.join(op.dirname(__file__), name)
    data = pickle.load(gzip.open('%s.pkl.gz' % name))
    outDict = {}
    for i in data:
        for j in range(1, len(i)):
            outDict[i[j]] = float(i[0])
    _fscores = outDict


def numBridgeheadsAndSpiro(mol, ri=None):
    nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    nBridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    return nBridgehead, nSpiro


def calculateSAS(m):
    if _fscores is None:
        readFragmentScores()

    # fragment score
    fp = rdMolDescriptors.GetMorganFingerprint(m,
                                               2)  # <- 2 is the *radius* of the circular fingerprint
    fps = fp.GetNonzeroElements()
    score1 = 0.
    nf = 0
    for bitId, v in fps.items():
        nf += v
        sfp = bitId
        score1 += _fscores.get(sfp, -4) * v
    score1 /= nf

    # features score
    nAtoms = m.GetNumAtoms()
    nChiralCenters = len(Chem.FindMolChiralCenters(m, includeUnassigned=True))
    ri = m.GetRingInfo()
    nBridgeheads, nSpiro = numBridgeheadsAndSpiro(m, ri)
    nMacrocycles = 0
    for x in ri.AtomRings():
        if len(x) > 8:
            nMacrocycles += 1

    sizePenalty = nAtoms**1.005 - nAtoms
    stereoPenalty = math.log10(nChiralCenters + 1)
    spiroPenalty = math.log10(nSpiro + 1)
    bridgePenalty = math.log10(nBridgeheads + 1)
    macrocyclePenalty = 0.
    # ---------------------------------------
    # This differs from the paper, which defines:
    #  macrocyclePenalty = math.log10(nMacrocycles+1)
    # This form generates better results when 2 or more macrocycles are present
    if nMacrocycles > 0:
        macrocyclePenalty = math.log10(2)

    score2 = 0. - sizePenalty - stereoPenalty - spiroPenalty - bridgePenalty - macrocyclePenalty

    # correction for the fingerprint density
    # not in the original publication, added in version 1.1
    # to make highly symmetrical molecules easier to synthetise
    score3 = 0.
    if nAtoms > len(fps):
        score3 = math.log(float(nAtoms) / len(fps)) * .5

    sascore = score1 + score2 + score3

    # need to transform "raw" value into scale between 1 and 10
    min = -4.0
    max = 2.5
    sascore = 11. - (sascore - min + 1) / (max - min) * 9.
    # smooth the 10-end
    if sascore > 8.:
        sascore = 8. + math.log(sascore + 1. - 9.)
    if sascore > 10.:
        sascore = 10.0
    elif sascore < 1.:
        sascore = 1.0
    return sascore


def SmilesToMol(S):
    M = Chem.MolFromSmiles(S)
    # if M is None:
    #     print(f"❌ Invalid SMILES: {S}")
    return M

def MolToSmiles(mol, canonical=True):
    """
    Convert an RDKit Mol object to a SMILES string.
    If canonical=True, returns the canonical (standardized) SMILES.
    """
    if mol is None:
        # print("❌ Cannot convert: Mol object is None.")
        return None
    
    smiles = Chem.MolToSmiles(mol, canonical=canonical)
    return smiles



def lipinski_properties(m):
    """
    Calculate Lipinski's Rule of Five properties from a SMILES string.
    Returns a dictionary with MW, LogP, HBD, HBA, and rule violations.
    """
    mw = Descriptors.MolWt(m)
    logp = Descriptors.MolLogP(m)
    hbd = Descriptors.NumHDonors(m)
    hba = Descriptors.NumHAcceptors(m)
    sas = calculateSAS(m)

    # Check rule violations
    violations = sum([
        mw > 500,
        logp > 5,
        hbd > 5,
        hba > 10,
        sas > 3.5
    ])

    return {
        "Molecular_Weight": mw,
        "LogP": logp,
        "H_Bond_Donors": hbd,
        "H_Bond_Acceptors": hba,
        "SAS":sas,
        "Violations": violations,
        "Lipinski_Compliant": violations <= 1
    }
def tanimoto_similarity(s1: str, s2: str) -> float:
    """
    Calculate the Tanimoto similarity between two SMILES strings.
    Returns a float between 0.0 (no similarity) and 1.0 (identical).
    """
    try:
        mol1 = Chem.MolFromSmiles(s1)
        mol2 = Chem.MolFromSmiles(s2)
        if not mol1 or not mol2:
            raise ValueError("Invalid SMILES input")

        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=2, nBits=2048)

        score = DataStructs.TanimotoSimilarity(fp1, fp2)
        return round(score, 4)  # 4 decimal precision
    except Exception as e:
        print(f"⚠️ Error computing Tanimoto similarity: {e}")
        return 0.0
