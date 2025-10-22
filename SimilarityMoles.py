from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as rdMD
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdFMCS, DataStructs

def mol_from_smiles(s):
    return Chem.MolFromSmiles(s) if s else None

def ecfp(mol, radius=2, nBits=2048):
    return rdMD.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)

def atompair_fp(mol, nBits=2048):
    return rdMD.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=nBits)

def tanimoto(fp1, fp2):
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def get_scaffold(mol):
    scaf = MurckoScaffold.GetScaffoldForMol(mol)
    # sanitize in case of weird valences after side-chain stripping
    Chem.SanitizeMol(scaf, catchErrors=True)
    return scaf

def mcs_similarity_on_scaffolds(scafA, scafB):
    if scafA is None or scafB is None:
        return 0.0
    res = rdFMCS.FindMCS([scafA, scafB],
                         ringMatchesRingOnly=True,
                         completeRingsOnly=True,
                         matchValences=True,
                         atomCompare=rdFMCS.AtomCompare.CompareElements,
                         bondCompare=rdFMCS.BondCompare.CompareOrder,
                         timeout=5)
    if res.canceled or not res.smartsString:
        return 0.0
    mcs = Chem.MolFromSmarts(res.smartsString)
    a = scafA.GetNumAtoms()
    b = scafB.GetNumAtoms()
    m = mcs.GetNumAtoms()
    if a == 0 or b == 0:
        return 0.0
    # Sørensen–Dice style normalization
    return (2.0 * m) / (a + b)

def scaffold_aware_similarity(smilesA, smilesB):
    mA = mol_from_smiles(smilesA); mB = mol_from_smiles(smilesB)
    if mA is None or mB is None:
        return 0.0

    fpA_full = ecfp(mA); fpB_full = ecfp(mB)
    T_full = tanimoto(fpA_full, fpB_full)

    scA = get_scaffold(mA); scB = get_scaffold(mB)
    # If either scaffold is empty (acyclic), fall back to BRICS later if you wish.
    if scA is None or scB is None or scA.GetNumAtoms()==0 or scB.GetNumAtoms()==0:
        T_scaf = 0.0; MCS_scaf = 0.0
    else:
        T_scaf = tanimoto(ecfp(scA), ecfp(scB))
        MCS_scaf = mcs_similarity_on_scaffolds(scA, scB)

    # Optional AtomPair on scaffolds
    T_pair_scaf = 0.0
    if scA and scB and scA.GetNumAtoms()>0 and scB.GetNumAtoms()>0:
        T_pair_scaf = tanimoto(atompair_fp(scA), atompair_fp(scB))

    # Composite
    S = 0.45*T_full + 0.25*T_scaf + 0.20*MCS_scaf + 0.10*T_pair_scaf

    # Same-scaffold micro-boost
    if Chem.MolToSmiles(scA, True) == Chem.MolToSmiles(scB, True) and scA.GetNumAtoms()>0:
        S = min(1.0, S + 0.05)

    # Different-scaffold penalty (guards decoration-only matches)
    if (MCS_scaf < 0.3) and (T_scaf < 0.3):
        S = min(S, 0.60)

    return S, {"T_full":T_full, "T_scaf":T_scaf, "MCS_scaf":MCS_scaf, "T_pair_scaf":T_pair_scaf}

if __name__ == '__main__':
    s1 = "O=C(O)CCCCCCCOc1ccc(-c2cccc(Nc3cc(-c4ccccc4)nc4ccccc34)c2)cc1"
    s2 = "O=C(NC1CCC(CCN2CCC(c3coc4ccccc34)CC2)CC1)c1ccc(N2CCOCC2)cc1"
    print(scaffold_aware_similarity(s1,s2))