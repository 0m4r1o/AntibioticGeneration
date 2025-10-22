from molutil import calculateSAS, SmilesToMol, MolToSmiles, lipinski_properties, tanimoto_similarity
import logging
from colorama import Fore, Style, init
from rdkit import RDLogger
from SimilarityMoles import scaffold_aware_similarity
import pandas as pd
from tqdm import tqdm


RDLogger.DisableLog('rdApp.*')  # disable all RDKit log messages
AntibioticPath = 'Results/Constants/antibiotics.smi'
AntibioticNamesPath = 'Results/Constants/names.txt'

init(autoreset=True)  # ensures colors reset after each line

class ColorFormatter(logging.Formatter):
    COLORS = {
        "INFO": Fore.GREEN,
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "CRITICAL": Fore.RED + Style.BRIGHT,
        "DEBUG": Fore.CYAN,
    }

    def format(self, record):
        color = self.COLORS.get(record.levelname, "")
        message = super().format(record)
        return f"{color}{message}{Style.RESET_ALL}"

# Configure logging
handler = logging.StreamHandler()
handler.setFormatter(ColorFormatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"))

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)


def ReadSmiles(path):
	try:
		with open(path,"r") as infile:
			smiles = infile.readlines()
			smiles = [s.strip() for s in smiles]
			return smiles
	except Exception as e:
		logging.error(f"Error Reading File : {e}")
		raise e
	
def GetMetrics(Smiles):
    valid = [s for s in Smiles if SmilesToMol(s)]

    validity = len(valid) / len(Smiles) * 100 if Smiles else 0
    uniqueness = len(set(valid)) / len(valid) * 100 if valid else 0

    return {
        "Validity": round(validity, 2),
        "Uniqueness": round(uniqueness, 2)
    }

def FilterSmilesLip(smiles_list):
    filtered_smiles = []
    for smi in smiles_list:
        mol = SmilesToMol(smi)
        if mol is None:
            continue  # Skip invalid SMILES
        props = lipinski_properties(mol)
        if props["Molecular_Weight"] < 500 and props["LogP"] < 5 and props["SAS"] < 3.5:
            filtered_smiles.append(smi)
    return filtered_smiles

def GetSimilarity(s):
	antibiotics = ReadSmiles(AntibioticPath)
	i, Max = max(((ind, tanimoto_similarity(a, s)) for ind, a in enumerate(antibiotics)),key=lambda t: t[1])
	names = ReadSmiles(AntibioticNamesPath)
	logging.warning(f"Max Similarity with : {names[i]}")
	return Max, names[i]


if __name__ == '__main__':
    generation = '0'
    path = f'Results/Generation/SMILES_gen_{generation}.smi'
    smiles_list = ReadSmiles(path)
    Metrics = GetMetrics(smiles_list)
    filtered_smiles = FilterSmilesLip(smiles_list)
    logging.info(f"Len of original = {len(smiles_list)}\n\t\t\t     Len of filtered = {len(filtered_smiles)}")
    logging.info(Metrics)
    AntibioticPath = 'Results/Antibiotics/antibiotics.smi'
    AntibioticNamePath = 'Results/Antibiotics/names.txt'
    antibiotics = ReadSmiles(AntibioticPath)
    Names = ReadSmiles(AntibioticNamePath)
    Smiles = []
    Scores = []
    AntNames = []
    for s in tqdm(filtered_smiles):
        Max = -1
        i = 0
        for ind, a in enumerate(antibiotics):
            sim = scaffold_aware_similarity(s,a)[0]
            if sim > Max:
                Max = sim
                i = ind
        Smiles.append(s)
        Scores.append(Max)
        AntNames.append(Names[i])
    ScoresDict = {"SMILES":Smiles,"Scores":Scores,"Names":AntNames}
    df = pd.DataFrame(ScoresDict)

    # Save to Excel file
    df.to_excel(f"scores_output_gen_{generation}.xlsx", index=False)

    print("DataFrame saved successfully to scores_output.xlsx!")




# s = 'CC1([C@@H](N2[C@H](S1)[C@@H](C2=O)NC(=O)[C@@H](C3=CC=C(C=C3)O)N)C(=O)O)C'

# m = SmilesToMol(s)
# sas = calculateScore(m)
# logging.info(f"SAS = {sas}")
# filters = lipinski_properties(m)
# logging.info(filters)
# path = r"Results\Constants\antibiotics.smi"
# path = 'Results/Antibiotics/antibiotics.smi'
# smiles_list = ReadSmiles(path)
# Metrics = GetMetrics(smiles_list)
# filtered_smiles = FilterSmilesLip(smiles_list)
# # logging.info(f"Len of original = {len(smiles_list)}\n\t\t\t     Len of filtered = {len(filtered_smiles)}")
# logging.info(Metrics)
# logging.info(f"Scaffold Similarity = {scaffold_aware_similarity(filtered_smiles[0],filtered_smiles[1])}")

# name_path = 'Results/Antibiotics/names.txt'
# names = ReadSmiles(name_path)
# logging.warning(f'{names[0]},{names[1]}')

# Max , name = GetSimilarity(s)
# logging.info(f"Maximum Similarity = {Max} with : {name}")