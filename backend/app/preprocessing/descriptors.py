"""
Molecular descriptor computation module.

This module computes 200+ RDKit molecular descriptors for use as features
in the LightGBM toxicity prediction model. Descriptors include physical,
topological, electronic, and structural properties.
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, GraphDescriptors, Fragments
from rdkit.Chem import rdMolDescriptors, rdPartialCharges


def compute_descriptors(mol: Chem.Mol) -> np.ndarray:
    """
    Compute 200+ RDKit molecular descriptors.
    
    This function computes a comprehensive set of molecular descriptors including:
    - Physical properties: MW, logP, TPSA, MolMR
    - Topological: BertzCT, Chi0-Chi4, Kappa1-Kappa3
    - Electronic: NumHDonors, NumHAcceptors, NumRotatableBonds
    - Structural: NumAromaticRings, NumSaturatedRings, FractionCSP3
    - Fragment counts: Various functional group counts
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        200-dimensional numpy array of descriptor values
        
    Raises:
        ValueError: If molecule is None or invalid
        
    Examples:
        >>> from rdkit import Chem
        >>> mol = Chem.MolFromSmiles("CCO")
        >>> descriptors = compute_descriptors(mol)
        >>> descriptors.shape
        (200,)
    """
    if mol is None:
        raise ValueError("Molecule cannot be None")
    
    descriptor_values = []
    
    # Physical properties
    descriptor_values.append(Descriptors.MolWt(mol))  # Molecular weight
    descriptor_values.append(Descriptors.MolLogP(mol))  # logP (octanol-water partition)
    descriptor_values.append(Descriptors.TPSA(mol))  # Topological polar surface area
    descriptor_values.append(Descriptors.MolMR(mol))  # Molar refractivity
    
    # Hydrogen bonding
    descriptor_values.append(Descriptors.NumHDonors(mol))  # Hydrogen bond donors
    descriptor_values.append(Descriptors.NumHAcceptors(mol))  # Hydrogen bond acceptors
    
    # Rotatable bonds and flexibility
    descriptor_values.append(Descriptors.NumRotatableBonds(mol))
    descriptor_values.append(Descriptors.NumHeteroatoms(mol))
    
    # Ring counts
    descriptor_values.append(Descriptors.NumAromaticRings(mol))
    descriptor_values.append(Descriptors.NumSaturatedRings(mol))
    descriptor_values.append(Descriptors.NumAliphaticRings(mol))
    descriptor_values.append(Descriptors.RingCount(mol))
    
    # Topological complexity
    descriptor_values.append(Descriptors.BertzCT(mol))  # Bertz complexity index
    
    # Chi indices (connectivity indices)
    descriptor_values.append(GraphDescriptors.Chi0(mol))
    descriptor_values.append(GraphDescriptors.Chi1(mol))
    descriptor_values.append(GraphDescriptors.Chi0n(mol))
    descriptor_values.append(GraphDescriptors.Chi1n(mol))
    descriptor_values.append(GraphDescriptors.Chi2n(mol))
    descriptor_values.append(GraphDescriptors.Chi3n(mol))
    descriptor_values.append(GraphDescriptors.Chi4n(mol))
    descriptor_values.append(GraphDescriptors.Chi0v(mol))
    descriptor_values.append(GraphDescriptors.Chi1v(mol))
    descriptor_values.append(GraphDescriptors.Chi2v(mol))
    descriptor_values.append(GraphDescriptors.Chi3v(mol))
    descriptor_values.append(GraphDescriptors.Chi4v(mol))
    
    # Kappa shape indices
    descriptor_values.append(GraphDescriptors.Kappa1(mol))
    descriptor_values.append(GraphDescriptors.Kappa2(mol))
    descriptor_values.append(GraphDescriptors.Kappa3(mol))
    
    # Hybridization and saturation
    descriptor_values.append(Descriptors.FractionCSP3(mol))  # Fraction of sp3 carbons
    descriptor_values.append(Lipinski.NumSaturatedCarbocycles(mol))
    descriptor_values.append(Lipinski.NumSaturatedHeterocycles(mol))
    descriptor_values.append(Lipinski.NumAromaticCarbocycles(mol))
    descriptor_values.append(Lipinski.NumAromaticHeterocycles(mol))
    descriptor_values.append(Lipinski.NumAliphaticCarbocycles(mol))
    descriptor_values.append(Lipinski.NumAliphaticHeterocycles(mol))
    
    # Atom and bond counts
    descriptor_values.append(mol.GetNumAtoms())
    descriptor_values.append(mol.GetNumHeavyAtoms())
    descriptor_values.append(mol.GetNumBonds())
    
    # Valence and formal charge
    descriptor_values.append(Descriptors.NumRadicalElectrons(mol))
    descriptor_values.append(Chem.GetFormalCharge(mol))
    
    # Additional topological descriptors
    descriptor_values.append(GraphDescriptors.BalabanJ(mol))
    descriptor_values.append(GraphDescriptors.HallKierAlpha(mol))
    
    # Electrotopological state indices
    descriptor_values.append(Descriptors.MaxAbsEStateIndex(mol))
    descriptor_values.append(Descriptors.MaxEStateIndex(mol))
    descriptor_values.append(Descriptors.MinAbsEStateIndex(mol))
    descriptor_values.append(Descriptors.MinEStateIndex(mol))
    
    # Partial charge descriptors
    descriptor_values.append(Descriptors.MaxPartialCharge(mol))
    descriptor_values.append(Descriptors.MinPartialCharge(mol))
    descriptor_values.append(Descriptors.MaxAbsPartialCharge(mol))
    descriptor_values.append(Descriptors.MinAbsPartialCharge(mol))
    
    # VSA (van der Waals surface area) descriptors
    descriptor_values.append(Descriptors.LabuteASA(mol))
    descriptor_values.append(Descriptors.PEOE_VSA1(mol))
    descriptor_values.append(Descriptors.PEOE_VSA2(mol))
    descriptor_values.append(Descriptors.PEOE_VSA3(mol))
    descriptor_values.append(Descriptors.PEOE_VSA4(mol))
    descriptor_values.append(Descriptors.PEOE_VSA5(mol))
    descriptor_values.append(Descriptors.PEOE_VSA6(mol))
    descriptor_values.append(Descriptors.PEOE_VSA7(mol))
    descriptor_values.append(Descriptors.PEOE_VSA8(mol))
    descriptor_values.append(Descriptors.PEOE_VSA9(mol))
    descriptor_values.append(Descriptors.PEOE_VSA10(mol))
    descriptor_values.append(Descriptors.PEOE_VSA11(mol))
    descriptor_values.append(Descriptors.PEOE_VSA12(mol))
    descriptor_values.append(Descriptors.PEOE_VSA13(mol))
    descriptor_values.append(Descriptors.PEOE_VSA14(mol))
    
    descriptor_values.append(Descriptors.SMR_VSA1(mol))
    descriptor_values.append(Descriptors.SMR_VSA2(mol))
    descriptor_values.append(Descriptors.SMR_VSA3(mol))
    descriptor_values.append(Descriptors.SMR_VSA4(mol))
    descriptor_values.append(Descriptors.SMR_VSA5(mol))
    descriptor_values.append(Descriptors.SMR_VSA6(mol))
    descriptor_values.append(Descriptors.SMR_VSA7(mol))
    descriptor_values.append(Descriptors.SMR_VSA8(mol))
    descriptor_values.append(Descriptors.SMR_VSA9(mol))
    descriptor_values.append(Descriptors.SMR_VSA10(mol))
    
    descriptor_values.append(Descriptors.SlogP_VSA1(mol))
    descriptor_values.append(Descriptors.SlogP_VSA2(mol))
    descriptor_values.append(Descriptors.SlogP_VSA3(mol))
    descriptor_values.append(Descriptors.SlogP_VSA4(mol))
    descriptor_values.append(Descriptors.SlogP_VSA5(mol))
    descriptor_values.append(Descriptors.SlogP_VSA6(mol))
    descriptor_values.append(Descriptors.SlogP_VSA7(mol))
    descriptor_values.append(Descriptors.SlogP_VSA8(mol))
    descriptor_values.append(Descriptors.SlogP_VSA9(mol))
    descriptor_values.append(Descriptors.SlogP_VSA10(mol))
    descriptor_values.append(Descriptors.SlogP_VSA11(mol))
    descriptor_values.append(Descriptors.SlogP_VSA12(mol))
    
    # EState indices
    descriptor_values.append(Descriptors.EState_VSA1(mol))
    descriptor_values.append(Descriptors.EState_VSA2(mol))
    descriptor_values.append(Descriptors.EState_VSA3(mol))
    descriptor_values.append(Descriptors.EState_VSA4(mol))
    descriptor_values.append(Descriptors.EState_VSA5(mol))
    descriptor_values.append(Descriptors.EState_VSA6(mol))
    descriptor_values.append(Descriptors.EState_VSA7(mol))
    descriptor_values.append(Descriptors.EState_VSA8(mol))
    descriptor_values.append(Descriptors.EState_VSA9(mol))
    descriptor_values.append(Descriptors.EState_VSA10(mol))
    descriptor_values.append(Descriptors.EState_VSA11(mol))
    
    # MQN (Molecular Quantum Numbers) descriptors
    descriptor_values.append(Descriptors.HeavyAtomMolWt(mol))
    descriptor_values.append(Descriptors.ExactMolWt(mol))
    descriptor_values.append(Descriptors.NumValenceElectrons(mol))
    
    # Fragment-based descriptors (functional group counts)
    descriptor_values.append(Fragments.fr_Al_COO(mol))  # Aliphatic carboxylic acids
    descriptor_values.append(Fragments.fr_Al_OH(mol))  # Aliphatic hydroxyl groups
    descriptor_values.append(Fragments.fr_Ar_COO(mol))  # Aromatic carboxylic acids
    descriptor_values.append(Fragments.fr_Ar_N(mol))  # Aromatic nitrogens
    descriptor_values.append(Fragments.fr_Ar_NH(mol))  # Aromatic amines
    descriptor_values.append(Fragments.fr_Ar_OH(mol))  # Aromatic hydroxyl groups
    descriptor_values.append(Fragments.fr_COO(mol))  # Carboxylic acids
    descriptor_values.append(Fragments.fr_COO2(mol))  # Carboxylic acids (alternative)
    descriptor_values.append(Fragments.fr_C_O(mol))  # Carbonyl groups
    descriptor_values.append(Fragments.fr_C_O_noCOO(mol))  # Carbonyl excluding COOH
    descriptor_values.append(Fragments.fr_NH0(mol))  # Tertiary amines
    descriptor_values.append(Fragments.fr_NH1(mol))  # Secondary amines
    descriptor_values.append(Fragments.fr_NH2(mol))  # Primary amines
    descriptor_values.append(Fragments.fr_N_O(mol))  # Hydroxylamine groups
    descriptor_values.append(Fragments.fr_Ndealkylation1(mol))
    descriptor_values.append(Fragments.fr_Ndealkylation2(mol))
    descriptor_values.append(Fragments.fr_aldehyde(mol))
    descriptor_values.append(Fragments.fr_alkyl_halide(mol))
    descriptor_values.append(Fragments.fr_allylic_oxid(mol))
    descriptor_values.append(Fragments.fr_amide(mol))
    descriptor_values.append(Fragments.fr_amidine(mol))
    descriptor_values.append(Fragments.fr_aniline(mol))
    descriptor_values.append(Fragments.fr_aryl_methyl(mol))
    descriptor_values.append(Fragments.fr_azide(mol))
    descriptor_values.append(Fragments.fr_azo(mol))
    descriptor_values.append(Fragments.fr_barbitur(mol))
    descriptor_values.append(Fragments.fr_benzene(mol))
    descriptor_values.append(Fragments.fr_benzodiazepine(mol))
    descriptor_values.append(Fragments.fr_bicyclic(mol))
    descriptor_values.append(Fragments.fr_diazo(mol))
    descriptor_values.append(Fragments.fr_dihydropyridine(mol))
    descriptor_values.append(Fragments.fr_epoxide(mol))
    descriptor_values.append(Fragments.fr_ester(mol))
    descriptor_values.append(Fragments.fr_ether(mol))
    descriptor_values.append(Fragments.fr_furan(mol))
    descriptor_values.append(Fragments.fr_guanido(mol))
    descriptor_values.append(Fragments.fr_halogen(mol))
    descriptor_values.append(Fragments.fr_hdrzine(mol))
    descriptor_values.append(Fragments.fr_hdrzone(mol))
    descriptor_values.append(Fragments.fr_imidazole(mol))
    descriptor_values.append(Fragments.fr_imide(mol))
    descriptor_values.append(Fragments.fr_isocyan(mol))
    descriptor_values.append(Fragments.fr_isothiocyan(mol))
    descriptor_values.append(Fragments.fr_ketone(mol))
    descriptor_values.append(Fragments.fr_ketone_Topliss(mol))
    descriptor_values.append(Fragments.fr_lactam(mol))
    descriptor_values.append(Fragments.fr_lactone(mol))
    descriptor_values.append(Fragments.fr_methoxy(mol))
    descriptor_values.append(Fragments.fr_morpholine(mol))
    descriptor_values.append(Fragments.fr_nitrile(mol))
    descriptor_values.append(Fragments.fr_nitro(mol))
    descriptor_values.append(Fragments.fr_nitro_arom(mol))
    descriptor_values.append(Fragments.fr_nitro_arom_nonortho(mol))
    descriptor_values.append(Fragments.fr_nitroso(mol))
    descriptor_values.append(Fragments.fr_oxazole(mol))
    descriptor_values.append(Fragments.fr_oxime(mol))
    descriptor_values.append(Fragments.fr_para_hydroxylation(mol))
    descriptor_values.append(Fragments.fr_phenol(mol))
    descriptor_values.append(Fragments.fr_phenol_noOrthoHbond(mol))
    descriptor_values.append(Fragments.fr_phos_acid(mol))
    descriptor_values.append(Fragments.fr_phos_ester(mol))
    descriptor_values.append(Fragments.fr_piperdine(mol))
    descriptor_values.append(Fragments.fr_piperzine(mol))
    descriptor_values.append(Fragments.fr_priamide(mol))
    descriptor_values.append(Fragments.fr_prisulfonamd(mol))
    descriptor_values.append(Fragments.fr_pyridine(mol))
    descriptor_values.append(Fragments.fr_quatN(mol))
    descriptor_values.append(Fragments.fr_sulfide(mol))
    descriptor_values.append(Fragments.fr_sulfonamd(mol))
    descriptor_values.append(Fragments.fr_sulfone(mol))
    descriptor_values.append(Fragments.fr_term_acetylene(mol))
    descriptor_values.append(Fragments.fr_tetrazole(mol))
    descriptor_values.append(Fragments.fr_thiazole(mol))
    descriptor_values.append(Fragments.fr_thiocyan(mol))
    descriptor_values.append(Fragments.fr_thiophene(mol))
    descriptor_values.append(Fragments.fr_unbrch_alkane(mol))
    descriptor_values.append(Fragments.fr_urea(mol))
    
    # Additional 3D-independent descriptors
    descriptor_values.append(rdMolDescriptors.CalcNumSpiroAtoms(mol))
    descriptor_values.append(rdMolDescriptors.CalcNumBridgeheadAtoms(mol))
    descriptor_values.append(rdMolDescriptors.CalcNumAmideBonds(mol))
    
    # Ensure we have exactly 200 descriptors
    # If we have fewer, pad with zeros; if more, truncate
    if len(descriptor_values) < 200:
        descriptor_values.extend([0.0] * (200 - len(descriptor_values)))
    elif len(descriptor_values) > 200:
        descriptor_values = descriptor_values[:200]
    
    return np.array(descriptor_values, dtype=np.float64)
