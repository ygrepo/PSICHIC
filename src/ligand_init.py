from __future__ import annotations

from rdkit import Chem
from rdkit.Chem.rdchem import BondType, HybridizationType

from rdkit.Chem import ChemicalFeatures
from rdkit.Chem import rdmolops

from rdkit import RDConfig
import os
import numpy as np

import torch
from torch import Tensor

## FILE from pytorch geometric version 2.
from itertools import chain
from typing import Any, Tuple, Union, Dict, List, Optional

from scipy.sparse.csgraph import minimum_spanning_tree


from torch_geometric.utils import (
    from_scipy_sparse_matrix,
    to_scipy_sparse_matrix,
    to_undirected,
)

from pathlib import Path

from tqdm import tqdm


from src.utils import get_logger

logger = get_logger(__name__)

fdefName = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
logger.info(fdefName)
factory = ChemicalFeatures.BuildFeatureFactory(fdefName)


def one_of_k_encoding(x: int, allowable_set: list) -> list:
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x: int, allowable_set: list) -> list:
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom: Chem.rdchem.Atom) -> np.ndarray:
    encoding = one_of_k_encoding(
        atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ) + one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    encoding += one_of_k_encoding_unk(
        atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    )

    encoding += one_of_k_encoding_unk(
        atom.GetHybridization(),
        [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
            "other",
        ],
    )

    encoding += [atom.GetIsAromatic()]

    # Chirality: only use CIP code if present
    if atom.HasProp("_CIPCode"):
        cip = atom.GetProp("_CIPCode")
        encoding += one_of_k_encoding_unk(cip, ["R", "S"])
    else:
        encoding += [0, 0]

    encoding += [atom.HasProp("_ChiralityPossible")]

    return np.array(encoding)


def tree_decomposition(
    mol: Chem.rdchem.Mol,
    return_vocab: bool = False,
) -> Union[Tuple[Tensor, Tensor, int], Tuple[Tensor, Tensor, int, Tensor]]:
    r"""The tree decomposition algorithm of molecules from the
    `"Junction Tree Variational Autoencoder for Molecular Graph Generation"
    <https://arxiv.org/abs/1802.04364>`_ paper.
    Returns the graph connectivity of the junction tree, the assignment
    mapping of each atom to the clique in the junction tree, and the number
    of cliques.

    Args:
        mol (rdkit.Chem.Mol): An :obj:`rdkit` molecule.
        return_vocab (bool, optional): If set to :obj:`True`, will return an
            identifier for each clique (ring, bond, bridged compounds, single).
            (default: :obj:`False`)

    :rtype: :obj:`(LongTensor, LongTensor, int)` if :obj:`return_vocab` is
        :obj:`False`, else :obj:`(LongTensor, LongTensor, int, LongTensor)`
    """

    # Cliques = rings and bonds.
    cliques = [list(x) for x in Chem.GetSymmSSSR(mol)]
    xs = [0] * len(cliques)
    for bond in mol.GetBonds():
        if not bond.IsInRing():
            cliques.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
            xs.append(1)

    # Generate `atom2clique` mappings.
    atom2clique = [[] for i in range(mol.GetNumAtoms())]
    for c in range(len(cliques)):
        for atom in cliques[c]:
            atom2clique[atom].append(c)

    # Merge rings that share more than 2 atoms as they form bridged compounds.
    for c1 in range(len(cliques)):
        for atom in cliques[c1]:
            for c2 in atom2clique[atom]:
                if c1 >= c2 or len(cliques[c1]) <= 2 or len(cliques[c2]) <= 2:
                    continue
                if len(set(cliques[c1]) & set(cliques[c2])) > 2:
                    cliques[c1] = set(cliques[c1]) | set(cliques[c2])
                    xs[c1] = 2
                    cliques[c2] = []
                    xs[c2] = -1
    cliques = [c for c in cliques if len(c) > 0]
    xs = [x for x in xs if x >= 0]

    # Update `atom2clique` mappings.
    atom2clique = [[] for i in range(mol.GetNumAtoms())]
    for c in range(len(cliques)):
        for atom in cliques[c]:
            atom2clique[atom].append(c)

    # Add singleton cliques in case there are more than 2 intersecting
    # cliques. We further compute the "initial" clique graph.
    edges = {}
    for atom in range(mol.GetNumAtoms()):
        cs = atom2clique[atom]
        if len(cs) <= 1:
            continue

        # Number of bond clusters that the atom lies in.
        bonds = [c for c in cs if len(cliques[c]) == 2]
        # Number of ring clusters that the atom lies in.
        rings = [c for c in cs if len(cliques[c]) > 4]

        if len(bonds) > 2 or (len(bonds) == 2 and len(cs) > 2):
            cliques.append([atom])
            xs.append(3)
            c2 = len(cliques) - 1
            for c1 in cs:
                edges[(c1, c2)] = 1

        elif len(rings) > 2:
            cliques.append([atom])
            xs.append(3)
            c2 = len(cliques) - 1
            for c1 in cs:
                edges[(c1, c2)] = 99

        else:
            for i in range(len(cs)):
                for j in range(i + 1, len(cs)):
                    c1, c2 = cs[i], cs[j]
                    count = len(set(cliques[c1]) & set(cliques[c2]))
                    edges[(c1, c2)] = min(count, edges.get((c1, c2), 99))

    # Update `atom2clique` mappings.
    atom2clique = [[] for i in range(mol.GetNumAtoms())]
    for c in range(len(cliques)):
        for atom in cliques[c]:
            atom2clique[atom].append(c)

    if len(edges) > 0:
        edge_index_T, weight = zip(*edges.items())
        edge_index = torch.tensor(edge_index_T).t()
        inv_weight = 100 - torch.tensor(weight)
        graph = to_scipy_sparse_matrix(edge_index, inv_weight, len(cliques))
        junc_tree = minimum_spanning_tree(graph)
        edge_index, _ = from_scipy_sparse_matrix(junc_tree)
        edge_index = to_undirected(edge_index, num_nodes=len(cliques))
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    rows = [[i] * len(atom2clique[i]) for i in range(mol.GetNumAtoms())]
    row = torch.tensor(list(chain.from_iterable(rows)))
    col = torch.tensor(list(chain.from_iterable(atom2clique)))
    atom2clique = torch.stack([row, col], dim=0).to(torch.long)

    if return_vocab:
        vocab = torch.tensor(xs, dtype=torch.long)
        return edge_index, atom2clique, len(cliques), vocab
    else:
        return edge_index, atom2clique, len(cliques)


###


def smiles2graph(m_str: str) -> dict | None:
    """
    Convert a molecule SMILES to a graph dict.
    Returns None for reaction strings (contains '>>') or invalid SMILES.
    """
    dropped = 0
    if m_str is None:
        return None

    m_str = str(m_str).strip()
    if m_str == "":
        return None

    # Drop reaction SMARTS / reaction SMILES
    if ">>" in m_str:
        logger.warning(f"Reaction string detected, dropping: {m_str}")
        dropped += 1
        return None

    mgd = MoleculeGraphDataset(halogen_detail=False)

    mol = Chem.MolFromSmiles(m_str)
    if mol is None:
        logger.warning(f"Invalid SMILES string, dropping: {m_str}")
        dropped += 1
        return None

    # Optional: ensure chemistry is sanitized (MolFromSmiles usually sanitizes, but be explicit)
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        logger.warning(f"Sanitization failed, dropping: {m_str}")
        dropped += 1
        return None

    logger.info(f"dropped: {dropped}")
    atom_feature, bond_feature = mgd.featurize(mol, "atom_full_feature")
    atom_idx, _ = mgd.featurize(mol, "atom_type")
    tree = mgd.junction_tree(mol)

    out_dict = {
        "smiles": m_str,
        "atom_feature": torch.as_tensor(atom_feature),
        "atom_types": "|".join([a.GetSymbol() for a in mol.GetAtoms()]),
        "atom_idx": torch.as_tensor(atom_idx),
        "bond_feature": torch.as_tensor(bond_feature),
    }

    # Keep tree tensors as returned by junction_tree (assumed torch tensors already)
    out_dict.update(tree)
    return out_dict


# def smiles2graph(m_str: str) -> dict:
#     mgd = MoleculeGraphDataset(halogen_detail=False)
#     mol = Chem.MolFromSmiles(m_str)
#     # mol = get_mol(m_str)
#     atom_feature, bond_feature = mgd.featurize(mol, "atom_full_feature")
#     atom_idx, _ = mgd.featurize(mol, "atom_type")
#     tree = mgd.junction_tree(mol)

#     out_dict = {
#         "smiles": m_str,
#         "atom_feature": torch.tensor(atom_feature),  # .to(torch.int8),
#         "atom_types": "|".join([i.GetSymbol() for i in mol.GetAtoms()]),
#         "atom_idx": torch.tensor(atom_idx),  # .to(torch.int8),
#         "bond_feature": torch.tensor(bond_feature),  # .to(torch.int8),
#     }
#     tree["tree_edge_index"] = tree["tree_edge_index"]  # .to(torch.int8)
#     tree["atom2clique_index"] = tree["atom2clique_index"]  # .to(torch.int8)
#     tree["x_clique"] = tree["x_clique"]  # .to(torch.int8)

#     out_dict.update(tree)

#     return out_dict


####


class MoleculeGraphDataset:
    """
    Utility class to construct graph-based representations and per-atom features
    from RDKit molecules for downstream machine learning models (e.g., GNNs).

    Responsibilities
    ----------------
    This class encapsulates:

    1. Atom type encoding / classes
       - Maps atomic numbers into a small set of atom classes (C, N, O, S, halogens, metals, etc.).
       - Supports either a coarse "halogen" group or detailed per-halogen splitting (F, Cl, Br, I).
       - Produces:
           * `ATOM_CODES`: dict[atomic_number] -> class index
           * `FEATURE_NAMES`: ordered list of class names, one per index

    2. Per-atom features
       - `atom_feature_extract`: hand-crafted features (degree, valence, hybridization, aromaticity, etc.).
       - `mol_feature`: aggregates per-atom features into (num_atoms, feat_dim) matrix.
       - `mol_extra_feature`: pharmacophore-style features (Donor, Acceptor, Hydrophobe, LumpedHydrophobe),
         using RDKit feature factory.
       - `mol_simplified_feature` / `mol_sequence_simplified_feature`: compact atom-type encodings
         (integer labels or one-hot over a fixed element list).
       - `mol_full_feature`: “full” feature vector using an external `atom_features` helper.

    3. Bond / edge features
       - `bond_feature`: constructs an atom–atom adjacency matrix with integer bond-type codes
         based on RDKit `BondType` (single, double, triple, aromatic).

    4. Junction tree representation
       - `junction_tree`: wrapper around `tree_decomposition` to generate:
           * clique tree edge index,
           * mapping from atoms to cliques,
           * per-clique features (x_clique),
           * and a fallback for “weird” molecules (no cliques detected).

    5. Unified featurization interface
       - `featurize`: given an RDKit Mol and a requested featurization type
         ("atom_type", "detailed_atom_type", "atom_feature", or "atom_full_feature"),
         returns:
           * atom_feature: (num_atoms, feat_dim)
           * bond_feature: (num_atoms, num_atoms) adjacency with bond-type codes.

    Parameters
    ----------
    atom_classes : list[tuple[list[int] | int, str]] or None
        Custom atom classes as a list of (atomic_number_or_list, name).
        If None, a default set of classes is used:
            - B, C, N, O, P, S, Se,
            - halogens (either grouped or detailed per element),
            - metals (predefined large list by atomic number).
    halogen_detail : bool
        If True, splits halogens into four distinct classes (F, Cl, Br, I).
        If False, groups them into a single "halogen" class.
    save_path : pathlib.Path or None
        Optional path for saving precomputed dataset artifacts (not used inside
        this class directly, but stored for external use).

    Example
    -------
    >>> from rdkit import Chem
    >>> ds = MoleculeGraphDataset()
    >>> mol = Chem.MolFromSmiles("CCO")
    >>> atom_feat, bond_feat = ds.featurize(mol, type="atom_feature")
    >>> atom_feat.shape  # (num_atoms, feat_dim)
    >>> bond_feat.shape  # (num_atoms, num_atoms)
    """

    def __init__(
        self,
        atom_classes: Optional[List[Tuple[int | List[int], str]]] = None,
        halogen_detail: bool = False,
        save_path: Optional[Path] = None,
    ) -> None:
        # Map RDKit atomic numbers to integer codes [0 .. NUM_ATOM_CLASSES-1]
        self.ATOM_CODES: Dict[int, int] = {}

        # Default atom-class definitions if none are provided
        if atom_classes is None:
            # List of metallic elements by atomic number
            metals: List[int] = (
                [3, 4, 11, 12, 13]
                + list(range(19, 32))
                + list(range(37, 51))
                + list(range(55, 84))
                + list(range(87, 104))
            )

            # Human-readable feature names for each atom class
            self.FEATURE_NAMES: List[str] = []

            if halogen_detail:
                # Each halogen is its own class: F, Cl, Br, I
                atom_classes = [
                    (5, "B"),
                    (6, "C"),
                    (7, "N"),
                    (8, "O"),
                    (15, "P"),
                    (16, "S"),
                    (34, "Se"),
                    # halogens
                    (9, "F"),
                    (17, "Cl"),
                    (35, "Br"),
                    (53, "I"),
                    # metals
                    (metals, "metal"),
                ]
            else:
                # All halogens grouped into a single "halogen" atom class
                atom_classes = [
                    (5, "B"),
                    (6, "C"),
                    (7, "N"),
                    (8, "O"),
                    (15, "P"),
                    (16, "S"),
                    (34, "Se"),
                    # halogens grouped
                    ([9, 17, 35, 53], "halogen"),
                    # metals
                    (metals, "metal"),
                ]
        else:
            # If atom_classes is provided, we still need FEATURE_NAMES initialized
            self.FEATURE_NAMES: List[str] = []

        # Number of distinct atom classes
        self.NUM_ATOM_CLASSES: int = len(atom_classes)

        # Populate ATOM_CODES and FEATURE_NAMES from atom_classes
        for code, (atom, name) in enumerate(atom_classes):
            if isinstance(atom, list):
                # Map each atomic number in the list to the same class index
                for a in atom:
                    self.ATOM_CODES[a] = code
            else:
                self.ATOM_CODES[atom] = code
            self.FEATURE_NAMES.append(name)

        # Extra per-atom RDKit pharmacophore-style features to extract
        # These correspond to RDKit FeatureFactory "families".
        self.feat_types: List[str] = [
            "Donor",
            "Acceptor",
            "Hydrophobe",
            "LumpedHydrophobe",
        ]

        # Bond feature dictionary: RDKit BondType -> integer code
        # This will be used to fill the adjacency matrix in `bond_feature`.
        self.edge_dict: Dict[BondType, int] = {
            BondType.SINGLE: 1,
            BondType.DOUBLE: 2,
            BondType.TRIPLE: 3,
            BondType.AROMATIC: 4,
            BondType.UNSPECIFIED: 1,
        }

        # Optional path where dataset artifacts / processed graphs could be stored
        self.save_path: Optional[Path] = save_path

    def hybridization_onehot(self, hybrid_type: rdchem.HybridizationType) -> np.ndarray:
        """
        One-hot encode the RDKit hybridization type.

        Parameters
        ----------
        hybrid_type : rdchem.HybridizationType
            RDKit hybridization enumeration.

        Returns
        -------
        encoding : np.ndarray
            One-hot vector over {S, SP, SP2, SP3, SP3D, SP3D2}.
            If an unknown hybridization type is encountered, returns all zeros
            and logs a warning.
        """
        hybrid_type_str = str(hybrid_type)
        types: Dict[str, int] = {
            "S": 0,
            "SP": 1,
            "SP2": 2,
            "SP3": 3,
            "SP3D": 4,
            "SP3D2": 5,
        }

        encoding = np.zeros(len(types), dtype=float)
        try:
            encoding[types[hybrid_type_str]] = 1.0
        except Exception as e:
            logger.warning(f"Hybridization error: {e}")
            # leave encoding as all zeros
            pass
        return encoding

    def encode_num(self, atomic_num: int) -> np.ndarray:
        """
        Encode atom type into a binary vector over the predefined atom classes.

        If the atomic number is not included in `atom_classes` (and thus not
        in `ATOM_CODES`), its encoding is an all-zeros vector.

        Parameters
        ----------
        atomic_num : int
            Atomic number of the element (e.g., 6 for carbon).

        Returns
        -------
        encoding : np.ndarray
            Binary vector of length `NUM_ATOM_CLASSES`. One-hot if atom is known,
            all zeros otherwise.
        """

        if not isinstance(atomic_num, int):
            raise TypeError(
                "Atomic number must be int, %s was given" % type(atomic_num)
            )

        encoding = np.zeros(self.NUM_ATOM_CLASSES, dtype=float)
        try:
            encoding[self.ATOM_CODES[atomic_num]] = 1.0
        except Exception as e:
            logger.warning(f"Atom encoding error: {e}")
            # leave encoding as all zeros
            pass
        return encoding

    def atom_feature_extract(self, atom: Chem.Atom) -> List[float]:
        """
        Extract a set of hand-crafted features from a single RDKit Atom.

        Feature layout
        --------------
            0  - Atom degree                      (atom.GetDegree())
            1  - Total valence                    (atom.GetTotalValence())
            2–7  - Hybridization one-hot S/SP/SP2/SP3/SP3D/SP3D2
            8  - Number of radical electrons      (atom.GetNumRadicalElectrons())
            9  - Formal charge                    (atom.GetFormalCharge())
            10 - Aromatic flag                    (atom.GetIsAromatic())
            11 - Ring membership flag             (atom.IsInRing())
            12 - (optional) Atom-class one-hot    (encode_num()) [currently commented]

        Returns
        -------
        feat : list
            List of numeric features for the given atom.
        """
        feat: List[float] = []

        # Basic graph / valence descriptors
        feat.append(float(atom.GetDegree()))
        feat.append(float(atom.GetTotalValence()))

        # Hybridization type one-hot
        feat += self.hybridization_onehot(atom.GetHybridization()).tolist()

        # Electronic and structural properties
        feat.append(float(atom.GetNumRadicalElectrons()))
        feat.append(float(atom.GetFormalCharge()))
        feat.append(float(int(atom.GetIsAromatic())))
        feat.append(float(int(atom.IsInRing())))

        # Optional class encoding (commented out in this implementation)
        # feat += self.encode_num(atom.GetAtomicNum()).tolist()

        return feat

    def mol_feature(self, mol: Chem.Mol) -> np.ndarray:
        """
        Construct a per-atom feature matrix using `atom_feature_extract`.

        Parameters
        ----------
        mol : Chem.Mol
            RDKit molecule.

        Returns
        -------
        feature : np.ndarray
            Array of shape (num_atoms, feat_dim), where feat_dim is defined
            by `atom_feature_extract`.
        """
        atom_ids: List[int] = []
        atom_feats: List[List[float]] = []

        for atom in mol.GetAtoms():
            atom_ids.append(atom.GetIdx())
            feat = self.atom_feature_extract(atom)
            atom_feats.append(feat)

        # Sort by atom index to ensure consistent ordering, then take only features
        feature = np.array(
            list(zip(*sorted(zip(atom_ids, atom_feats))))[-1], dtype=float
        )

        return feature

    def mol_extra_feature(self, mol: Chem.Mol) -> np.ndarray:
        """
        Extract pharmacophore-style atom features using RDKit FeatureFactory.

        For each atom, this marks whether it belongs to one of:
            - Donor
            - Acceptor
            - Hydrophobe
            - LumpedHydrophobe

        Parameters
        ----------
        mol : Chem.Mol

        Returns
        -------
        feature : np.ndarray
            Binary matrix of shape (num_atoms, len(self.feat_types)).
        """
        atom_num = len(mol.GetAtoms())
        feature = np.zeros((atom_num, len(self.feat_types)), dtype=float)

        fact_feats = factory.GetFeaturesForMol(mol)
        for f in fact_feats:
            f_type = f.GetFamily()
            if f_type in self.feat_types:
                f_index = self.feat_types.index(f_type)
                atom_ids = f.GetAtomIds()
                feature[atom_ids, f_index] = 1.0

        return feature

    def mol_simplified_feature(self, mol: Chem.Mol) -> np.ndarray:
        """
        Simplified atom-type encoding based on the predefined atom classes.

        Each atom is assigned:
            - self.ATOM_CODES[atomic_num] + 1  if atomic_num is known
            - 0                                otherwise

        This is often used as an integer label per atom (e.g., for embeddings).

        Parameters
        ----------
        mol : Chem.Mol

        Returns
        -------
        feature : np.ndarray
            Shape (num_atoms, 1), containing integer class labels in [0..NUM_CLASSES].
        """
        atom_ids: List[int] = []
        atom_feats: List[List[int]] = []

        for atom in mol.GetAtoms():
            atom_ids.append(atom.GetIdx())
            atomic_num = atom.GetAtomicNum()

            if atomic_num in self.ATOM_CODES:
                atom_feats.append([self.ATOM_CODES[atomic_num] + 1])
            else:
                atom_feats.append([0])

        feature = np.array(list(zip(*sorted(zip(atom_ids, atom_feats))))[-1], dtype=int)

        return feature

    def mol_sequence_simplified_feature(self, mol: Chem.Mol) -> np.ndarray:
        """
        Simplified atom-type encoding using a fixed element list and one-hot.

        For each atom, we:
            - One-hot encode `atom.GetSymbol()` over a predefined symbol list.
            - Take the index of the non-zero position as that atom's label.

        The resulting label is an integer index into the symbol vocabulary.

        Parameters
        ----------
        mol : Chem.Mol

        Returns
        -------
        feature : np.ndarray
            Shape (num_atoms,), each entry is an integer label index.
        """
        atom_ids: List[int] = []
        atom_feats: List[np.ndarray] = []

        vocab = [
            "C",
            "N",
            "O",
            "S",
            "F",
            "Si",
            "P",
            "Cl",
            "Br",
            "Mg",
            "Na",
            "Ca",
            "Fe",
            "As",
            "Al",
            "I",
            "B",
            "V",
            "K",
            "Tl",
            "Yb",
            "Sb",
            "Sn",
            "Ag",
            "Pd",
            "Co",
            "Se",
            "Ti",
            "Zn",
            "H",
            "Li",
            "Ge",
            "Cu",
            "Au",
            "Ni",
            "Cd",
            "In",
            "Mn",
            "Zr",
            "Cr",
            "Pt",
            "Hg",
            "Pb",
            "Unknown",
        ]

        for atom in mol.GetAtoms():
            atom_ids.append(atom.GetIdx())
            onehot_label = one_of_k_encoding_unk(atom.GetSymbol(), vocab)
            out = np.array(onehot_label).nonzero()[0]
            atom_feats.append(out)

        feature = np.array(list(zip(*sorted(zip(atom_ids, atom_feats))))[-1], dtype=int)

        return feature

    def mol_full_feature(self, mol: Chem.Mol) -> np.ndarray:
        """
        Return a rich per-atom feature matrix using an external `atom_features` helper.

        This function:
            - Assigns stereochemistry information (CIP tags, etc.).
            - Calls `atom_features(atom)` for each atom.
            - Stacks the resulting vectors into an array.

        Parameters
        ----------
        mol : Chem.Mol

        Returns
        -------
        features : np.ndarray
            Shape (num_atoms, feat_dim) as defined by `atom_features`.
        """
        # Ensure stereochemistry info is properly assigned
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

        atom_feats = [atom_features(atom) for atom in mol.GetAtoms()]
        features = np.vstack(atom_feats)
        return features

    def bond_feature(self, mol: Chem.Mol) -> np.ndarray:
        """
        Construct a symmetric adjacency matrix of bond features.

        Each non-zero entry (i, j) corresponds to a bond between atoms i and j.
        The value is an integer code for bond type (SINGLE, DOUBLE, TRIPLE, AROMATIC)
        taken from `self.edge_dict`.

        Note: the code uses (v1 - 1, v2 - 1); this assumes that atom indices are
        1-based or that the downstream pipeline expects this shift. If atom indices
        are 0-based in your data flow, adjust accordingly.

        Parameters
        ----------
        mol : Chem.Mol

        Returns
        -------
        adj : np.ndarray
            Shape (num_atoms, num_atoms), containing integer bond-type codes.
        """
        atom_num = len(mol.GetAtoms())
        adj = np.zeros((atom_num, atom_num), dtype=int)

        for b in mol.GetBonds():
            v1 = b.GetBeginAtomIdx()
            v2 = b.GetEndAtomIdx()
            b_type = self.edge_dict[b.GetBondType()]
            adj[v1 - 1, v2 - 1] = b_type
            adj[v2 - 1, v1 - 1] = b_type

        return adj

    def junction_tree(self, mol: Chem.Mol) -> Dict[str, Tensor]:
        """
        Construct the junction tree (clique tree) representation of a molecule.

        This wraps `tree_decomposition` and provides a robust fallback for molecules
        where no cliques are detected (e.g., unusual structures). In that case, each
        atom is assigned to its own clique.

        Parameters
        ----------
        mol : Chem.Mol

        Returns
        -------
        tree : dict
            Dictionary with keys:
                - "tree_edge_index": clique–clique edge index (2 × E tensor)
                - "atom2clique_index": mapping from atoms to cliques (2 × num_atoms)
                - "num_cliques": number of cliques in the tree
                - "x_clique": per-clique feature vector (e.g., vocab index)
        """
        (
            tree_edge_index,
            atom2clique_index,
            num_cliques,
            x_clique,
        ) = tree_decomposition(mol, return_vocab=True)

        # Fallback: if no cliques found, treat each atom as its own clique
        if atom2clique_index.nelement() == 0:
            num_cliques = len(mol.GetAtoms())
            x_clique = torch.tensor([3] * num_cliques, dtype=torch.long)
            atom2clique_index = torch.stack(
                [torch.arange(num_cliques), torch.arange(num_cliques)]
            )

        tree: Dict[str, Tensor] = dict(
            tree_edge_index=tree_edge_index,
            atom2clique_index=atom2clique_index,
            num_cliques=torch.tensor(num_cliques),
            x_clique=x_clique,
        )

        return tree

    def featurize(
        self,
        mol: Chem.Mol,
        type: str = "atom_type",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Unified entry point to obtain per-atom features and bond adjacency.

        Parameters
        ----------
        mol : Chem.Mol
            RDKit molecule.
        type : {"atom_type", "detailed_atom_type", "atom_feature", "atom_full_feature"}
            Controls which atom-level representation is computed:
                - "atom_type":
                    integer labels based on `mol_simplified_feature`
                - "detailed_atom_type":
                    element-index labels based on `mol_sequence_simplified_feature`
                - "atom_feature":
                    concatenation of `mol_feature` and `mol_extra_feature`
                    (handcrafted + pharmacophore)
                - "atom_full_feature":
                    rich external `atom_features` representation

        Returns
        -------
        atom_feature : np.ndarray
            Per-atom feature matrix of shape (num_atoms, feat_dim).
        bond_feature : np.ndarray
            Bond adjacency matrix of shape (num_atoms, num_atoms) with bond-type codes.
        """
        if type == "atom_type":
            atom_feature = self.mol_simplified_feature(mol)

        elif type == "detailed_atom_type":
            atom_feature = self.mol_sequence_simplified_feature(mol)

        elif type == "atom_feature":
            base_feat = self.mol_feature(mol)
            extra_feat = self.mol_extra_feature(mol)
            atom_feature = np.concatenate((base_feat, extra_feat), axis=1)

        elif type == "atom_full_feature":
            atom_feature = self.mol_full_feature(mol)
            # If you want, you could also add `mol_extra_feature` here
            # and concatenate, similar to "atom_feature".
        else:
            msg = (
                "Featurization type not implemented. Use one of: "
                "'atom_type', 'detailed_atom_type', 'atom_feature', 'atom_full_feature'."
            )
            logger.error(msg)
            raise ValueError(msg)

        bond_feature = self.bond_feature(mol)

        return atom_feature, bond_feature


# def ligand_init(smiles_list: list[str]) -> dict[str, dict]:
#     ligand_dict = {}
#     for smiles in tqdm(smiles_list):
#         ligand_dict[smiles] = smiles2graph(smiles)

#     return ligand_dict


def ligand_init(smiles_list: list[str]) -> dict[str, dict]:
    ligand_dict = {}
    dropped = 0

    for smi in smiles_list:
        g = smiles2graph(smi)
        if g is None:
            dropped += 1
            continue
        ligand_dict[smi] = g

    logger.info("Ligands built: %d, dropped: %d", len(ligand_dict), dropped)
    return ligand_dict
