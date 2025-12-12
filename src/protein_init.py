import numpy as np

from pathlib import Path
from Bio.PDB import PDBParser

from tqdm import tqdm

import torch
from torch import Tensor
import esm
from torch_geometric.utils import (
    add_self_loops,
    to_undirected,
    remove_self_loops,
    coalesce,
)
from typing import Callable, Tuple
import math


from src.utils import get_logger

logger = get_logger(__name__)


def protein_init(model, alphabet, seqs: list[str]) -> dict[str, dict]:
    """Initializes protein graphs from sequences."""
    logger.info("Initializing protein graphs from sequences")
    result_dict = {}
    batch_converter = alphabet.get_batch_converter()

    for seq in tqdm(seqs):
        seq_feat = seq_feature(seq)
        token_repr, contact_map_proba, _ = esm_extract(
            model, batch_converter, seq, layer=33, approach="last", dim=1280
        )

        assert len(contact_map_proba) == len(seq)
        edge_index, edge_weight = contact_map(contact_map_proba)

        result_dict[seq] = {
            "seq": seq,
            "seq_feat": torch.from_numpy(seq_feat),
            "token_representation": token_repr.half(),
            "num_nodes": len(seq),
            "num_pos": torch.arange(len(seq)).reshape(-1, 1),
            "edge_index": edge_index,
            "edge_weight": edge_weight,
        }

    return result_dict


# normalize
def dic_normalize(dic):
    # logger.info(dic)
    max_value = dic[max(dic, key=dic.get)]
    min_value = dic[min(dic, key=dic.get)]
    # logger.info(max_value)
    interval = float(max_value) - float(min_value)
    for key in dic.keys():
        dic[key] = (dic[key] - min_value) / interval
    dic["X"] = (max_value + min_value) / 2.0
    return dic


pro_res_table = [
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
    "X",
]

pro_res_aliphatic_table = ["A", "I", "L", "M", "V"]
pro_res_aromatic_table = ["F", "W", "Y"]
pro_res_polar_neutral_table = ["C", "N", "Q", "S", "T"]
pro_res_acidic_charged_table = ["D", "E"]
pro_res_basic_charged_table = ["H", "K", "R"]

res_weight_table = {
    "A": 71.08,
    "C": 103.15,
    "D": 115.09,
    "E": 129.12,
    "F": 147.18,
    "G": 57.05,
    "H": 137.14,
    "I": 113.16,
    "K": 128.18,
    "L": 113.16,
    "M": 131.20,
    "N": 114.11,
    "P": 97.12,
    "Q": 128.13,
    "R": 156.19,
    "S": 87.08,
    "T": 101.11,
    "V": 99.13,
    "W": 186.22,
    "Y": 163.18,
}

res_pka_table = {
    "A": 2.34,
    "C": 1.96,
    "D": 1.88,
    "E": 2.19,
    "F": 1.83,
    "G": 2.34,
    "H": 1.82,
    "I": 2.36,
    "K": 2.18,
    "L": 2.36,
    "M": 2.28,
    "N": 2.02,
    "P": 1.99,
    "Q": 2.17,
    "R": 2.17,
    "S": 2.21,
    "T": 2.09,
    "V": 2.32,
    "W": 2.83,
    "Y": 2.32,
}

res_pkb_table = {
    "A": 9.69,
    "C": 10.28,
    "D": 9.60,
    "E": 9.67,
    "F": 9.13,
    "G": 9.60,
    "H": 9.17,
    "I": 9.60,
    "K": 8.95,
    "L": 9.60,
    "M": 9.21,
    "N": 8.80,
    "P": 10.60,
    "Q": 9.13,
    "R": 9.04,
    "S": 9.15,
    "T": 9.10,
    "V": 9.62,
    "W": 9.39,
    "Y": 9.62,
}

res_pkx_table = {
    "A": 0.00,
    "C": 8.18,
    "D": 3.65,
    "E": 4.25,
    "F": 0.00,
    "G": 0,
    "H": 6.00,
    "I": 0.00,
    "K": 10.53,
    "L": 0.00,
    "M": 0.00,
    "N": 0.00,
    "P": 0.00,
    "Q": 0.00,
    "R": 12.48,
    "S": 0.00,
    "T": 0.00,
    "V": 0.00,
    "W": 0.00,
    "Y": 0.00,
}

res_pl_table = {
    "A": 6.00,
    "C": 5.07,
    "D": 2.77,
    "E": 3.22,
    "F": 5.48,
    "G": 5.97,
    "H": 7.59,
    "I": 6.02,
    "K": 9.74,
    "L": 5.98,
    "M": 5.74,
    "N": 5.41,
    "P": 6.30,
    "Q": 5.65,
    "R": 10.76,
    "S": 5.68,
    "T": 5.60,
    "V": 5.96,
    "W": 5.89,
    "Y": 5.96,
}

res_hydrophobic_ph2_table = {
    "A": 47,
    "C": 52,
    "D": -18,
    "E": 8,
    "F": 92,
    "G": 0,
    "H": -42,
    "I": 100,
    "K": -37,
    "L": 100,
    "M": 74,
    "N": -41,
    "P": -46,
    "Q": -18,
    "R": -26,
    "S": -7,
    "T": 13,
    "V": 79,
    "W": 84,
    "Y": 49,
}

res_hydrophobic_ph7_table = {
    "A": 41,
    "C": 49,
    "D": -55,
    "E": -31,
    "F": 100,
    "G": 0,
    "H": 8,
    "I": 99,
    "K": -23,
    "L": 97,
    "M": 74,
    "N": -28,
    "P": -46,
    "Q": -10,
    "R": -14,
    "S": -5,
    "T": 13,
    "V": 76,
    "W": 97,
    "Y": 63,
}

res_weight_table = dic_normalize(res_weight_table)
res_pka_table = dic_normalize(res_pka_table)
res_pkb_table = dic_normalize(res_pkb_table)
res_pkx_table = dic_normalize(res_pkx_table)
res_pl_table = dic_normalize(res_pl_table)
res_hydrophobic_ph2_table = dic_normalize(res_hydrophobic_ph2_table)
res_hydrophobic_ph7_table = dic_normalize(res_hydrophobic_ph7_table)


def residue_features(residue):
    res_property1 = [
        1 if residue in pro_res_aliphatic_table else 0,
        1 if residue in pro_res_aromatic_table else 0,
        1 if residue in pro_res_polar_neutral_table else 0,
        1 if residue in pro_res_acidic_charged_table else 0,
        1 if residue in pro_res_basic_charged_table else 0,
    ]
    res_property2 = [
        res_weight_table[residue],
        res_pka_table[residue],
        res_pkb_table[residue],
        res_pkx_table[residue],
        res_pl_table[residue],
        res_hydrophobic_ph2_table[residue],
        res_hydrophobic_ph7_table[residue],
    ]
    # logger.info(np.array(res_property1 + res_property2).shape)
    return np.array(res_property1 + res_property2)


# one ont encoding
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        # logger.info(x)
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def seq_feature(pro_seq: str) -> np.ndarray:
    #    U and B are not used in the PDB
    #    They appear rarely (<1% of residues)
    #    They do not occur consistently in catalytically important sites
    if "U" in pro_seq or "B" in pro_seq:
        logger.info("U or B in Sequence")
    pro_seq = pro_seq.replace("U", "X").replace("B", "X")
    pro_hot = np.zeros((len(pro_seq), len(pro_res_table)))
    pro_property = np.zeros((len(pro_seq), 12))
    for i in range(len(pro_seq)):
        # if 'X' in pro_seq:
        #     logger.info(pro_seq)
        pro_hot[i,] = one_of_k_encoding(pro_seq[i], pro_res_table)
        pro_property[i,] = residue_features(pro_seq[i])
    return np.concatenate((pro_hot, pro_property), axis=1)


def contact_map(
    contact_map_proba: np.ndarray,
    contact_threshold: float = 0.5,
) -> tuple[Tensor, Tensor]:
    """
    Construct a protein graph from an ESM-predicted contact probability map.

    This function converts a residue–residue contact probability matrix
    (L × L, where L is the number of residues) into a PyTorch Geometric-style
    graph representation consisting of:

        - edge_index: 2 × E tensor of residue pair indices
        - edge_weight: E-dimensional tensor of edge strengths (contact probabilities)

    The graph includes two types of edges:

    1. **Predicted structural contacts**
       Residue pairs with contact probability ≥ contact_threshold are included.
       These edges capture long-range 3D interactions learned by the ESM model.

    2. **Sequence-adjacency edges**
       To prevent disconnected nodes (isolated residues), we add edges that
       connect:
           - i ↔ i+1  (backbone adjacency)
           - i ↔ i+2  (short-range proximity)
       This ensures the graph is fully connected even if the contact map is sparse,
       which stabilizes downstream GNN training and message passing.

    After assembling all edges, we:
        - coalesce duplicates using max() weight aggregation
        - enforce undirected edges
        - remove self-loops
        - add self-loops (with weight = 1), which many GNN layers require.

    Parameters
    ----------
    contact_map_proba : np.ndarray
        Square (L × L) matrix of residue–residue contact probabilities.
    contact_threshold : float
        Minimum probability to include a predicted contact as a graph edge.

    Returns
    -------
    edge_index : torch.Tensor
        Shape (2, E). Graph edges.
    edge_weight : torch.Tensor
        Shape (E,). Weights for each edge (contact probabilities or default seq weights).
    """
    # Number of residues in the sequence
    num_residues = contact_map_proba.shape[0]

    # Binary adjacency matrix based on the contact threshold
    prot_contact_adj = (contact_map_proba >= contact_threshold).long()

    # Extract edges where adjacency = 1
    edge_index = prot_contact_adj.nonzero(as_tuple=False).t().contiguous()
    row, col = edge_index

    # Edge weights are the raw probabilities for those contacts
    edge_weight = torch.tensor(contact_map_proba[row, col], dtype=torch.float32)

    # ----------------------------------------------------------------------
    # CONNECT ISOLATED NODES (Sequence edges) – Level 1: i <-> i+1
    # ----------------------------------------------------------------------
    seq = torch.arange(num_residues)

    # i -> i+1 edges
    seq_edge_head1 = torch.stack([seq[:-1], seq[1:]])
    # i+1 -> i edges
    seq_edge_tail1 = torch.stack([seq[1:], seq[:-1]])

    # All sequence edges receive a uniform weight = contact_threshold
    seq_edge_weight1 = (
        torch.ones(seq_edge_head1.size(1) + seq_edge_tail1.size(1)) * contact_threshold
    )

    # Append to graph
    edge_index = torch.cat([edge_index, seq_edge_head1, seq_edge_tail1], dim=1)
    edge_weight = torch.cat([edge_weight, seq_edge_weight1], dim=0)

    # ----------------------------------------------------------------------
    # CONNECT ISOLATED NODES – Level 2: i <-> i+2
    # These help enforce short-range continuity and improve GNN propagation.
    # ----------------------------------------------------------------------
    seq_edge_head2 = torch.stack([seq[:-2], seq[2:]])
    seq_edge_tail2 = torch.stack([seq[2:], seq[:-2]])

    seq_edge_weight2 = (
        torch.ones(seq_edge_head2.size(1) + seq_edge_tail2.size(1)) * contact_threshold
    )

    edge_index = torch.cat([edge_index, seq_edge_head2, seq_edge_tail2], dim=1)
    edge_weight = torch.cat([edge_weight, seq_edge_weight2], dim=0)

    # ----------------------------------------------------------------------
    # FINAL GRAPH CLEANING
    # ----------------------------------------------------------------------

    # Remove duplicates; if multiple edges exist, keep the max weight
    edge_index, edge_weight = coalesce(edge_index, edge_weight, reduce="max")

    # Ensure undirected graph: if i→j exists, ensure j→i exists (max weight kept)
    edge_index, edge_weight = to_undirected(edge_index, edge_weight, reduce="max")

    # Remove any i→i self-loops before adding clean ones
    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

    # Add self-loops required by many GNN architectures (weight = 1)
    edge_index, edge_weight = add_self_loops(edge_index, edge_weight, fill_value=1.0)

    return edge_index, edge_weight


def esm_extract(
    model: torch.nn.Module,
    batch_converter: Callable,
    seq: str,
    layer: int = 33,  # esm1v_t33_650M_UR90S_5
    approach: str = "mean",  # 'mean', 'sum', or 'last' over layers
    dim: int = 1280,  # esm1v embed_dim
    interval: int = 350,
    return_nan_on_error: bool = True,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    ESM1v extraction for esm1v_t33_650M_UR90S_5 with sanitization and token validity checks.

    Parameters
    ----------
    model : torch.nn.Module
        ESM1v model instance (e.g. esm1v_t33_650M_UR90S_5).
    batch_converter : callable
        Alphabet batch converter returned by esm.Alphabet.get_batch_converter().
    seq : str
        Amino acid sequence (will be uppercased and non-20 AAs mapped to 'X').
    layer : int, default 33
        Number of layers to use / aggregate over (clipped to model.num_layers).
    approach : {'mean', 'sum', 'last'}
        How to combine layer-wise representations.
    dim : int, default 1280
        Embedding dimension (model.embed_dim for esm1v_t33_650M_UR90S_5).
    interval : int, default 350
        Chunk size for long-sequence sliding-window inference.
    return_nan_on_error : bool, default True
        If True, return NaN tensors instead of raising on errors.

    Returns
    -------
    token_representation : (L, dim) torch.FloatTensor
        Per-residue embeddings (special tokens removed).
    contact_prob_map : (L, L) torch.FloatTensor
        Contact probability matrix.
    logits : (L, vocab_size) torch.FloatTensor
        Per-residue logits over vocabulary.
    """

    device = next(model.parameters()).device

    # Model properties (robust even if you switch to another ESM1v variant)
    num_layers = int(getattr(model, "num_layers", layer))
    embed_dim = int(getattr(model, "embed_dim", dim))
    max_len_model = int(getattr(model, "max_positions", 1022))

    # Clip layer to model.num_layers if user passes a larger value
    if layer > num_layers:
        logger.warning(
            "Requested layer=%d > model.num_layers=%d; clipping to num_layers.",
            layer,
            num_layers,
        )
        layer = num_layers

    # Use model.embed_dim as true dim if not matching default
    if dim != embed_dim:
        logger.debug(
            "Overriding dim=%d with model.embed_dim=%d for ESM1v.", dim, embed_dim
        )
        dim = embed_dim

    # Helper to generate NaN outputs with proper shapes
    def _nan_outputs(L: int) -> Tuple[Tensor, Tensor, Tensor]:
        vocab_size = int(getattr(model.embed_tokens, "num_embeddings", 1))
        token_repr = torch.full((L, dim), float("nan"), dtype=torch.float32)
        contact = torch.full((L, L), float("nan"), dtype=torch.float32)
        logits = torch.full((L, vocab_size), float("nan"), dtype=torch.float32)
        return token_repr, contact, logits

    try:
        # Sanitize sequence: strip, uppercase, map non-20 AAs to 'X'
        seq = (seq or "").strip().upper()
        clean = []
        replaced = 0
        for aa in seq:
            if aa in "ACDEFGHIKLMNPQRSTVWY":
                clean.append(aa)
            else:
                clean.append("X")
                replaced += 1
        seq = "".join(clean)

        if replaced > 0:
            logger.debug(
                "Sequence len %d → %d residues replaced with 'X' (ESM1v-safe).",
                len(seq),
                replaced,
            )

        L = len(seq)
        if L == 0:
            logger.warning("Empty sequence after sanitization; returning NaNs.")
            return _nan_outputs(0)

        pro_id = "A"
        vocab_size = int(model.embed_tokens.num_embeddings)

        # Short sequence branch: single forward pass (no chunking)
        # Keep some margin from max_positions because of special tokens
        if L <= min(700, max_len_model):
            # Tokenize on CPU
            _, _, batch_tokens = batch_converter([(pro_id, seq)])

            # Token ID validity check before GPU
            max_id = int(batch_tokens.max().item())
            if max_id >= vocab_size:
                bad_idx = (
                    (batch_tokens >= vocab_size).nonzero(as_tuple=True)[1].tolist()
                )
                bad_chars = [seq[i - 1] for i in bad_idx if 0 <= i - 1 < L]
                msg = (
                    f"Invalid token IDs for seq len {L} (max id {max_id}, "
                    f"vocab {vocab_size}). Chars: {bad_chars}"
                )
                logger.error(msg)
                if return_nan_on_error:
                    return _nan_outputs(L)
                raise RuntimeError(msg)

            batch_tokens = batch_tokens.to(device, non_blocking=True)

            # Forward
            with torch.no_grad():
                results = model(
                    batch_tokens,
                    repr_layers=[i for i in range(1, layer + 1)],
                    return_contacts=True,
                )

            # logits: (1, L+2, vocab_size) → (L, vocab_size)
            logits_np = results["logits"][0].cpu().numpy()[1 : L + 1]

            # contacts: may be (L+2, L+2) → slice, or already (L, L)
            if "contacts" in results:
                contacts_raw = results["contacts"][0].cpu().numpy()
                if contacts_raw.shape[0] == L + 2:
                    contact_prob_map_np = contacts_raw[1 : L + 1, 1 : L + 1]
                else:
                    contact_prob_map_np = contacts_raw
            else:
                contact_prob_map_np = np.zeros((L, L), dtype=np.float32)

            # representations: [layer, 1, L+2, dim]
            token_representation = torch.cat(
                [results["representations"][i] for i in range(1, layer + 1)]
            )
            assert token_representation.size(0) == layer

            # Combine layers
            if approach == "last":
                token_representation = token_representation[-1]
            elif approach == "sum":
                token_representation = token_representation.sum(dim=0)
            elif approach == "mean":
                token_representation = token_representation.mean(dim=0)
            else:
                raise ValueError(
                    f"Unknown approach='{approach}' (expected 'last', 'sum', 'mean')."
                )

            # Drop special tokens → (L, dim)
            token_representation_np = token_representation.cpu().numpy()[1 : L + 1]

        # Long sequence branch: sliding-window with overlap + averaging
        else:
            contact_prob_map_np = np.zeros((L, L), dtype=np.float32)
            token_representation_np = np.zeros((L, dim), dtype=np.float32)
            logits_np = np.zeros((L, vocab_size), dtype=np.float32)

            n_chunks = math.ceil(L / interval)

            for s in range(n_chunks):
                start = s * interval
                end = min((s + 2) * interval, L)  # allow overlap window

                temp_seq = seq[start:end]
                _, _, batch_tokens = batch_converter([(pro_id, temp_seq)])

                # Token ID validity check before GPU
                max_id = int(batch_tokens.max().item())
                if max_id >= vocab_size:
                    bad_idx = (
                        (batch_tokens >= vocab_size).nonzero(as_tuple=True)[1].tolist()
                    )
                    bad_chars = [
                        temp_seq[i - 1] for i in bad_idx if 0 <= i - 1 < len(temp_seq)
                    ]
                    msg = (
                        f"Invalid token IDs in chunk [{start}:{end}] (max id {max_id}, "
                        f"vocab {vocab_size}). Chars: {bad_chars}"
                    )
                    logger.error(msg)
                    if return_nan_on_error:
                        return _nan_outputs(L)
                    raise RuntimeError(msg)

                batch_tokens = batch_tokens.to(device, non_blocking=True)

                with torch.no_grad():
                    results = model(
                        batch_tokens,
                        repr_layers=[i for i in range(1, layer + 1)],
                        return_contacts=True,
                    )

                # Local logits (len(temp_seq), vocab_size)
                local_logits = results["logits"][0].cpu().numpy()[1 : len(temp_seq) + 1]

                # Local contacts
                if "contacts" in results:
                    contacts_raw = results["contacts"][0].cpu().numpy()
                    if contacts_raw.shape[0] == len(temp_seq) + 2:
                        local_contacts = contacts_raw[
                            1 : len(temp_seq) + 1, 1 : len(temp_seq) + 1
                        ]
                    else:
                        local_contacts = contacts_raw
                else:
                    local_contacts = np.zeros(
                        (end - start, end - start), dtype=np.float32
                    )

                # Merge contacts into global contact map
                existing_mask = contact_prob_map_np[start:end, start:end] != 0
                row, col = np.where(existing_mask)
                row = row + start
                col = col + start

                contact_prob_map_np[start:end, start:end] += local_contacts
                if row.size > 0:
                    contact_prob_map_np[row, col] = contact_prob_map_np[row, col] / 2.0

                # Merge logits
                logits_np[start:end] += local_logits
                if row.size > 0:
                    logits_np[row] = logits_np[row] / 2.0

                # Local token representations
                subtoken_repr = torch.cat(
                    [results["representations"][i] for i in range(1, layer + 1)]
                )
                assert subtoken_repr.size(0) == layer

                if approach == "last":
                    subtoken_repr = subtoken_repr[-1]
                elif approach == "sum":
                    subtoken_repr = subtoken_repr.sum(dim=0)
                elif approach == "mean":
                    subtoken_repr = subtoken_repr.mean(dim=0)
                else:
                    raise ValueError(
                        f"Unknown approach='{approach}' (expected 'last', 'sum', 'mean')."
                    )

                subtoken_repr_np = subtoken_repr.cpu().numpy()[1 : len(temp_seq) + 1]

                # Overlap positions for embeddings
                trow = np.where(token_representation_np[start:end].sum(axis=-1) != 0)[0]
                trow = trow + start

                token_representation_np[start:end] += subtoken_repr_np
                if trow.size > 0:
                    token_representation_np[trow] = token_representation_np[trow] / 2.0

                if end == L:
                    break

        # Convert to float32 tensors
        token_representation = torch.from_numpy(
            token_representation_np.astype(np.float32, copy=False)
        )
        contact_prob_map = torch.from_numpy(
            contact_prob_map_np.astype(np.float32, copy=False)
        )
        logits = torch.from_numpy(logits_np.astype(np.float32, copy=False))

        return token_representation, contact_prob_map, logits

    except Exception as e:
        logger.error("Error in esm_extract for sequence len %d: %s", len(seq), e)
        if return_nan_on_error:
            return _nan_outputs(len(seq))
        raise


# def generate_ESM_structure(
#     model: torch.nn.Module, filename: Path, sequence: str
# ) -> bool:
#     model.set_chunk_size(256)
#     chunk_size = 256
#     output = None

#     while output is None:
#         try:
#             with torch.no_grad():
#                 output = model.infer_pdb(sequence)

#             with open(filename, "w") as f:
#                 f.write(output)
#                 logger.info("saved", filename)
#         except RuntimeError as e:
#             if "out of memory" in str(e):
#                 logger.info("| WARNING: ran out of memory on chunk_size", chunk_size)
#                 for p in model.parameters():
#                     if p.grad is not None:
#                         del p.grad  # free some memory
#                 torch.cuda.empty_cache()
#                 chunk_size = chunk_size // 2
#                 if chunk_size > 2:
#                     model.set_chunk_size(chunk_size)
#                 else:
#                     logger.info("Not enough memory for ESMFold")
#                     break
#             else:
#                 raise e
#     return output is not None


biopython_parser = PDBParser()

one_to_three = {
    "A": "ALA",
    "C": "CYS",
    "D": "ASP",
    "E": "GLU",
    "F": "PHE",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "K": "LYS",
    "L": "LEU",
    "M": "MET",
    "N": "ASN",
    "P": "PRO",
    "Q": "GLN",
    "R": "ARG",
    "S": "SER",
    "T": "THR",
    "V": "VAL",
    "W": "TRP",
    "Y": "TYR",
    "B": "ASX",
    "Z": "GLX",
    "X": "UNK",
    "*": " * ",
}

three_to_one = {}
for _key, _value in one_to_three.items():
    three_to_one[_value] = _key
three_to_one["SEC"] = "C"
three_to_one["MSE"] = "M"


def extract_pdb_seq(protein_path: Path) -> tuple[str, str]:

    structure = biopython_parser.get_structure("random_id", protein_path)[0]
    seq = ""
    chain_str = ""
    for _, chain in enumerate(structure):
        for _, residue in enumerate(chain):
            if residue.get_resname() == "HOH":
                continue
            c_alpha, n, c = None, None, None
            for atom in residue:
                if atom.name == "CA":
                    c_alpha = list(atom.get_vector())
                if atom.name == "N":
                    n = list(atom.get_vector())
                if atom.name == "C":
                    c = list(atom.get_vector())
            if (
                c_alpha != None and n != None and c != None
            ):  # only append residue if it is an amino acid and not
                try:
                    seq += three_to_one[residue.get_resname()]
                    chain_str += str(chain.id)
                except Exception as e:
                    seq += "X"
                    chain_str += str(chain.id)
                    logger.info(
                        "encountered unknown AA: ",
                        residue.get_resname(),
                        " in the complex. Replacing it with a dash X.",
                    )

    return seq, chain_str
