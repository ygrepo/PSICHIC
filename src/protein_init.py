from __future__ import annotations
import numpy as np

from pathlib import Path

# from Bio.PDB import PDBParser

from tqdm import tqdm

import torch
from torch import Tensor
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


def sanitize_seq(s: str) -> str:
    s = (s or "").strip().upper()
    allowed = set("ACDEFGHIKLMNPQRSTVWY")
    return "".join([aa if aa in allowed else "X" for aa in s])


def protein_init(model, alphabet, seqs: list[str]) -> dict[str, dict]:
    """Initializes protein graphs from sequences (FAIR ESM1v/ESM2)."""
    logger.info("Initializing protein graphs from sequences")
    result_dict = {}
    batch_converter = alphabet.get_batch_converter()

    model_layers = int(getattr(model, "num_layers", 0))
    model_dim = int(getattr(model, "embed_dim", 0))
    use_layer = model_layers if model_layers > 0 else 33
    use_dim = model_dim or 1280

    for seq in tqdm(seqs):
        seq_clean = sanitize_seq(seq)
        L = len(seq_clean)

        # If sanitization produces empty (unlikely), skip
        if L == 0:
            logger.warning("Skipping empty sequence after sanitization.")
            continue

        # Ensure seq_feature uses the same cleaned sequence
        seq_feat = seq_feature(seq_clean)

        token_repr, contact_map_proba, _ = esm_extract(
            model,
            batch_converter,
            seq_clean,
            layer=use_layer,
            approach="last",
            dim=use_dim,
            interval=350,  # explicit
        )

        # Fallback instead of skipping (prevents dataset KeyError)
        if torch.isnan(token_repr).any() or torch.isnan(contact_map_proba).any():
            logger.warning("Protein ESM failed; using zero features (len=%d)", L)
            token_repr = torch.zeros((L, use_dim), dtype=torch.float32)
            contact_map_proba = torch.zeros((L, L), dtype=torch.float32)

        # Shape checks (now against seq_clean length)
        assert (
            token_repr.shape[0] == L
        ), f"token_repr len mismatch: {token_repr.shape[0]} vs {L}"
        assert contact_map_proba.shape == (
            L,
            L,
        ), f"contact_map shape mismatch: {tuple(contact_map_proba.shape)} vs {(L, L)}"

        seq_feat_t = torch.from_numpy(seq_feat)
        assert (
            seq_feat_t.shape[0] == L
        ), f"seq_feat len mismatch: {seq_feat_t.shape[0]} vs {L}"

        num_pos = torch.arange(L).reshape(-1, 1)

        # Build edges from contacts
        edge_index, edge_weight = contact_map(contact_map_proba)

        # IMPORTANT: choose a stable key
        # If dataset uses raw seq strings as keys, keep seq as the key.
        # If you can, key by seq_clean and ensure dataset uses seq_clean too.
        result_dict[seq] = {
            "seq": seq,
            "seq_feat": seq_feat_t,
            "token_representation": token_repr.half(),
            "num_nodes": L,
            "num_pos": num_pos,
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
    layer: int = 33,
    approach: str = "mean",  # "last" | "sum" | "mean"
    dim: int = 1280,
    interval: int = 350,  # stride-like parameter for your overlap scheme
    return_nan_on_error: bool = True,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Extract per-residue embeddings, contact probabilities, and logits from FAIR ESM models
    (ESM1v / ESM2 via esm.pretrained.*).

    Returns
    -------
    token_representation : (L, dim) float32
    contact_prob_map     : (L, L) float32
    logits               : (L, vocab_size) float32
    """

    device = next(model.parameters()).device

    # Model properties (robust across ESM variants)
    num_layers = int(getattr(model, "num_layers", layer))
    embed_dim = int(getattr(model, "embed_dim", dim))
    max_positions = int(
        getattr(model, "max_positions", 1022)
    )  # FAIR ESM often uses 1022 residues (+2 specials)

    if layer > num_layers:
        logger.warning(
            "Requested layer=%d > model.num_layers=%d; clipping.", layer, num_layers
        )
        layer = num_layers
    if dim != embed_dim:
        dim = embed_dim

    # ---------- helpers ----------

    def _nan_outputs(L: int) -> Tuple[Tensor, Tensor, Tensor]:
        vocab_size = int(getattr(model.embed_tokens, "num_embeddings", 1))
        token_repr = torch.full((L, dim), float("nan"), dtype=torch.float32)
        contact = torch.full((L, L), float("nan"), dtype=torch.float32)
        logits = torch.full((L, vocab_size), float("nan"), dtype=torch.float32)
        return token_repr, contact, logits

    def _sanitize(seq_in: str) -> str:
        s = (seq_in or "").strip().upper()
        allowed = set("ACDEFGHIKLMNPQRSTVWY")
        # map anything else to X (does not change length)
        return "".join([aa if aa in allowed else "X" for aa in s])

    def _choose_repr_layers(layer: int, approach: str):
        if approach == "last":
            return [layer]
        elif approach in ("sum", "mean"):
            return list(range(1, layer + 1))
        raise ValueError(
            f"Unknown approach='{approach}' (expected 'last', 'sum', 'mean')."
        )

    def _get_token_repr_from_results(
        results, *, repr_layers, layer: int, approach: str
    ) -> torch.Tensor:
        """
        Return token representations shaped (T, D), T includes special tokens (typically L+2).
        """
        if approach == "last":
            rep = results["representations"][layer]  # (1, T, D)
        elif approach in ("sum", "mean"):
            reps = [
                results["representations"][i] for i in repr_layers
            ]  # list of (1, T, D)
            stack = torch.stack(reps, dim=0)  # (n_layers, 1, T, D)
            rep = (
                stack.sum(dim=0) if approach == "sum" else stack.mean(dim=0)
            )  # (1, T, D)
        else:
            raise ValueError(
                f"Unknown approach='{approach}' (expected 'last', 'sum', 'mean')."
            )

        # remove batch dim -> (T, D)
        rep = rep.squeeze(0)
        return rep

    def _extract_contacts(results, L_expected: int) -> np.ndarray:
        """
        Returns contacts sliced to (L, L) if possible, else zeros.
        """
        if "contacts" not in results:
            return np.zeros((L_expected, L_expected), dtype=np.float32)
        contacts_raw = results["contacts"][0].detach().cpu().numpy()
        # Some ESM returns (L+2, L+2), some already (L, L)
        if contacts_raw.shape[0] == L_expected + 2:
            return contacts_raw[1 : L_expected + 1, 1 : L_expected + 1].astype(
                np.float32, copy=False
            )
        if contacts_raw.shape[0] == L_expected:
            return contacts_raw.astype(np.float32, copy=False)
        # Unexpected shape: return zeros (or raise if you prefer)
        logger.warning(
            "Unexpected contacts shape %s for L=%d; returning zeros.",
            tuple(contacts_raw.shape),
            L_expected,
        )
        return np.zeros((L_expected, L_expected), dtype=np.float32)

    # ---------- main ----------

    try:
        seq = _sanitize(seq)
        L = len(seq)
        if L == 0:
            logger.warning("Empty sequence after sanitization; returning NaNs.")
            return _nan_outputs(0)

        pro_id = "A"
        vocab_size = int(model.embed_tokens.num_embeddings)

        # Conservative max residue length for single pass:
        # FAIR ESM uses special tokens, so residues typically <= max_positions (often 1022).
        # We'll keep the prior 700 guard and also respect max_positions.
        max_residues_single = min(700, max_positions)

        # =========================================================
        # BRANCH 1: short sequence (single forward pass)
        # =========================================================
        if L <= max_residues_single:
            _, _, batch_tokens = batch_converter([(pro_id, seq)])

            # Token-ID validity check before GPU
            max_id = int(batch_tokens.max().item())
            if max_id >= vocab_size:
                bad_idx = (
                    (batch_tokens >= vocab_size).nonzero(as_tuple=True)[1].tolist()
                )
                bad_chars = [seq[i - 1] for i in bad_idx if 0 <= i - 1 < L]
                msg = (
                    f"Invalid token IDs for seq len {L} (max id {max_id}, vocab {vocab_size}). "
                    f"Chars: {bad_chars}"
                )
                logger.error(msg)
                if return_nan_on_error:
                    return _nan_outputs(L)
                raise RuntimeError(msg)

            batch_tokens = batch_tokens.to(device, non_blocking=True)

            repr_layers = _choose_repr_layers(layer, approach)

            with torch.no_grad():
                results = model(
                    batch_tokens, repr_layers=repr_layers, return_contacts=True
                )

            # logits: (1, L+2, vocab) -> (L, vocab)
            logits_np = (
                results["logits"][0]
                .detach()
                .cpu()
                .numpy()[1 : L + 1]
                .astype(np.float32, copy=False)
            )

            # contacts: -> (L, L)
            contact_prob_map_np = _extract_contacts(results, L)

            # representations: (T, D) then slice -> (L, D)
            token_rep = _get_token_repr_from_results(
                results, repr_layers=repr_layers, layer=layer, approach=approach
            )
            # token_rep is (L+2, D) typically
            token_representation_np = (
                token_rep.detach()
                .cpu()
                .numpy()[1 : L + 1]
                .astype(np.float32, copy=False)
            )

        # =========================================================
        # BRANCH 2: long sequence (sliding windows + count-based merge)
        # =========================================================
        else:
            # Allocate accumulators
            token_sum = np.zeros((L, dim), dtype=np.float32)
            token_cnt = np.zeros(L, dtype=np.int32)

            logits_sum = np.zeros((L, vocab_size), dtype=np.float32)
            logits_cnt = np.zeros(L, dtype=np.int32)

            # Contacts are heavy but L is typically ~1k here; OK.
            contact_sum = np.zeros((L, L), dtype=np.float32)
            contact_cnt = np.zeros((L, L), dtype=np.uint16)

            repr_layers = _choose_repr_layers(layer, approach)

            n_chunks = math.ceil(L / interval)

            for s in range(n_chunks):
                start = s * interval
                end = min((s + 2) * interval, L)  # your original overlap scheme
                temp_seq = seq[start:end]
                chunk_len = len(temp_seq)

                _, _, batch_tokens = batch_converter([(pro_id, temp_seq)])

                max_id = int(batch_tokens.max().item())
                if max_id >= vocab_size:
                    bad_idx = (
                        (batch_tokens >= vocab_size).nonzero(as_tuple=True)[1].tolist()
                    )
                    bad_chars = [
                        temp_seq[i - 1] for i in bad_idx if 0 <= i - 1 < chunk_len
                    ]
                    msg = (
                        f"Invalid token IDs in chunk [{start}:{end}] (max id {max_id}, vocab {vocab_size}). "
                        f"Chars: {bad_chars}"
                    )
                    logger.error(msg)
                    if return_nan_on_error:
                        return _nan_outputs(L)
                    raise RuntimeError(msg)

                batch_tokens = batch_tokens.to(device, non_blocking=True)

                with torch.no_grad():
                    results = model(
                        batch_tokens, repr_layers=repr_layers, return_contacts=True
                    )

                # ---- logits: (chunk_len, vocab) ----
                local_logits = (
                    results["logits"][0]
                    .detach()
                    .cpu()
                    .numpy()[1 : chunk_len + 1]
                    .astype(np.float32, copy=False)
                )
                logits_sum[start:end] += local_logits
                logits_cnt[start:end] += 1

                # ---- contacts: (chunk_len, chunk_len) ----
                local_contacts = _extract_contacts(results, chunk_len)
                # if "contacts" in results:
                #     contacts_raw = results["contacts"][0].detach().cpu().numpy()
                #     if contacts_raw.shape[0] == chunk_len + 2:
                #         local_contacts = contacts_raw[
                #             1 : chunk_len + 1, 1 : chunk_len + 1
                #         ].astype(np.float32, copy=False)
                #     elif contacts_raw.shape[0] == chunk_len:
                #         local_contacts = contacts_raw.astype(np.float32, copy=False)
                #     else:
                #         logger.warning(
                #             "Unexpected local contacts shape %s for chunk_len=%d; using zeros.",
                #             tuple(contacts_raw.shape),
                #             chunk_len,
                #         )
                #         local_contacts = np.zeros(
                #             (chunk_len, chunk_len), dtype=np.float32
                #         )
                # else:
                #     local_contacts = np.zeros((chunk_len, chunk_len), dtype=np.float32)

                contact_sum[start:end, start:end] += local_contacts
                contact_cnt[start:end, start:end] += 1

                # ---- token representations: (chunk_len, dim) ----
                subtoken_rep = _get_token_repr_from_results(
                    results, repr_layers=repr_layers, layer=layer, approach=approach
                )
                subtoken_repr_np = (
                    subtoken_rep.detach()
                    .cpu()
                    .numpy()[1 : chunk_len + 1]
                    .astype(np.float32, copy=False)
                )

                token_sum[start:end] += subtoken_repr_np
                token_cnt[start:end] += 1

                if end == L:
                    break

            # Finalize averages
            # Embeddings
            token_representation_np = np.zeros_like(token_sum, dtype=np.float32)
            mask_t = token_cnt > 0
            token_representation_np[mask_t] = (
                token_sum[mask_t] / token_cnt[mask_t][:, None]
            )

            # Logits
            logits_np = np.zeros_like(logits_sum, dtype=np.float32)
            mask_l = logits_cnt > 0
            logits_np[mask_l] = logits_sum[mask_l] / logits_cnt[mask_l][:, None]

            # Contacts
            contact_prob_map_np = np.zeros_like(contact_sum, dtype=np.float32)
            mask_c = contact_cnt > 0
            contact_prob_map_np[mask_c] = contact_sum[mask_c] / contact_cnt[
                mask_c
            ].astype(np.float32, copy=False)

        # Convert to tensors float32 on CPU (consistent with downstream)
        token_representation = torch.from_numpy(
            token_representation_np.astype(np.float32, copy=False)
        )
        contact_prob_map = torch.from_numpy(
            contact_prob_map_np.astype(np.float32, copy=False)
        )
        logits = torch.from_numpy(logits_np.astype(np.float32, copy=False))

        return token_representation, contact_prob_map, logits

    except Exception:
        logger.exception(
            "Error in esm_extract for sequence len %d",
            len(seq) if seq is not None else -1,
        )
        if return_nan_on_error:
            return _nan_outputs(len(seq) if seq is not None else 0)
        raise


# biopython_parser = PDBParser()

# one_to_three = {
#     "A": "ALA",
#     "C": "CYS",
#     "D": "ASP",
#     "E": "GLU",
#     "F": "PHE",
#     "G": "GLY",
#     "H": "HIS",
#     "I": "ILE",
#     "K": "LYS",
#     "L": "LEU",
#     "M": "MET",
#     "N": "ASN",
#     "P": "PRO",
#     "Q": "GLN",
#     "R": "ARG",
#     "S": "SER",
#     "T": "THR",
#     "V": "VAL",
#     "W": "TRP",
#     "Y": "TYR",
#     "B": "ASX",
#     "Z": "GLX",
#     "X": "UNK",
#     "*": " * ",
# }

# three_to_one = {}
# for _key, _value in one_to_three.items():
#     three_to_one[_value] = _key
# three_to_one["SEC"] = "C"
# three_to_one["MSE"] = "M"


# def extract_pdb_seq(protein_path: Path) -> tuple[str, str]:

#     structure = biopython_parser.get_structure("random_id", protein_path)[0]
#     seq = ""
#     chain_str = ""
#     for _, chain in enumerate(structure):
#         for _, residue in enumerate(chain):
#             if residue.get_resname() == "HOH":
#                 continue
#             c_alpha, n, c = None, None, None
#             for atom in residue:
#                 if atom.name == "CA":
#                     c_alpha = list(atom.get_vector())
#                 if atom.name == "N":
#                     n = list(atom.get_vector())
#                 if atom.name == "C":
#                     c = list(atom.get_vector())
#             if (
#                 c_alpha != None and n != None and c != None
#             ):  # only append residue if it is an amino acid and not
#                 try:
#                     seq += three_to_one[residue.get_resname()]
#                     chain_str += str(chain.id)
#                 except Exception as e:
#                     seq += "X"
#                     chain_str += str(chain.id)
#                     logger.info(
#                         "encountered unknown AA: ",
#                         residue.get_resname(),
#                         " in the complex. Replacing it with a dash X.",
#                     )

#     return seq, chain_str
