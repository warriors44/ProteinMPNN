from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pytest
import torch

from protein_mpnn_lo_utils import ProteinMPNN_LO


def _format_pdb_atom_line(
    atom_serial: int,
    atom_name: str,
    res_name: str,
    chain_id: str,
    res_seq: int,
    x: float,
    y: float,
    z: float,
) -> str:
    """Format a minimal ATOM line in PDB fixed-width format."""
    # Columns used by parse_PDB_biounits:
    # [0:4] "ATOM", [12:16] atom name, [17:20] resname, [21:22] chain,
    # [22:26] resseq, [30:38],[38:46],[46:54] coords
    return (
        f"ATOM  {atom_serial:5d} {atom_name:<4s} {res_name:>3s} {chain_id:1s}"
        f"{res_seq:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {atom_name[0]:>2s}\n"
    )


def _write_minimal_pdb(path: Path, n_res: int = 5, chain_id: str = "A") -> None:
    """Write a tiny single-chain PDB with N/CA/C/O atoms."""
    res_names = ["ALA", "GLY", "SER", "THR", "VAL"]
    atom_names = ["N", "CA", "C", "O"]
    serial = 1
    lines: list[str] = []
    for i in range(n_res):
        res_name = res_names[i % len(res_names)]
        res_seq = i + 1
        # Simple backbone trace with small offsets
        base_x = float(i) * 1.5
        for a_i, atom in enumerate(atom_names):
            lines.append(
                _format_pdb_atom_line(
                    atom_serial=serial,
                    atom_name=atom,
                    res_name=res_name,
                    chain_id=chain_id,
                    res_seq=res_seq,
                    x=base_x + 0.1 * a_i,
                    y=0.2 * a_i,
                    z=0.3 * a_i,
                )
            )
            serial += 1
    lines.append("TER\nEND\n")
    path.write_text("".join(lines))


@dataclass(frozen=True)
class _Args:
    """Namespace-like args container for protein_mpnn_lo_run.main()."""

    suppress_print: int = 1
    ca_only: bool = False
    path_to_model_weights: str = ""
    model_name: str = "test_model"
    use_soluble_model: bool = False
    seed: int = 123
    save_score: int = 0
    save_probs: int = 0
    score_only: int = 1
    path_to_fasta: str = ""
    conditional_probs_only: int = 0
    conditional_probs_only_backbone: int = 0
    unconditional_probs_only: int = 0
    backbone_noise: float = 0.0
    num_seq_per_target: int = 1
    batch_size: int = 1
    max_length: int = 1000
    sampling_temp: str = "0.1"
    out_folder: str = ""
    pdb_path: str = ""
    pdb_path_chains: str = ""
    jsonl_path: Optional[str] = None
    chain_id_jsonl: str = ""
    fixed_positions_jsonl: str = ""
    omit_AAs: list[str] = None  # type: ignore[assignment]
    bias_AA_jsonl: str = ""
    bias_by_res_jsonl: str = ""
    omit_AA_jsonl: str = ""
    pssm_jsonl: str = ""
    pssm_multi: float = 0.0
    pssm_threshold: float = 0.0
    pssm_log_odds_flag: int = 0
    pssm_bias_flag: int = 0
    tied_positions_jsonl: str = ""
    order_temperature: float = 1.0
    num_lo_samples: int = 2
    num_is_samples: int = 4
    use_is_scoring: bool = True


@pytest.mark.integration
def test_lo_run_score_only_smoke(tmp_path: Path) -> None:
    """Run protein_mpnn_lo_run.main() end-to-end in score_only mode.

    This is a lightweight integration smoke test that:
    - writes a minimal PDB
    - writes a minimal checkpoint compatible with protein_mpnn_lo_run
    - runs score_only with IS scoring
    """
    # Prepare minimal PDB
    pdb_path = tmp_path / "mini.pdb"
    _write_minimal_pdb(pdb_path, n_res=6, chain_id="A")

    # Prepare minimal checkpoint at expected location
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    model_name = "test_model"

    model = ProteinMPNN_LO(
        ca_only=False,
        num_letters=21,
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        augment_eps=0.0,
        k_neighbors=64,
        num_samples=2,
        separate_q_decoder=False,
    )
    ckpt = {
        "model_state_dict": model.state_dict(),
        "num_edges": 64,
        "noise_level": 0.0,
        "num_samples": 2,
    }
    torch.save(ckpt, weights_dir / f"{model_name}.pt")

    # Run script main()
    from protein_mpnn_lo_run import main

    args = _Args(
        path_to_model_weights=str(weights_dir) + "/",
        model_name=model_name,
        out_folder=str(tmp_path / "out"),
        pdb_path=str(pdb_path),
        omit_AAs=list("X"),
    )
    main(args)  # should complete without raising

