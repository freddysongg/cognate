"""Fork 2 correctness, at the points where a masking bug would be invisible.

A padding bug in this model does not crash and does not look wrong: it produces a slightly
different number for every row, trains happily, and quietly turns sequence length into a
feature. These tests pin the two invariants that rule that out -- padding must not change any
output, and pooling must divide by the true length -- plus the ablation property that makes 2a
minus 2b mean anything.
"""

import numpy as np
import pytest
import torch

from cognate.crossattn import (
    ResidueBindingModel,
    masked_mean,
    pad_residues,
)
from cognate.embed import ResidueCache

HIDDEN = 8


def make_residue_cache(sequences: dict[str, int], seed: int = 0) -> ResidueCache:
    rng = np.random.default_rng(seed)
    names = sorted(sequences)
    lengths = np.array([sequences[s] for s in names], dtype=np.int64)
    offsets = np.zeros(len(names) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    residues = rng.normal(size=(1, int(offsets[-1]), HIDDEN)).astype(np.float32)
    return ResidueCache(
        model_name="toy",
        sequences=np.array(names, dtype=object),
        layer_indices=np.array([0], dtype=np.int64),
        offsets=offsets,
        residues=residues,
        index={s: i for i, s in enumerate(names)},
    )


def batch_from(values: np.ndarray, mask: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.from_numpy(values), torch.from_numpy(mask)


def test_masked_mean_divides_by_true_length_not_padded_width():
    values = torch.tensor([[[2.0], [4.0], [0.0]]])
    mask = torch.tensor([[True, True, False]])
    assert float(masked_mean(values, mask)) == pytest.approx(3.0)


def test_masked_mean_ignores_whatever_sits_in_the_padding():
    values = torch.tensor([[[2.0], [4.0], [999.0]]])
    mask = torch.tensor([[True, True, False]])
    assert float(masked_mean(values, mask)) == pytest.approx(3.0)


def test_pad_residues_shapes_and_mask():
    cache = make_residue_cache({"AAA": 3, "CCCCC": 5, "GG": 2})
    padded = pad_residues(cache, ["AAA", "CCCCC", "GG"], 0)
    assert padded.values.shape == (3, 5, HIDDEN)
    assert padded.mask.sum(axis=1).tolist() == [3, 5, 2]
    assert padded.mask[padded.index["GG"]].tolist() == [True, True, False, False, False]


def test_pad_residues_zeroes_the_padding():
    cache = make_residue_cache({"AAA": 3, "CCCCC": 5})
    padded = pad_residues(cache, ["AAA", "CCCCC"], 0)
    assert not padded.values[padded.index["AAA"], 3:].any()


@pytest.mark.parametrize("use_attention", [True, False])
def test_extra_padding_does_not_change_outputs(use_attention):
    """The epic's required check, for both variants.

    Same sequences, two different padded widths. Every score must be identical, or the model is
    reading the pad positions.
    """
    torch.manual_seed(0)
    model = ResidueBindingModel(HIDDEN, use_attention=use_attention, width=8, heads=2)
    model.eval()

    peptide_lengths, tcr_lengths = [3, 5], [2, 6]

    def build(lengths: list[int], width: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        local = np.random.default_rng(seed)
        values = np.zeros((len(lengths), width, HIDDEN), dtype=np.float32)
        mask = np.zeros((len(lengths), width), dtype=bool)
        for row, length in enumerate(lengths):
            values[row, :length] = local.normal(size=(length, HIDDEN))
            mask[row, :length] = True
        return batch_from(values, mask)

    tight_peptide, tight_peptide_mask = build(peptide_lengths, max(peptide_lengths), 1)
    tight_tcr, tight_tcr_mask = build(tcr_lengths, max(tcr_lengths), 2)
    wide_peptide, wide_peptide_mask = build(peptide_lengths, max(peptide_lengths) + 7, 1)
    wide_tcr, wide_tcr_mask = build(tcr_lengths, max(tcr_lengths) + 9, 2)

    with torch.no_grad():
        tight = model(tight_peptide, tight_peptide_mask, tight_tcr, tight_tcr_mask)
        wide = model(wide_peptide, wide_peptide_mask, wide_tcr, wide_tcr_mask)
    assert tight.numpy() == pytest.approx(wide.numpy(), abs=1e-5)


@pytest.mark.parametrize("use_attention", [True, False])
def test_garbage_in_the_padding_does_not_change_outputs(use_attention):
    """Padding is zero in practice, so a mask bug could hide behind that. Poison it."""
    torch.manual_seed(0)
    model = ResidueBindingModel(HIDDEN, use_attention=use_attention, width=8, heads=2)
    model.eval()

    rng = np.random.default_rng(0)
    values = np.zeros((2, 6, HIDDEN), dtype=np.float32)
    mask = np.zeros((2, 6), dtype=bool)
    for row, length in enumerate([2, 4]):
        values[row, :length] = rng.normal(size=(length, HIDDEN))
        mask[row, :length] = True
    poisoned = values.copy()
    poisoned[~mask] = 1e3

    clean_t, mask_t = batch_from(values, mask)
    poisoned_t, _ = batch_from(poisoned, mask)
    with torch.no_grad():
        clean = model(clean_t, mask_t, clean_t, mask_t)
        dirty = model(poisoned_t, mask_t, poisoned_t, mask_t)
    assert clean.numpy() == pytest.approx(dirty.numpy(), abs=1e-4)


def test_four_residue_sequences_are_handled():
    """Two evaluation CDR3b are 4-mers, shorter than the 5-23 the work order assumes."""
    cache = make_residue_cache({"AGGC": 4, "CASSLGQYF": 9})
    padded = pad_residues(cache, ["AGGC", "CASSLGQYF"], 0)
    assert padded.mask.sum(axis=1).tolist() == [4, 9]

    model = ResidueBindingModel(HIDDEN, use_attention=True, width=8, heads=2)
    model.eval()
    values, mask = batch_from(padded.values, padded.mask)
    with torch.no_grad():
        scores = model(values, mask, values, mask)
    assert scores.shape == (2,)
    assert np.isfinite(scores.numpy()).all()


def test_the_two_variants_differ_only_by_the_attention_module():
    with_attention = ResidueBindingModel(HIDDEN, use_attention=True, width=8, heads=2)
    without = ResidueBindingModel(HIDDEN, use_attention=False, width=8, heads=2)
    assert with_attention.attention is not None
    assert without.attention is None

    shared = {"peptide_projection", "tcr_projection", "head"}
    for name in shared:
        left = dict(getattr(with_attention, name).named_parameters())
        right = dict(getattr(without, name).named_parameters())
        assert left.keys() == right.keys()
        for key in left:
            assert left[key].shape == right[key].shape


def test_attention_changes_the_output_it_is_supposed_to_change():
    """If 2a and 2b produced identical scores the ablation would be measuring nothing."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    values = rng.normal(size=(3, 5, HIDDEN)).astype(np.float32)
    mask = np.ones((3, 5), dtype=bool)
    values_t, mask_t = batch_from(values, mask)

    torch.manual_seed(0)
    attentive = ResidueBindingModel(HIDDEN, use_attention=True, width=8, heads=2).eval()
    torch.manual_seed(0)
    pooled = ResidueBindingModel(HIDDEN, use_attention=False, width=8, heads=2).eval()
    with torch.no_grad():
        a = attentive(values_t, mask_t, values_t, mask_t).numpy()
        b = pooled(values_t, mask_t, values_t, mask_t).numpy()
    assert not np.allclose(a, b)
