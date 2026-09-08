"""Fork 1 objectives, pinned at the level where they could silently be wrong.

The failure this file exists to prevent is a contrastive loss that trains smoothly, reports a
falling curve, and has learned nothing -- either because self-similarity supplied the gradient
or because the batch contained no positive pair to begin with.
"""

import numpy as np
import pytest
import torch

from cognate.contrastive import (
    ProjectionHead,
    encode_labels,
    fit_tcr_metric,
    project,
    projected_cache,
    pxk_batches,
    supervised_info_nce,
)
from cognate.embed import EmbeddingCache

TEMPERATURE = 0.1


def unit(rows: list[list[float]]) -> torch.Tensor:
    tensor = torch.tensor(rows, dtype=torch.float32)
    return tensor / tensor.norm(dim=-1, keepdim=True)


def test_projection_head_outputs_unit_vectors():
    head = ProjectionHead(16, hidden=8, output=4)
    output = head(torch.randn(5, 16))
    assert output.shape == (5, 4)
    assert output.norm(dim=-1).detach().numpy() == pytest.approx(np.ones(5), abs=1e-5)


def test_loss_is_lower_when_same_label_points_are_together():
    labels = torch.tensor([0, 0, 1, 1])
    clustered = unit([[1, 0], [1, 0.05], [0, 1], [0.05, 1]])
    scattered = unit([[1, 0], [0, 1], [1, 0.05], [0.05, 1]])
    together = supervised_info_nce(
        clustered, clustered, labels, labels, TEMPERATURE, exclude_self=True
    )
    apart = supervised_info_nce(
        scattered, scattered, labels, labels, TEMPERATURE, exclude_self=True
    )
    assert float(together) < float(apart)


def test_self_similarity_hides_a_failure_the_diagonal_mask_exposes():
    """A same-label pair pulled apart, with a different-label point sitting next to one of
    them: the worst arrangement the loss should be able to see. Counting each point as its own
    positive reports it as substantially better than it is."""
    labels = torch.tensor([0, 0, 1])
    scattered = unit([[1.0, 0.0], [0.0, 1.0], [1.0, 0.05]])
    with_self = supervised_info_nce(
        scattered, scattered, labels, labels, TEMPERATURE, exclude_self=False
    )
    without_self = supervised_info_nce(
        scattered, scattered, labels, labels, TEMPERATURE, exclude_self=True
    )
    assert float(with_self) < float(without_self)


def test_masked_loss_still_separates_clustered_from_scattered():
    """The mask must not flatten the loss -- with self excluded it still has to rank a good
    arrangement far below a bad one."""
    labels = torch.tensor([0, 0, 1])
    clustered = unit([[1.0, 0.0], [1.0, 0.03], [0.0, 1.0]])
    scattered = unit([[1.0, 0.0], [0.0, 1.0], [1.0, 0.05]])
    good = supervised_info_nce(
        clustered, clustered, labels, labels, TEMPERATURE, exclude_self=True
    )
    bad = supervised_info_nce(
        scattered, scattered, labels, labels, TEMPERATURE, exclude_self=True
    )
    assert float(good) < 0.01 < float(bad)


def test_no_positive_pair_contributes_no_loss_rather_than_zero_loss():
    labels = torch.tensor([0, 1])
    points = unit([[1.0, 0.0], [0.0, 1.0]])
    loss = supervised_info_nce(
        points, points, labels, labels, TEMPERATURE, exclude_self=True
    )
    assert float(loss) == 0.0


def test_pxk_batches_have_p_peptides_of_k_tcrs_each():
    labels = np.repeat(np.arange(10), 5)
    rng = np.random.default_rng(0)
    batches = list(
        pxk_batches(
            labels, peptides_per_batch=4, tcrs_per_peptide=3, batches_per_epoch=6, rng=rng
        )
    )
    assert len(batches) == 6
    for batch in batches:
        assert len(batch) == 12
        counts = np.unique(labels[batch], return_counts=True)[1]
        assert counts.tolist() == [3, 3, 3, 3]


def test_pxk_skips_peptides_that_cannot_form_a_positive_pair():
    labels = np.array([0, 0, 1, 2, 3])
    rng = np.random.default_rng(0)
    batch = next(
        pxk_batches(
            labels, peptides_per_batch=4, tcrs_per_peptide=2, batches_per_epoch=1, rng=rng
        )
    )
    assert set(labels[batch].tolist()) == {0}


def test_pxk_raises_when_no_positive_pair_exists_anywhere():
    with pytest.raises(ValueError, match="two or more TCRs"):
        next(pxk_batches(np.arange(5), batches_per_epoch=1, rng=np.random.default_rng(0)))


def test_loss_decreases_on_a_memorizable_batch():
    """The epic's required check: a tiny, perfectly separable problem must be learnable."""
    rng = np.random.default_rng(0)
    centres = rng.normal(size=(4, 12)) * 3.0
    labels = np.repeat(np.arange(4), 8)
    features = centres[labels] + rng.normal(size=(32, 12)) * 0.05

    _, _, history = fit_tcr_metric(
        features, labels, features, labels,
        hidden=16, output=8, max_epochs=30, patience=30,
        peptides_per_batch=4, tcrs_per_peptide=4, batches_per_epoch=8, seed=0,
    )
    assert history.train_loss[-1] < history.train_loss[0]
    assert history.val_loss[-1] < history.val_loss[0]


def test_projected_cache_keeps_the_index_and_normalises_rows():
    sequences = np.array(["AAA", "CCC", "GGG"], dtype=object)
    cache = EmbeddingCache(
        model_name="toy",
        sequences=sequences,
        layers=np.random.default_rng(0).normal(size=(1, 3, 12)).astype(np.float32),
        index={s: i for i, s in enumerate(sequences)},
    )
    head = ProjectionHead(12, hidden=8, output=4)
    standardisation = (np.zeros(12), np.ones(12))
    projected = projected_cache(cache, 0, head, standardisation)

    assert projected.index == cache.index
    assert projected.sequences.tolist() == cache.sequences.tolist()
    assert projected.layers.shape == (1, 3, 4)
    norms = np.linalg.norm(projected.layer(0), axis=1)
    assert norms == pytest.approx(np.ones(3), abs=1e-5)


def test_encode_labels_is_stable_and_contiguous():
    labels = encode_labels(["B", "A", "B", "C"])
    assert labels.tolist() == [1, 0, 1, 2]


def test_project_is_deterministic():
    head = ProjectionHead(8, hidden=4, output=4)
    features = np.random.default_rng(0).normal(size=(6, 8))
    standardisation = (np.zeros(8), np.ones(8))
    first = project(head, features, standardisation)
    second = project(head, features, standardisation)
    assert first == pytest.approx(second)
