import torch

from eeg2qwerty.models import StandardizedM2


def test_standardized_m2_preserves_sentence_shapes():
    model = StandardizedM2(
        n_channels=4,
        n_subjects=3,
        width=24,
        encoder_blocks=2,
        transformer_layers=1,
        transformer_heads=2,
        transformer_ff=48,
        encoder_dropout=0.0,
        transformer_dropout=0.0,
    )
    eeg = torch.randn(5, 4, 25)
    subject_ids = torch.tensor([0, 0, 0, 1, 1])
    groups = [[0, 1, 2], [3, 4]]

    output = model(eeg, subject_ids, groups)

    assert output.logits.shape == (2, 3, 29)
    assert output.padding_mask.tolist() == [
        [False, False, False],
        [False, False, True],
    ]
