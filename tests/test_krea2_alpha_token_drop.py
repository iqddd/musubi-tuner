import copy
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from musubi_tuner.krea2.krea2_mmdit import (
    SingleMMDiTConfig,
    SingleStreamDiT,
    pack_valid_prefix,
    unpack_packed_sequence,
)
from musubi_tuner.krea2_train_network import (
    Krea2NetworkTrainer,
    align_alpha_mask_to_token_grid,
    make_alpha_token_keep_mask,
)
from musubi_tuner.utils.train_utils import apply_alpha_masked_loss


def test_pack_valid_prefix_is_stable_and_round_trips_gradients():
    sequence = torch.arange(2 * 7 * 2, dtype=torch.float32).reshape(2, 7, 2).requires_grad_()
    pos = torch.arange(2 * 7 * 3, dtype=torch.float32).reshape(2, 7, 3)
    valid = torch.tensor(
        [[True, False, True, False, True, True, False], [False, True, True, False, False, True, False]]
    )

    packed, packed_pos, packed_mask, permutation, original_length = pack_valid_prefix(
        sequence, pos, valid, pad_multiple=4
    )
    assert packed.shape == (2, 4, 2)
    assert packed_mask.tolist() == [[True] * 4, [True] * 3 + [False]]
    assert torch.equal(packed[0], sequence.detach()[0, [0, 2, 4, 5]])
    assert torch.equal(packed_pos[0], pos[0, [0, 2, 4, 5]])

    restored = unpack_packed_sequence(packed, packed_mask, permutation, original_length)
    assert torch.equal(restored[valid], sequence.detach()[valid])
    assert torch.count_nonzero(restored[~valid]) == 0
    restored.square().sum().backward()
    assert sequence.grad[valid].abs().sum() > 0
    assert torch.count_nonzero(sequence.grad[~valid]) == 0


def test_alpha_keep_mask_preserves_soft_positive_alpha_and_rejects_mixed_tokens():
    alpha = torch.ones(1, 32, 32)
    alpha[:, :16, :16] = 0
    alpha[:, :16, 16:] = 1 / 255
    keep = make_alpha_token_keep_mask(alpha, (4, 4), patch=2, device=torch.device("cpu"))
    assert keep.tolist() == [[False, True, True, True]]

    alpha[:, 8:16, :16] = 1
    with pytest.raises(ValueError, match="both exact-zero and positive"):
        make_alpha_token_keep_mask(alpha, (4, 4), patch=2, device=torch.device("cpu"))


@pytest.mark.parametrize("frame_axis", [False, True])
def test_snap16_excludes_only_touched_cells_and_preserves_soft_weights(frame_axis):
    alpha = torch.full((2, 32, 48), 1 / 255)
    alpha[0, 15, 15] = 0
    alpha[1, 16, 32] = 0
    expected = alpha.clone()
    expected[0, :16, :16] = 0
    expected[1, 16:, 32:] = 0
    if frame_axis:
        alpha, expected = alpha[:, None], expected[:, None]
    original = alpha.clone()
    aligned = align_alpha_mask_to_token_grid(alpha, (4, 6), 2)
    assert torch.equal(aligned, expected)
    assert torch.equal(alpha, original)
    assert torch.equal(align_alpha_mask_to_token_grid(aligned, (4, 6), 2), aligned)
    keep = make_alpha_token_keep_mask(aligned, (4, 6), 2, torch.device("cpu"))
    assert keep.tolist() == [[False, True, True, True, True, True], [True, True, True, True, True, False]]


@pytest.mark.parametrize("enabled,has_alpha", [(True, True), (False, True), (True, False)])
def test_trainer_shares_snap16_mask_between_loss_and_token_drop(enabled, has_alpha):
    class Model:
        config = SimpleNamespace(patch=2)

        def __call__(self, **kwargs):
            self.image_mask = kwargs["image_mask"]
            return kwargs["img"]

    model = Model()
    accelerator = SimpleNamespace(device=torch.device("cpu"), autocast=nullcontext)
    args = SimpleNamespace(alpha_masked_token_drop=enabled, gradient_checkpointing=False)
    latents = torch.zeros(1, 2, 1, 4, 4)
    alpha = torch.full((1, 32, 32), 1 / 255)
    alpha[0, 15, 15] = 0
    batch = {"latents": latents, "krea2_vl_embed": [torch.zeros(3, 2, 32)]}
    if has_alpha:
        batch["alpha_mask"] = alpha
    output = Krea2NetworkTrainer.call_dit(
        None, args, accelerator, model, latents, batch, torch.ones_like(latents),
        latents, torch.tensor([500.0]), torch.float32,
    )
    if enabled and has_alpha:
        assert model.image_mask.tolist() == [[False, True, True, True]]
        weights = apply_alpha_masked_loss(torch.ones_like(output.pred), batch)
        assert torch.count_nonzero(weights[..., :2, :2]) == 0
        torch.testing.assert_close(weights[..., 2:, :], torch.full_like(weights[..., 2:, :], 1 / 255))
        assert torch.count_nonzero(alpha == 0) == 1  # cached input was not mutated
    else:
        assert model.image_mask is None
        if has_alpha:
            assert batch["alpha_mask"] is alpha
        else:
            assert "alpha_mask" not in batch


def _tiny_model():
    config = SingleMMDiTConfig(
        features=32,
        tdim=16,
        txtdim=32,
        heads=2,
        kvheads=2,
        multiplier=1,
        layers=2,
        patch=2,
        channels=2,
        txtlayers=2,
        txtheads=2,
        txtkvheads=2,
    )
    return SingleStreamDiT(config, attn_mode="torch").float().train()


def _inputs():
    torch.manual_seed(123)
    img = torch.randn(2, 6, 8)
    context = torch.randn(2, 4, 2, 32)
    t = torch.tensor([0.25, 0.75])
    imgpos = torch.tensor(
        [[[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0], [0, 1, 1], [0, 1, 2]]] * 2,
        dtype=torch.float32,
    )
    txtpos = torch.zeros(2, 4, 3)
    txtmask = torch.tensor([[True, True, True, False], [True, True, False, False]])
    mask = torch.cat((torch.ones(2, 6, dtype=torch.bool), txtmask), dim=1)
    pos = torch.cat((imgpos, txtpos), dim=1)
    keep = torch.tensor(
        [[True, False, True, True, False, True], [False, True, True, False, True, False]]
    )
    return img, context, t, pos, mask, keep


def _per_image_reference(model, img, context, t, pos, mask, keep):
    outputs = []
    image_len = img.shape[1]
    for index in range(img.shape[0]):
        image_indices = keep[index].nonzero().flatten()
        selected = torch.cat((image_indices, torch.arange(image_len, pos.shape[1])))
        prediction = model(
            img=img[index : index + 1, image_indices],
            context=context[index : index + 1],
            t=t[index : index + 1],
            pos=pos[index : index + 1, selected],
            mask=mask[index : index + 1, selected],
        )
        full = prediction.new_zeros(1, image_len, prediction.shape[-1])
        outputs.append(full.scatter(1, image_indices[None, :, None].expand_as(prediction), prediction))
    return torch.cat(outputs)


def _run_with_grads(model, batched):
    img, context, t, pos, mask, keep = _inputs()
    img.requires_grad_()
    context.requires_grad_()
    if batched:
        output = model(img=img, context=context, t=t, pos=pos, mask=mask, image_mask=keep)
    else:
        output = _per_image_reference(model, img, context, t, pos, mask, keep)
    loss = output.square().mean()
    loss.backward()
    parameter_grads = {name: value.grad.detach().clone() for name, value in model.named_parameters() if value.grad is not None}
    return output.detach(), loss.detach(), img.grad.detach(), context.grad.detach(), parameter_grads


def test_batched_token_drop_matches_per_image_outputs_and_gradients():
    torch.manual_seed(7)
    reference_model = _tiny_model()
    batched_model = copy.deepcopy(reference_model)
    reference = _run_with_grads(reference_model, batched=False)
    batched = _run_with_grads(batched_model, batched=True)

    for reference_tensor, batched_tensor in zip(reference[:4], batched[:4]):
        torch.testing.assert_close(batched_tensor, reference_tensor, atol=2e-5, rtol=2e-5)
    keep = _inputs()[-1]
    assert torch.count_nonzero(batched[0][~keep]) == 0
    assert torch.count_nonzero(batched[2][~keep]) == 0
    assert reference[4].keys() == batched[4].keys()
    for name in reference[4]:
        torch.testing.assert_close(batched[4][name], reference[4][name], atol=3e-5, rtol=3e-5)


def test_all_valid_mask_matches_legacy_forward():
    torch.manual_seed(11)
    model = _tiny_model().eval()
    img, context, t, pos, mask, _ = _inputs()
    with torch.no_grad():
        legacy = model(img=img, context=context, t=t, pos=pos, mask=mask)
        packed = model(
            img=img,
            context=context,
            t=t,
            pos=pos,
            mask=mask,
            image_mask=torch.ones(img.shape[:2], dtype=torch.bool),
        )
    torch.testing.assert_close(packed, legacy, atol=2e-5, rtol=2e-5)
