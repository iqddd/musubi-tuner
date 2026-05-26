import numpy as np
import torch

from musubi_tuner.cache_latents import set_alpha_mask_for_image_item
from musubi_tuner.dataset.image_video_dataset import ItemInfo
from musubi_tuner.utils.train_utils import apply_alpha_masked_loss


def test_set_alpha_mask_for_png_uses_alpha_channel():
    content = np.zeros((4, 4, 4), dtype=np.uint8)
    content[..., 3] = np.arange(16, dtype=np.uint8).reshape(4, 4)
    item = ItemInfo("sample.png", "", (4, 4), content=content)

    set_alpha_mask_for_image_item(item)

    assert item.alpha_mask is not None
    torch.testing.assert_close(item.alpha_mask, torch.from_numpy(content[..., 3].copy()).float() / 255.0)


def test_set_alpha_mask_for_jpg_ignores_alpha_channel():
    content = np.zeros((4, 4, 4), dtype=np.uint8)
    item = ItemInfo("sample.jpg", "", (4, 4), content=content)

    set_alpha_mask_for_image_item(item)

    assert item.alpha_mask is not None
    torch.testing.assert_close(item.alpha_mask, torch.ones((4, 4), dtype=torch.float32))


def test_apply_alpha_masked_loss_for_image_loss():
    loss = torch.ones((1, 2, 2, 2))
    alpha_mask = torch.tensor([[[1.0, 0.0], [0.5, 1.0]]])

    masked_loss = apply_alpha_masked_loss(loss, {"alpha_mask": alpha_mask})

    expected = alpha_mask.unsqueeze(1).expand_as(loss)
    torch.testing.assert_close(masked_loss, expected)


def test_apply_alpha_masked_loss_for_video_loss():
    loss = torch.ones((1, 3, 2, 2, 2))
    alpha_mask = torch.tensor([[[[1.0, 0.0], [0.5, 1.0]], [[0.0, 1.0], [1.0, 0.5]]]])

    masked_loss = apply_alpha_masked_loss(loss, {"alpha_mask": alpha_mask})

    expected = alpha_mask.unsqueeze(1).expand_as(loss)
    torch.testing.assert_close(masked_loss, expected)


def test_apply_alpha_masked_loss_for_layered_loss():
    loss = torch.ones((1, 2, 3, 2, 2))
    alpha_mask = torch.tensor([[[[1.0, 0.0], [0.5, 1.0]], [[0.0, 1.0], [1.0, 0.5]]]])

    masked_loss = apply_alpha_masked_loss(loss, {"alpha_mask": alpha_mask})

    expected = alpha_mask.unsqueeze(2).expand_as(loss)
    torch.testing.assert_close(masked_loss, expected)
