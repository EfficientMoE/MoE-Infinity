import torch

from moe_infinity.utils.arguments import copy_args_to_device, copy_kwargs_to_device


def test_copy_args_to_device_with_nested_structures():
    device = torch.device("cpu")
    tensor_a = torch.randn(2, 3)
    tensor_b = torch.randn(4)

    args = (
        tensor_a,
        [tensor_b, 7],
        ("keep", {"inner": tensor_a + 1}),
        {"flag": True, "value": tensor_b * 2},
    )

    copied = copy_args_to_device(device, args)

    assert isinstance(copied, tuple)
    assert torch.allclose(copied[0], tensor_a)
    assert copied[1][1] == 7
    assert copied[2][0] == "keep"
    assert torch.allclose(copied[2][1]["inner"], tensor_a + 1)
    assert copied[3]["flag"] is True
    assert torch.allclose(copied[3]["value"], tensor_b * 2)


def test_copy_kwargs_to_device_only_moves_tensors():
    device = torch.device("cpu")
    tensor = torch.randn(3, 3)
    kwargs = {
        "tensor": tensor,
        "nested": (tensor + 1, 5),
        "name": "test",
    }

    copied = copy_kwargs_to_device(device, kwargs)

    assert copied["name"] == "test"
    assert torch.allclose(copied["tensor"], tensor)
    assert torch.allclose(copied["nested"][0], tensor + 1)
    assert copied["nested"][1] == 5