import torch
import torch.nn as nn


class MockAttentionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_idx = 0

    def forward(self, hidden_states, past_key_values=None, **kwargs):
        batch, seq, _dim = hidden_states.shape
        output = hidden_states.clone()
        present_kv = torch.zeros(2, batch, 4, seq, 16)
        return output, present_kv


def test_pre_hook_fires_on_forward():
    module = MockAttentionModule()
    pre_hook_calls = []

    def pre_hook(mod, inputs, kwargs):
        pre_hook_calls.append(
            {"module": mod, "inputs": inputs, "kwargs": kwargs}
        )

    handle = module.register_forward_pre_hook(pre_hook, with_kwargs=True)

    x = torch.zeros(1, 4, 32)
    pkv = (torch.zeros(2, 1, 4, 2, 16), torch.zeros(2, 1, 4, 2, 16))
    module(x, past_key_values=pkv)

    assert len(pre_hook_calls) == 1
    assert pre_hook_calls[0]["module"] is module
    assert pre_hook_calls[0]["inputs"][0].shape == x.shape
    assert "past_key_values" in pre_hook_calls[0]["kwargs"]

    handle.remove()


def test_post_hook_fires_on_forward():
    module = MockAttentionModule()
    post_hook_calls = []

    def post_hook(mod, inputs, output):
        post_hook_calls.append({"module": mod, "output": output})

    handle = module.register_forward_hook(post_hook)

    x = torch.zeros(1, 4, 32)
    module(x)

    assert len(post_hook_calls) == 1
    output = post_hook_calls[0]["output"]
    assert isinstance(output, tuple)
    assert len(output) == 2

    handle.remove()


def test_post_hook_can_access_kv_cache():
    module = MockAttentionModule()
    captured_kv = []

    def kv_capture_hook(mod, inputs, output):
        if isinstance(output, tuple) and len(output) >= 2:
            captured_kv.append(output[1])

    handle = module.register_forward_hook(kv_capture_hook)

    x = torch.zeros(1, 4, 32)
    module(x)

    assert len(captured_kv) == 1
    assert captured_kv[0].shape == (2, 1, 4, 4, 16)

    handle.remove()


def test_pre_hook_can_inject_inputs():
    module = MockAttentionModule()

    def input_injection_hook(mod, inputs):
        hidden_states = inputs[0]
        modified = hidden_states * 2
        return (modified,)

    handle = module.register_forward_pre_hook(input_injection_hook)

    x = torch.ones(1, 4, 32)
    output, _ = module(x)

    assert torch.allclose(output, x * 2)

    handle.remove()


def test_multiple_hooks_on_same_module():
    module = MockAttentionModule()
    call_order = []

    def hook1(mod, inputs):
        call_order.append("pre1")

    def hook2(mod, inputs):
        call_order.append("pre2")

    h1 = module.register_forward_pre_hook(hook1)
    h2 = module.register_forward_pre_hook(hook2)

    module(torch.zeros(1, 4, 32))

    assert "pre1" in call_order
    assert "pre2" in call_order

    h1.remove()
    h2.remove()


def test_hook_registration_on_nested_modules():
    class MockTransformerLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = MockAttentionModule()
            self.mlp = nn.Linear(32, 32)

        def forward(self, x):
            attn_out, _ = self.self_attn(x)
            return self.mlp(attn_out)

    model = MockTransformerLayer()
    hooked_modules = []

    for name, module in model.named_modules():
        if isinstance(module, MockAttentionModule):

            def hook(mod, inputs, output, name=name):
                hooked_modules.append(name)

            module.register_forward_hook(hook)

    model(torch.zeros(1, 4, 32))

    assert "self_attn" in hooked_modules
