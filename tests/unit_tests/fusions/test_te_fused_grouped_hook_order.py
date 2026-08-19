# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import warnings

import pytest
import torch

from megatron.core.extensions import transformer_engine as te_ext
from megatron.core.utils import is_te_min_version

pytestmark = [
    pytest.mark.skipif(not te_ext.HAVE_TE, reason="Transformer Engine is not available"),
    pytest.mark.skipif(
        not is_te_min_version("2.14.0"),
        reason="Grouped dense fused MLP requires Transformer Engine >= 2.14.0",
    ),
]


def test_grouped_tp1_forwards_source_hooks_at_their_execution_boundaries(monkeypatch):
    monkeypatch.setattr(te_ext, "get_tensor_model_parallel_world_size", lambda: 1)

    module = te_ext.TEFusedMLPWithGroupedLinear.__new__(te_ext.TEFusedMLPWithGroupedLinear)
    torch.nn.Module.__init__(module)
    module.linear_fc1 = torch.nn.Linear(2, 2)
    module.linear_fc1.inner = torch.nn.Identity()
    module.linear_fc2 = torch.nn.Linear(2, 2)
    events = []

    class RecordingIdentity(torch.nn.Module):

        def __init__(self, name):
            super().__init__()
            self.name = name

        def forward(self, inputs):
            events.append(self.name)
            return inputs

    norm_impl = RecordingIdentity("norm")
    fused_impl = RecordingIdentity("fused")
    module._norm_seq = (norm_impl,)

    def post_hook(name):
        return lambda _module, _inputs, _output: events.append(name)

    module.linear_fc1.register_forward_hook(post_hook("fc1-post"))
    module.linear_fc1.inner.register_forward_hook(post_hook("fc1-inner-post"))
    module.linear_fc2.register_forward_hook(post_hook("fc2-post"))

    with pytest.warns(UserWarning, match="post-forward hook"):
        module._register_hooks_on_fused_impl(fused_impl)

    def pre_hook(name):
        return lambda _module, _inputs: events.append(name)

    module.linear_fc1.register_forward_pre_hook(pre_hook("fc1-pre"))
    module.linear_fc1.inner.register_forward_pre_hook(pre_hook("fc1-inner-pre"))
    module.linear_fc2.register_forward_pre_hook(pre_hook("fc2-pre"))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        norm_impl(torch.ones(1, 2))
        fused_impl(torch.ones(1, 2))

    assert events == [
        "fc1-pre",
        "fc1-inner-pre",
        "norm",
        "fc2-pre",
        "fused",
        "fc1-post",
        "fc1-inner-post",
        "fc2-post",
    ]
