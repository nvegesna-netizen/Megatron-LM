# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.pipeline_parallel.schedules import custom_backward, deallocate_output_tensor
from megatron.core.tensor_parallel.random import (
    HAVE_TE,
    initialize_rng_tracker,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer.cuda_graphs import (
    CudaGraphManager,
    _CudagraphGlobalRecord,
    create_cudagraphs,
)
from megatron.core.transformer.enums import CudaGraphModule
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version
from tests.unit_tests.test_utilities import Utils


@pytest.mark.skipif(
    not (HAVE_TE and is_te_min_version("1.5.0")),
    reason="use_te_rng_tracker requires TransformerEngine version >= 1.5",
)
class TestLocalCudagraphPipeline:
    def setup_method(self, method):
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=2, pipeline_model_parallel_size=2
        )
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()
        _CudagraphGlobalRecord.cudagraph_created = False
        _CudagraphGlobalRecord.cudagraph_record = []
        CudaGraphManager.global_mempool = None

    @staticmethod
    def _make_config(**kwargs):
        return TransformerConfig(
            num_layers=4,
            hidden_size=64,
            num_attention_heads=4,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            cuda_graph_impl="local",
            cuda_graph_warmup_steps=1,
            use_cpu_initialization=True,
            **kwargs,
        )

    def test_dense_mlp_scope_constructs(self):
        config = self._make_config(cuda_graph_modules=[CudaGraphModule.mlp])

        block = TransformerBlock(config, get_gpt_layer_with_transformer_engine_spec())

        assert len(block.layers) > 0
        assert all(hasattr(layer, "cudagraph_manager") for layer in block.layers)
        assert all(not layer.is_moe_layer for layer in block.layers)

    def test_last_replay_output_supports_pipeline_deallocation(self):
        config = self._make_config(
            cuda_graph_modules=[CudaGraphModule.attn],
            deallocate_pipeline_outputs=True,
        )
        block = TransformerBlock(config, get_gpt_layer_with_transformer_engine_spec()).cuda()
        block.train()
        for param in block.parameters():
            param.main_grad = torch.zeros_like(param)

        sequence_length = 32
        hidden_states = torch.randn(
            (sequence_length, 1, config.hidden_size),
            device="cuda",
            requires_grad=True,
        )
        attention_mask = torch.ones(
            (1, 1, sequence_length, sequence_length), dtype=bool, device="cuda"
        )

        eager_out = block(hidden_states=hidden_states, attention_mask=attention_mask)
        eager_out.sum().backward()
        create_cudagraphs()

        graphed_out = block(hidden_states=hidden_states, attention_mask=attention_mask)
        assert graphed_out.shape == eager_out.shape
        assert torch.isfinite(graphed_out).all()
        assert graphed_out._base is None

        graphed_grad = torch.ones_like(graphed_out)
        deallocate_output_tensor(graphed_out, deallocate_pipeline_outputs=True)
        custom_backward(graphed_out, graphed_grad)

        for layer in block.layers:
            for runner in layer.cudagraph_manager.cudagraph_runners:
                if hasattr(runner, "fwd_graph"):
                    del runner.fwd_graph
                if hasattr(runner, "bwd_graph"):
                    del runner.bwd_graph
        torch.cuda.synchronize()
