# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Multi-iteration pipeline regressions for local training CUDA graphs."""

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.tensor_parallel.random import (
    HAVE_TE,
    initialize_rng_tracker,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer.cuda_graphs import CudaGraphManager, _CudagraphGlobalRecord
from megatron.core.transformer.enums import CudaGraphModule
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version
from tests.unit_tests.test_utilities import Utils


@pytest.mark.skipif(
    not (HAVE_TE and is_te_min_version("1.5.0")),
    reason="use_te_rng_tracker requires TransformerEngine version >= 1.5",
)
class TestLocalCudagraphPipelineIterations:
    """Exercise record/capture and repeated replay through real PP communication."""

    def setup_method(self, method):
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=2,
        )
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()
        _CudagraphGlobalRecord.cudagraph_created = False
        _CudagraphGlobalRecord.cudagraph_record = []
        _CudagraphGlobalRecord.cudagraph_inference_record = []
        _CudagraphGlobalRecord._disable_saved_tensors_observer()
        CudaGraphManager.global_mempool = None

    @staticmethod
    def _make_config(cuda_graph_module):
        return TransformerConfig(
            num_layers=4,
            hidden_size=64,
            num_attention_heads=4,
            bf16=True,
            params_dtype=torch.bfloat16,
            pipeline_dtype=torch.bfloat16,
            pipeline_model_parallel_size=2,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            gradient_accumulation_fusion=True,
            deallocate_pipeline_outputs=True,
            cuda_graph_impl="local",
            cuda_graph_modules=[cuda_graph_module],
            cuda_graph_warmup_steps=1,
            use_cpu_initialization=True,
        )

    @pytest.mark.parametrize(
        "cuda_graph_module",
        [CudaGraphModule.mlp, CudaGraphModule.attn],
    )
    def test_pipeline_schedule_replays_across_iterations(self, cuda_graph_module):
        config = self._make_config(cuda_graph_module)
        block = TransformerBlock(
            config,
            get_gpt_layer_with_transformer_engine_spec(),
            pre_process=parallel_state.is_pipeline_first_stage(),
            post_process=parallel_state.is_pipeline_last_stage(),
        ).cuda()
        block.model_type = ModelType.encoder_or_decoder
        block.train()

        # Local CUDA-graph backward capture expects DDP-style persistent
        # gradient-accumulation buffers.
        for param in block.parameters():
            param.main_grad = torch.zeros_like(param)

        sequence_length = 32
        micro_batch_size = 1
        num_microbatches = 4
        hidden_states = torch.randn(
            (sequence_length, micro_batch_size, config.hidden_size),
            dtype=torch.bfloat16,
            device="cuda",
            requires_grad=True,
        )
        attention_mask = torch.ones(
            (1, 1, sequence_length, sequence_length),
            dtype=bool,
            device="cuda",
        )

        def forward_step_func(data_iterator, model):
            next(data_iterator)
            output = model(hidden_states=hidden_states, attention_mask=attention_mask)

            def loss_func(output_tensor):
                loss = output_tensor.float().square().mean()
                return loss, {"loss": loss.detach()}

            return output, loss_func

        forward_backward_func = get_forward_backward_func()
        observed_losses = []

        # Iteration 1 records and captures. Iterations 2-4 replay the same
        # local graphs, crossing the iteration-3 boundary where the full model
        # previously stalled at its first PP P2P operation.
        for iteration in range(4):
            losses = forward_backward_func(
                forward_step_func=forward_step_func,
                data_iterator=iter([None] * num_microbatches),
                model=[block],
                num_microbatches=num_microbatches,
                seq_length=sequence_length,
                micro_batch_size=micro_batch_size,
                forward_only=False,
            )
            torch.cuda.synchronize()
            if losses:
                observed_losses.append(float(losses[-1]["loss"]))

            assert _CudagraphGlobalRecord.cudagraph_created
            for layer in block.layers:
                runners = layer.cudagraph_manager.cudagraph_runners
                assert len(runners) == num_microbatches
                assert all(runner.fwd_graph is not None for runner in runners)
                assert all(runner.bwd_graph is not None for runner in runners)

        if observed_losses:
            assert len(observed_losses) == 4
            assert all(torch.isfinite(torch.tensor(loss)) for loss in observed_losses)

        for layer in block.layers:
            for runner in layer.cudagraph_manager.cudagraph_runners:
                if hasattr(runner, "fwd_graph"):
                    del runner.fwd_graph
                if hasattr(runner, "bwd_graph"):
                    del runner.bwd_graph
        torch.cuda.synchronize()
