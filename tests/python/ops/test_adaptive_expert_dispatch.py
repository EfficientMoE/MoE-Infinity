# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

import pytest

from moe_infinity import _store


@pytest.mark.gpu
def test_dispatcher_publishes_adaptive_control_surface():
    required = {
        "register_expert_variant",
        "set_precision_targets",
        "set_adaptive_hbm_budget_bytes",
        "get_precision_metrics",
        "get_residency_manager_id",
        "configure_residency_manager",
    }
    assert required <= set(dir(_store.expert_dispatcher))


@pytest.mark.gpu
def test_failed_transition_binding_is_testing_only():
    assert hasattr(
        _store.expert_dispatcher, "inject_transition_failure_once_for_test"
    )
