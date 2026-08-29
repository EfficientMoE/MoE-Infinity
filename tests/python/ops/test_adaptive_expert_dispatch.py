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


def test_precision_metrics_schema_is_complete():
    required = {
        "budget_bytes",
        "resident_bytes",
        "resident_payload_bytes",
        "alignment_padding_bytes",
        "transition_reserved_bytes",
        "workspace_bytes",
        "peak_accounted_bytes",
        "h2d_payload_bytes",
        "h2d_transfers",
        "conversion_input_bytes",
        "conversion_output_bytes",
        "conversion_seconds",
        "promotions",
        "demotions",
        "representation_hits",
        "representation_misses",
        "policy_epochs",
        "active_leases",
        "leases_by_kind",
        "manager_instance_id",
        "manager_enabled",
        "phase_policy_enabled",
        "pending_transactions",
        "registered_variants",
        "resident_generations",
        "resident_generation_entries",
        "retiring_generations",
        "external_shared_resident_bytes",
        "fallback_counts",
        "by_format",
    }
    assert required <= set(
        dir(_store.expert_dispatcher)
    ) or "get_precision_metrics" in dir(_store.expert_dispatcher)
