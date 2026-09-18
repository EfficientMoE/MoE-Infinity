from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_disk_to_gpu_expert_fetch_uses_group_read() -> None:
    source = (ROOT / "core/model/model_topology.cpp").read_text()
    branch = source.split("if (from_disk && target_device.is_cuda()) {", 1)[
        1
    ].split("} else if (from_disk) {", 1)[0]

    assert "SetModuleMemoryFromDisk(tensor_ids" in branch
    assert "ReadTensor(" not in branch
