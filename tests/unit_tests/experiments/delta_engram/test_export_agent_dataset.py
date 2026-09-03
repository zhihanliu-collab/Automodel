# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json

from experiments.delta_engram.export_agent_dataset import _export_agent_dataset


def test_export_agent_dataset_matches_cache_and_masks_stale_turn(tmp_path):
    source = tmp_path / "agents.jsonl"
    rows = []
    for bill_id in (10, 20):
        rows.append(
            {
                "messages": [
                    {"role": "user", "content": f"TASK bill(id={bill_id})"},
                    {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "grounded",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "odoo_edit_bill_header",
                                    "arguments": {"bill_id": bill_id, "accounting_date": "2026-01-01"},
                                }
                            }
                        ],
                    },
                ],
                "tools": [{"type": "function", "function": {"name": "odoo_edit_bill_header"}}],
            }
        )
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    # Derive the expected split from the production grouping helper first, then
    # make the cache fixture assert that the exporter preserves that contract.
    from experiments.delta_engram.odoo_corpus import _agent_tasks

    tasks = _agent_tasks(source, validation_fraction=0.5, seed=7)
    cache_manifest = tmp_path / "cache.json"
    cache_manifest.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "sample_id": task["sample_id"],
                        "source": "agent_trajectories",
                        "split": task["split"],
                        "group": task["group"],
                        "rendered_tokens": 101 + index,
                        "length": 100 + index,
                        "supervised_tokens": 11 + index,
                    }
                    for index, task in enumerate(tasks)
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "export.jsonl"
    manifest = tmp_path / "export.manifest.json"

    _export_agent_dataset(
        agent_jsonl=source,
        cache_manifest=cache_manifest,
        output_jsonl=output,
        output_manifest=manifest,
        validation_fraction=0.5,
        seed=7,
    )

    exported = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["split"] for row in exported] == [task["split"] for task in tasks]
    assert [row["rendered_tokens"] for row in exported] == [101, 102]
    assert all(row["messages"][1]["step_loss_mask"] == 0 for row in exported)
    assert all(row["messages"][1]["reasoning_content"] == "grounded" for row in exported)
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    assert metadata["num_samples"] == 2
    assert metadata["masked_outdated_accounting_turns"] == 2
    assert metadata["output_sha256"]
