# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Graded-PASS host runs in the evaluation world -> Flash-Next training rows, from the proxy wire dump.

Why: every Delta variant trained on the 960 ccsdk trajectories (another world: no vendor NPC,
``mcp__neogym__*`` tool names, internal-only chatter, month-end accounting dates) lost the same
eval20 tasks by never answering the vendor after ``post_bill``. The evaluation world's own runs
contain behaviourally correct trajectories: the ones the grader passed. Only tasks outside eval20
are eligible (``--tasks`` = lists/syn80.txt; ``--forbid`` = lists/eval20.txt is a hard check).

Source: the claude-sdk thinking proxy's wire dump (``NEOGYM_PROXY_WIRE_DUMP=1`` +
``NEOGYM_PROXY_EVENTS_PATH=<dir>/proxy_events.jsonl`` -> ``<dir>/proxy_wire.jsonl``). Each record
is ``{"kind": anthropic_request|request|response, "trace": baseline::<run_id>::inst<N>::attempt<k>,
"body": ...}``; ``request`` bodies are the exact OpenAI-wire messages the served model saw
(system prompt, user turn with the seeded memory index, assistant turns with ``reasoning_content``
and ``tool_calls``, tool results) plus the 28 tool schemas. The LAST request of a trace carries the
whole conversation prefix; appending the last response's assistant message gives the complete
trajectory in the model-facing format (no session-log reconstruction, no guessed system prompt).

Output rows follow jian's ccsdk export so ``odoo_corpus.py --eval-world-agent-jsonl`` ingests them:
``{"messages": [...], "tools": [...], "meta": {...}}`` with tool-call arguments as dicts and user
content flattened to text.

    python -m experiments.delta_engram.build_eval_world_trajectories \
        --wire /mnt/data/zhihan/delta-engram/corpus/raw/wire/syn80a/proxy_wire.jsonl \
        --wire /mnt/data/zhihan/delta-engram/corpus/raw/wire/syn80b/proxy_wire.jsonl \
        --runs-root /mnt/data/zhihan/reviewer_reduce100/runs \
        --tasks /mnt/data/zhihan/reviewer_reduce100/lists/syn80.txt \
        --forbid /mnt/data/zhihan/reviewer_reduce100/lists/eval20.txt \
        --out /mnt/data/zhihan/delta-engram/corpus/raw/eval_world_passing_syn80_v1.jsonl
"""

from __future__ import annotations

import argparse
import ast
import collections
import json
import os
import re
from typing import Any

_TRACE_RE = re.compile(r"^baseline::(?P<run>.+)::inst(?P<inst>\d+)::attempt(?P<attempt>\d+)$")
_BILL_RE = re.compile(r"vendor bill \(id=(\d+)\)")


def _read_list(path: str) -> set[str]:
    return {t for t in open(path).read().replace("\n", "").split(",") if t}


def _graded(runs_root: str, run_id: str, inst: int) -> tuple[bool, str, str] | None:
    """(passed, task_id, task prompt) for one task dir, or None when it is not graded yet."""
    task_dir = os.path.join(runs_root, run_id, f"task-{inst:03d}")
    result_path = os.path.join(task_dir, "result.json")
    if not os.path.exists(result_path):
        return None
    record = json.load(open(result_path))
    result = record["result"]
    if isinstance(result, str):
        result = ast.literal_eval(result)
    by_task = list(result["by_task"].values())[0]
    prompt = ""
    session = os.path.join(task_dir, "conversations", "session-000.jsonl")
    if os.path.exists(session):
        for line in open(session, encoding="utf-8"):
            event = json.loads(line)
            if event.get("role") == "user":
                for block in event.get("content") or []:
                    if isinstance(block, dict) and block.get("type") == "text":
                        prompt = block.get("text", "")
                        break
                break
    return bool(by_task["success"]), record["task_id"], prompt


def _flatten(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, dict) and block.get("type") in ("image", "image_url"):
                parts.append("[image omitted]")
            elif isinstance(block, dict):
                parts.append(json.dumps(block, ensure_ascii=False))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return json.dumps(content, ensure_ascii=False)


def _clean_message(message: dict[str, Any]) -> dict[str, Any]:
    """Wire message -> ccsdk-style row message (text content, dict tool arguments)."""
    out: dict[str, Any] = {"role": message["role"], "content": _flatten(message.get("content"))}
    if message["role"] == "assistant":
        if message.get("reasoning_content"):
            out["reasoning_content"] = str(message["reasoning_content"])
        calls = []
        for call in message.get("tool_calls") or []:
            function = dict(call.get("function") or {})
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"_raw": arguments}
            if not isinstance(arguments, dict):
                arguments = {"input": arguments}
            calls.append(
                {
                    "id": call.get("id"),
                    "type": "function",
                    "function": {"name": function.get("name"), "arguments": arguments},
                }
            )
        if calls:
            out["tool_calls"] = calls
    elif message["role"] == "tool":
        out["tool_call_id"] = message.get("tool_call_id")
    return out


def _trajectory(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | None:
    """Last tool-bearing request + the response that followed it, or None when incomplete."""
    last_request_index = None
    for index, record in enumerate(records):
        if record["kind"] == "request" and record["body"].get("tools"):
            last_request_index = index
    if last_request_index is None:
        return None
    response = next((r for r in records[last_request_index + 1 :] if r["kind"] == "response"), None)
    if response is None:
        return None
    body = records[last_request_index]["body"]
    messages = [dict(m) for m in body["messages"]]
    choice = (response["body"].get("choices") or [{}])[0].get("message")
    if choice:
        messages.append({"role": "assistant", **{k: v for k, v in choice.items() if k != "role"}})
    return messages, body["tools"]


_RUN_DATE_RE = re.compile(r"_(\d{2})(\d{2})-\d{6}-")


def _session_messages(path: str) -> list[dict[str, Any]] | None:
    """claude-sdk session-000.jsonl -> wire-shaped messages (no system / first-user wrapper yet).

    Assistant records arrive one block per line (thinking, text, tool_use); consecutive assistant
    records form one turn. A user record holds tool_result blocks (-> tool messages) and, rarely,
    text (the task prompt, a compaction summary). Returns None when any assistant turn has an empty
    thinking block: the trajectory would teach "no reasoning" (gemini logs carry empty thinking).
    """
    messages: list[dict[str, Any]] = []
    pending: dict[str, Any] | None = None
    empty_thinking = 0

    def flush() -> None:
        nonlocal pending
        if pending is None:
            return
        message: dict[str, Any] = {"role": "assistant", "content": "".join(pending["text"])}
        reasoning = "".join(pending["thinking"])
        if reasoning.strip():
            message["reasoning_content"] = reasoning
        if pending["tool_calls"]:
            message["tool_calls"] = pending["tool_calls"]
        messages.append(message)
        pending = None

    for line in open(path, encoding="utf-8"):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        role = event.get("role")
        if role == "assistant":
            if pending is None:
                pending = {"text": [], "thinking": [], "tool_calls": []}
            for block in event.get("content") or []:
                if not isinstance(block, dict):
                    continue
                kind = block.get("type")
                if kind == "thinking":
                    text = block.get("thinking") or ""
                    if not text.strip():
                        empty_thinking += 1
                    pending["thinking"].append(text)
                elif kind == "text":
                    pending["text"].append(block.get("text", ""))
                elif kind == "tool_use":
                    arguments = block.get("input")
                    if not isinstance(arguments, dict):
                        arguments = {"input": arguments}
                    pending["tool_calls"].append(
                        {
                            "id": block.get("id"),
                            "type": "function",
                            "function": {"name": block.get("name"), "arguments": arguments},
                        }
                    )
        elif role == "user":
            flush()
            blocks = event.get("content")
            if isinstance(blocks, str):
                messages.append({"role": "user", "content": blocks})
                continue
            texts = []
            for block in blocks or []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_result":
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": block.get("tool_use_id"),
                            "content": _flatten(block.get("content")),
                        }
                    )
                elif block.get("type") == "text":
                    texts.append(block.get("text", ""))
            if texts:
                messages.append({"role": "user", "content": "\n".join(texts)})
    flush()
    if empty_thinking or not any(m["role"] == "assistant" for m in messages):
        return None
    return messages


def _session_rows(args: argparse.Namespace, eligible: set[str], forbidden: set[str]) -> list[dict[str, Any]]:
    """Graded-PASS session logs of other arms/models -> rows shaped like the wire ones.

    The system prompt, the first-user wrapper (Claude Code's system-reminder with the seeded MEMORY.md)
    and the tool schemas come from a wire-dump template (``--wire-template``, written by
    wire_template.py from a seeded no-reviewer base run); ``<RUN>`` is replaced by this task's own dir
    and the machine date by the run's date.
    """
    import glob as _glob

    template = json.load(open(args.wire_template))
    rows: list[dict[str, Any]] = []
    per_task: collections.Counter = collections.Counter()
    skipped: collections.Counter = collections.Counter()
    for pattern in args.session_runs:
        for run_dir in sorted(_glob.glob(pattern)):
            name = os.path.basename(run_dir)
            match = re.search(r"claude-sdk_(seeded_)?(.+?)_(reduce100|dev47|dev_)", name)
            model = match.group(2) if match else "?"
            if args.session_model_substr and not any(s in model for s in args.session_model_substr):
                skipped["model"] += 1
                continue
            if "Delta" in name or "syn80" in name:
                skipped["excluded_family"] += 1
                continue
            date_match = _RUN_DATE_RE.search(name)
            run_date = f"2026-{date_match.group(1)}-{date_match.group(2)}" if date_match else "2026-09-04"
            for result_path in sorted(_glob.glob(os.path.join(run_dir, "task-*", "result.json"))):
                task_dir = os.path.dirname(result_path)
                graded = _graded(
                    args.runs_root,
                    os.path.relpath(run_dir, args.runs_root) if run_dir.startswith(args.runs_root) else run_dir,
                    int(os.path.basename(task_dir).split("-")[1]),
                )
                if graded is None:
                    skipped["ungraded"] += 1
                    continue
                passed, task_id, prompt = graded
                if task_id in forbidden:
                    raise SystemExit(f"eval firewall: {task_dir} is eval task {task_id}; refusing to build")
                if task_id not in eligible or not passed:
                    skipped["ineligible_or_failed"] += 1
                    continue
                if per_task[task_id] >= args.max_per_task:
                    skipped["cap"] += 1
                    continue
                session = os.path.join(task_dir, "conversations", "session-000.jsonl")
                if not os.path.exists(session):
                    skipped["no_session"] += 1
                    continue
                body = _session_messages(session)
                if body is None:
                    skipped["empty_thinking_or_no_assistant"] += 1
                    continue
                if body[0]["role"] != "user" or not body[0]["content"].startswith("TASK:"):
                    skipped["no_task_prompt_first"] += 1
                    continue
                system_prompt = template["system"].replace("<RUN>", task_dir)
                prefix = re.sub(
                    r"Today's date is 20\d\d-\d\d-\d\d", f"Today's date is {run_date}", template["user1_prefix"]
                ).replace("<RUN>", task_dir)
                body[0] = {"role": "user", "content": prefix + body[0]["content"]}
                cleaned = [{"role": "system", "content": system_prompt}] + [_clean_message(m) for m in body]
                if cleaned[-1]["role"] != "assistant":
                    skipped["no_final_assistant"] += 1
                    continue
                names = [
                    str(tc["function"]["name"]).split("__")[-1]
                    for m in cleaned
                    if m["role"] == "assistant"
                    for tc in m.get("tool_calls") or []
                ]
                last_post = max((i for i, n in enumerate(names) if n == "post_bill"), default=None)
                rows.append(
                    {
                        "messages": cleaned,
                        "tools": template["tools"],
                        "meta": {
                            "task_id": task_id,
                            "source": "session",
                            "model": model,
                            "run": name,
                            "seeded": "_seeded_" in name,
                            "reviewer": re.search(r"_base(_|$)", name) is None,
                            "n_messages": len(cleaned),
                            "tool_calls": len(names),
                            "chatter_messages": names.count("send_chatter_message"),
                            "chatter_after_post_bill": names[last_post + 1 :].count("send_chatter_message")
                            if last_post is not None
                            else 0,
                            "finished": "finish_task" in names,
                        },
                    }
                )
                per_task[task_id] += 1
    print(f"session rows: {len(rows)} over {len(per_task)} tasks; skipped {dict(skipped)}")
    print("session per model:", dict(collections.Counter(r["meta"]["model"] for r in rows)))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wire", action="append", default=[], help="proxy_wire.jsonl files")
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--tasks", required=True, help="eligible task ids (comma list file, e.g. lists/syn80.txt)")
    parser.add_argument("--forbid", action="append", default=[], help="task-id lists that must never appear (eval20)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-per-task", type=int, default=8)
    parser.add_argument(
        "--session-runs",
        action="append",
        default=[],
        help="run-dir globs whose graded-PASS session logs are added (other models/arms)",
    )
    parser.add_argument(
        "--wire-template",
        default=None,
        help="system/first-user/tools template json (wire_template.py) for --session-runs",
    )
    parser.add_argument(
        "--session-model-substr",
        action="append",
        default=[],
        help="only session runs whose served model contains one of these",
    )
    args = parser.parse_args()

    eligible = _read_list(args.tasks)
    forbidden = set().union(*(_read_list(p) for p in args.forbid)) if args.forbid else set()
    if eligible & forbidden:
        raise SystemExit(f"eligible list overlaps forbidden list: {sorted(eligible & forbidden)}")

    by_trace: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    bad_lines = 0
    for path in args.wire:
        for line in open(path, encoding="utf-8"):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                bad_lines += 1  # concurrent appenders can interleave a line; drop it
                continue
            if _TRACE_RE.match(str(record.get("trace", ""))):
                by_trace[record["trace"]].append(record)

    rows: list[dict[str, Any]] = []
    per_task: collections.Counter = collections.Counter()
    skipped: collections.Counter = collections.Counter()
    for trace, records in by_trace.items():
        match = _TRACE_RE.match(trace)
        run_id, inst = match.group("run"), int(match.group("inst"))
        graded = _graded(args.runs_root, run_id, inst)
        if graded is None:
            skipped["ungraded"] += 1
            continue
        passed, task_id, prompt = graded
        if task_id in forbidden:
            raise SystemExit(f"eval firewall: {trace} is eval task {task_id}; refusing to build")
        if task_id not in eligible:
            skipped["ineligible"] += 1
            continue
        if not passed:
            skipped["failed"] += 1
            continue
        if per_task[task_id] >= args.max_per_task:
            skipped["cap"] += 1
            continue
        built = _trajectory(records)
        if built is None:
            skipped["incomplete"] += 1
            continue
        messages, tools = built
        first_user = next((m for m in messages if m["role"] == "user"), None)
        wire_bill = _BILL_RE.search(_flatten(first_user["content"]) if first_user else "")
        session_bill = _BILL_RE.search(prompt)
        if wire_bill and session_bill and wire_bill.group(1) != session_bill.group(1):
            raise SystemExit(
                f"{trace}: wire bill {wire_bill.group(1)} != session bill {session_bill.group(1)} "
                "(inst->task-dir mapping broken)"
            )
        cleaned = [_clean_message(m) for m in messages]
        if cleaned[-1]["role"] != "assistant":
            skipped["no_final_assistant"] += 1
            continue
        names = [
            str(tc["function"]["name"]).split("__")[-1]
            for m in cleaned
            if m["role"] == "assistant"
            for tc in m.get("tool_calls") or []
        ]
        last_post = max((i for i, n in enumerate(names) if n == "post_bill"), default=None)
        rows.append(
            {
                "messages": cleaned,
                "tools": tools,
                "meta": {
                    "task_id": task_id,
                    "trace": trace,
                    "run": run_id,
                    "n_messages": len(cleaned),
                    "tool_calls": len(names),
                    "chatter_messages": names.count("send_chatter_message"),
                    "chatter_after_post_bill": names[last_post + 1 :].count("send_chatter_message")
                    if last_post is not None
                    else 0,
                    "finished": names[-1] == "finish_task" if names else False,
                },
            }
        )
        per_task[task_id] += 1

    if args.session_runs:
        if not args.wire_template:
            raise SystemExit("--session-runs needs --wire-template")
        rows.extend(_session_rows(args, eligible, forbidden))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} trajectories over {len(per_task)} tasks -> {args.out}  (bad wire lines: {bad_lines})")
    print("skipped:", dict(skipped))
    if rows:
        after = sum(1 for r in rows if r["meta"]["chatter_after_post_bill"] > 0)
        finished = sum(1 for r in rows if r["meta"]["finished"])
        mean_chat = sum(r["meta"]["chatter_messages"] for r in rows) / len(rows)
        print(
            f"chatter after post_bill in {after}/{len(rows)}; finish_task last in {finished}/{len(rows)}; "
            f"mean chatter msgs {mean_chat:.2f}"
        )
        print("per task:", dict(sorted(per_task.items())))


if __name__ == "__main__":
    main()
