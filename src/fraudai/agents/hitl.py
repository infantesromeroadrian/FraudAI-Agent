"""Human approval of individual offensive tool calls (R2 / SEC-006).

The model proposes tool calls; a human approves *each concrete call*
(tool, arguments, target) before anything runs.  This module holds the pure
domain pieces so the graph nodes stay thin:

- which calls need approval (fail-closed for Mike: anything not read-only);
- the frozen proposal recorded in graph state *before* the graph pauses, so a
  node replay after ``interrupt()`` never re-plans or changes arguments;
- the interrupt payload shown to the client and the strict decision parser.

Limitation: approval is bound to one ``action_id`` in the checkpoint of one
session.  Checkpoints are not an idempotency ledger: a crash between running a
tool and persisting its result can repeat the call after recovery.  Exactly-once
would need a destination that deduplicates ``action_id``.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any, Final, Literal, TypedDict

APPROVAL_KIND: Final = "offensive_tool_approval"

# Tools that are offensive wherever they appear.
OFFENSIVE_TOOLS: Final[frozenset[str]] = frozenset(
    {"adversarial_evasion", "prompt_injection_suite"}
)
# Agents whose tool calls are gated by default; a tool new to their list is
# treated as offensive until someone lists it as read-only below.
APPROVAL_GATED_AGENTS: Final[frozenset[str]] = frozenset({"mike"})
READ_ONLY_TOOLS: Final[frozenset[str]] = frozenset({"search_boe"})

# The reviewer must see exactly what runs, so oversize arguments are refused
# rather than truncated for display.
MAX_ARGUMENTS_BYTES: Final = 16 * 1024
MAX_CALLS_PER_PLAN: Final = 5
MAX_DECISION_ATTEMPTS: Final = 3

_TARGET_KEYS: Final = ("target_endpoint", "target", "endpoint", "url")

ActionStatus = Literal["awaiting_approval", "approved", "rejected", "executed", "failed"]

OFFENSIVE_WARNING: Final = (
    "This action will run an offensive security tool against the target shown. "
    "Approve only if you are authorised to test it."
)


class ActionProposal(TypedDict):
    """One planned tool call, frozen when planned and updated only by status."""

    action_id: str
    tool_call_id: str
    agent: str
    tool: str
    arguments: dict[str, Any]
    target: str | None
    digest: str
    requires_approval: bool
    status: ActionStatus
    result: dict[str, Any] | None
    note: str | None


def requires_approval(agent: str, tool: str) -> bool:
    """Return whether this call must wait for a human decision."""
    if tool in OFFENSIVE_TOOLS:
        return True
    return agent in APPROVAL_GATED_AGENTS and tool not in READ_ONLY_TOOLS


def target_of(arguments: dict[str, Any]) -> str | None:
    """Best-effort target shown to the reviewer; the arguments remain authoritative."""
    for key in _TARGET_KEYS:
        value = arguments.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        msg = "arguments are not JSON-serialisable"
        raise ValueError(msg) from exc


def action_digest(session_id: str, agent: str, tool: str, arguments: dict[str, Any]) -> str:
    """Fingerprint of what is approved; re-checked right before execution.

    Raises:
        ValueError: arguments are not JSON-serialisable or exceed the review limit.
    """
    arguments_json = _canonical_json(arguments)
    if len(arguments_json.encode()) > MAX_ARGUMENTS_BYTES:
        msg = "arguments exceed the reviewable size"
        raise ValueError(msg)
    envelope = _canonical_json(
        {"session": session_id, "agent": agent, "tool": tool, "arguments": arguments_json}
    )
    return hashlib.sha256(envelope.encode()).hexdigest()


def _call_field(call: Any, field: str, default: Any) -> Any:
    if isinstance(call, dict):
        return call.get(field, default)
    return getattr(call, field, default)


def build_proposals(
    agent: str,
    session_id: str,
    tool_calls: list[Any],
    known_tools: set[str],
) -> list[ActionProposal]:
    """Freeze the model's tool calls into proposals with server-issued ids.

    Calls that cannot run (unknown tool, unreviewable arguments, over the
    per-plan cap) are given a terminal status here, so they never reach an
    approval prompt or an executor.
    """
    proposals: list[ActionProposal] = []
    seen_call_ids: set[str] = set()
    for index, call in enumerate(tool_calls):
        tool = str(_call_field(call, "name", ""))
        raw_args = _call_field(call, "args", {})
        arguments = raw_args if isinstance(raw_args, dict) else {}
        call_id = str(_call_field(call, "id", "") or f"call-{index}")
        if call_id in seen_call_ids:
            call_id = f"{call_id}-{index}"
        seen_call_ids.add(call_id)

        gated = requires_approval(agent, tool)
        status: ActionStatus = "awaiting_approval" if gated else "approved"
        note: str | None = None
        digest = ""
        if index >= MAX_CALLS_PER_PLAN:
            status, note = "rejected", "Too many tool calls in one plan; not executed."
        elif tool not in known_tools:
            status, note = "failed", f"Tool '{tool}' is not available to this agent."
        elif not isinstance(raw_args, dict):
            status, note = "rejected", "Tool arguments were not an object; not executed."
        else:
            try:
                digest = action_digest(session_id, agent, tool, arguments)
            except ValueError:
                status, note = "rejected", "Arguments cannot be reviewed; not executed."

        proposals.append(
            ActionProposal(
                action_id=str(uuid.uuid4()),
                tool_call_id=call_id,
                agent=agent,
                tool=tool,
                arguments=arguments,
                target=target_of(arguments),
                digest=digest,
                requires_approval=gated,
                status=status,
                result=None,
                note=note,
            )
        )
    return proposals


def approval_payload(action: ActionProposal) -> dict[str, Any]:
    """JSON-serialisable interrupt value: exactly what the human is approving."""
    return {
        "kind": APPROVAL_KIND,
        "action_id": action["action_id"],
        "agent": action["agent"],
        "tool": action["tool"],
        "arguments": action["arguments"],
        "target": action["target"],
        "warning": OFFENSIVE_WARNING,
    }


def parse_decision(raw: Any, action_id: str) -> bool | None:
    """Return the decision, or ``None`` when the answer is not valid for this action.

    Only ``{"action_id": <this id>, "approved": <bool>}`` counts; truthy
    strings or numbers never approve.
    """
    if not isinstance(raw, dict) or set(raw) != {"action_id", "approved"}:
        return None
    if raw["action_id"] != action_id:
        return None
    approved = raw["approved"]
    if type(approved) is not bool:
        return None
    return approved
