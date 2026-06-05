"""
LangGraph ReAct reasoning agent.

The agent runs a Thought → Action → Observation loop until it produces
a final answer or hits MAX_ITERATIONS.

Graph nodes:
  think     — LLM decides next action or emits Final Answer
  act       — Execute the chosen tool
  respond   — Format and return the final answer

State keys:
  task          str   original user request
  history       list  [(thought, action, observation), ...]
  final_answer  str   set when the agent is done
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TypedDict, List, Tuple, Optional, Annotated
import operator

# Matches serial-number-like tokens: 8+ uppercase alphanumeric chars
_SN_RE = re.compile(r'\b([A-Z0-9]{8,})\b')

from langgraph.graph import StateGraph, END

from src.config import MAX_ITERATIONS
from src.tools import ALL_TOOLS

_PROMPT_DIR = Path(__file__).resolve().parent.parent / "system_prompt"


def _load_system_prompt() -> str:
    """
    Assemble the system prompt from system_prompt/*.md files in order:
      main.md → sfis_workflow.md → response_format.md
    Falls back to a minimal inline prompt if files are missing.
    """
    parts = []
    for name in ("main.md", "sfis_workflow.md", "response_format.md"):
        path = _PROMPT_DIR / name
        if path.exists():
            parts.append(path.read_text(encoding="utf-8").rstrip())
        else:
            logger.warning("system_prompt/%s not found — skipping", name)
    return "\n\n".join(parts)

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# State
# ------------------------------------------------------------------

class AgentState(TypedDict):
    task: str
    history: Annotated[List[Tuple[str, str, str]], operator.add]
    final_answer: Optional[str]
    iterations: int
    chat_history: Optional[list]
    _pending_action: Optional[str]
    _pending_input: Optional[str]


# ------------------------------------------------------------------
# Prompt helpers
# ------------------------------------------------------------------

TOOL_DESCRIPTIONS = "\n".join(
    f"- {t.name}: {t.description}" for t in ALL_TOOLS
)

# Assembled from system_prompt/*.md at startup — edit those files to tune behaviour.
SYSTEM_PROMPT = _load_system_prompt().format(TOOL_DESCRIPTIONS=TOOL_DESCRIPTIONS)


def _build_prompt(state: AgentState, chat_history: list | None = None) -> str:
    lines = [SYSTEM_PROMPT]

    # Auto memory recall on the first step — inject relevant past context
    if not state["history"]:
        try:
            from src.memory import MemoryStore
            memories = MemoryStore().search_text(state["task"], k=3)
            if memories and "No relevant memories" not in memories:
                lines.append(f"\n## Relevant Memories\n{memories}\n")
        except Exception:
            pass

    # Inject recent conversation so the model can answer follow-ups
    if chat_history:
        lines.append("\n## Conversation History")
        for user_msg, assistant_msg in chat_history[-6:]:
            lines.append(f"User: {user_msg}")
            short = assistant_msg[:600] + "…" if len(assistant_msg) > 600 else assistant_msg
            lines.append(f"Assistant: {short}")
        lines.append("")

    lines.append(f"## Current Task\n{state['task']}\n")
    for thought, action, observation in state["history"]:
        lines.append(f"Thought: {thought}")
        if action:
            lines.append(f"Action: {action}")
        if observation:
            lines.append(f"Observation: {observation}")
    lines.append("Thought:")
    return "\n".join(lines)


def _parse_llm_output(raw: str) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    """Parse LLM output into (thought, action, action_input, final_answer)."""
    # Strip Qwen3 <think>...</think> blocks (safety net — also done in inference.py).
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw = raw.strip()
    if raw.startswith("Thought:"):
        raw = raw[len("Thought:"):].strip()

    # Safety net: if the model hallucinated an Observation block despite stop
    # sequences, cut everything from there.  This prevents fake tool results
    # from being read as the Final Answer.
    obs_idx = raw.find("\nObservation:")
    if obs_idx != -1:
        raw = raw[:obs_idx].strip()

    final_match = re.search(r"Final Answer:\s*(.*)", raw, re.DOTALL)
    if final_match:
        thought_part = raw[: final_match.start()].strip()
        answer = final_match.group(1).strip()
        # Cut off any trailing reasoning the model appended after the answer
        for cutoff in ("\nThought:", "\nAction:", "\nWait,", "\nActually,", "\nHowever,", "\nBut "):
            idx = answer.find(cutoff)
            if idx != -1:
                answer = answer[:idx].strip()
        return thought_part, None, None, answer

    action_match = re.search(r"Action:\s*(\w+)", raw)
    input_match = re.search(r"Action Input:\s*(.*?)(?=\nThought:|\nAction:|\Z)", raw, re.DOTALL)

    thought = raw.split("\nAction:")[0].strip() if action_match else raw
    action = action_match.group(1).strip() if action_match else None
    action_input = input_match.group(1).strip() if input_match else ""

    return thought, action, action_input, None


# ------------------------------------------------------------------
# Graph nodes
# ------------------------------------------------------------------

def think_node(state: AgentState) -> AgentState:
    """Ask the LLM what to do next."""
    from src.inference import get_llm

    i = state["iterations"]
    if i >= MAX_ITERATIONS:
        return {**state, "final_answer": "Reached maximum iterations without a final answer."}

    print(f"\n[AGENT] ── iter {i} ──────────────────────────────────")
    print(f"[AGENT iter={i}] Calling LLM ...")
    prompt = _build_prompt(state, state.get("chat_history"))
    llm = get_llm()
    raw = llm.invoke(prompt, stop=["\nObservation:"])
    print(f"[AGENT iter={i}] Raw LLM output:\n{raw}\n---")

    thought, action, action_input, final_answer = _parse_llm_output(raw)
    print(f"[AGENT iter={i}] Parsed → thought={thought[:80]!r}  action={action!r}  input={action_input!r}  final_answer={'YES' if final_answer else 'NO'}")

    if final_answer:
        print(f"[AGENT iter={i}] Final Answer produced — done.")
        return {**state, "final_answer": final_answer}

    print(f"[AGENT iter={i}] Tool call → {action!r}  input={action_input!r}")
    new_entry = (thought, f"{action}|{action_input}", "")
    return {
        **state,
        "history": state["history"] + [new_entry],
        "iterations": state["iterations"] + 1,
        "_pending_action": action,
        "_pending_input": action_input,
    }


def act_node(state: AgentState) -> AgentState:
    """Execute the tool chosen in think_node."""
    action = state.get("_pending_action")
    action_input = state.get("_pending_input", "")

    if not action:
        return state

    tool_map = {t.name: t for t in ALL_TOOLS}
    tool = tool_map.get(action)

    # Tools that legitimately accept empty input
    _EMPTY_INPUT_OK = {"list_dir", "memory_recall"}

    if tool is None:
        observation = f"Unknown tool '{action}'. Available: {list(tool_map.keys())}"
    elif not action_input and action not in _EMPTY_INPUT_OK:
        observation = f"Error: empty Action Input for '{action}'. Provide a non-empty input or output a Final Answer."
    else:
        print(f"[AGENT] Executing tool {action!r} with input={action_input!r}")
        try:
            observation = tool.run(action_input)
        except Exception as e:
            observation = f"Tool error: {e}"

    print(f"[AGENT] Tool result ({len(observation)} chars): {observation[:200]}")

    # Patch the last history entry with the observation
    history = list(state["history"])
    if history:
        thought, act_str, _ = history[-1]
        history[-1] = (thought, act_str, observation)

    return {**state, "history": history}


def respond_node(state: AgentState) -> AgentState:
    """Terminal node — final_answer is already set."""
    return state


# ------------------------------------------------------------------
# Routing
# ------------------------------------------------------------------

def _route_after_think(state: AgentState) -> str:
    if state.get("final_answer"):
        return "respond"
    return "act"


# ------------------------------------------------------------------
# Build graph
# ------------------------------------------------------------------

def build_agent_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("think", think_node)
    graph.add_node("act", act_node)
    graph.add_node("respond", respond_node)

    graph.set_entry_point("think")
    graph.add_conditional_edges("think", _route_after_think, {"act": "act", "respond": "respond"})
    graph.add_edge("act", "think")
    graph.add_edge("respond", END)

    return graph.compile()


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

_agent_graph = None


def get_agent():
    global _agent_graph
    if _agent_graph is None:
        _agent_graph = build_agent_graph()
    return _agent_graph


TOOL_ICONS = {
    "web_search": "🔍",
    "fetch_url": "🌐",
    "read_file": "📄",
    "list_dir": "📁",
    "memory_store": "💾",
    "memory_recall": "🧠",
    "sfis_query": "🏭",
    "sfis_2a_defects": "📊",
    "sfis_pvs_query": "🔩",
}


def _sfis_guard(task: str, final_answer: Optional[str], history: list, tool_map: dict):
    """
    If the model produced a Final Answer without ever calling sfis_query,
    and the task looks like an SFIS/SN query, force the tool call first.
    Returns (sn, result) to inject, or (None, None) to do nothing.
    """
    if not final_answer:
        return None, None
    already_called = any("sfis_query" in act for _, act, _ in history)
    if already_called:
        return None, None
    task_up = task.upper()
    is_sfis = (
        bool(_SN_RE.search(task))
        or "SERIAL" in task_up
        or " SN " in task_up
        or "SFIS" in task_up
        or task_up.startswith("SN")
    )
    if not is_sfis:
        return None, None
    m = _SN_RE.search(task)
    sn = m.group(1) if m else ""
    if not sn:
        return None, None
    print(f"[AGENT] WARNING: model skipped sfis_query — forcing tool call for SN={sn}")
    result = tool_map["sfis_query"].run(sn)
    return sn, result


def stream_agent(task: str, chat_history: list | None = None, stop_event=None):
    """Generator yielding UI events as the agent works. Used by the web server."""
    from src.inference import get_llm

    tool_map = {t.name: t for t in ALL_TOOLS}
    history: List[Tuple[str, str, str]] = []
    llm = get_llm()

    yield {"type": "start", "task": task}

    for i in range(MAX_ITERATIONS):
        if stop_event and stop_event.is_set():
            print(f"[AGENT] Stop requested — aborting at iter {i}")
            yield {"type": "stopped", "content": "Generation stopped by user."}
            return
        print(f"\n[AGENT] ── iter {i} ──────────────────────────────────")
        state: AgentState = {
            "task": task,
            "history": history,
            "final_answer": None,
            "iterations": i,
        }
        prompt = _build_prompt(state, chat_history)
        print(f"[AGENT iter={i}] Calling LLM ...")
        raw = llm.invoke(prompt, stop=["\nObservation:"])
        print(f"[AGENT iter={i}] Raw LLM output:\n{raw}\n---")
        thought, action, action_input, final_answer = _parse_llm_output(raw)
        print(f"[AGENT iter={i}] Parsed → thought={thought[:80]!r}  action={action!r}  input={action_input!r}  final_answer={'YES' if final_answer else 'NO'}")

        if thought:
            yield {"type": "thought", "content": thought}

        # Hard guard: if model skipped sfis_query for an SN query, force it now
        forced_sn, forced_result = _sfis_guard(task, final_answer, history, tool_map)
        if forced_sn:
            print(f"[AGENT iter={i}] GUARD triggered — forcing sfis_query for SN={forced_sn}")
            yield {"type": "tool_call", "name": "sfis_query", "icon": TOOL_ICONS.get("sfis_query", "🏭"), "input": forced_sn}
            yield {"type": "tool_result", "name": "sfis_query", "output": forced_result[:2000]}
            history.append(("Forced SFIS lookup before answering.", f"sfis_query|{forced_sn}", forced_result))
            print(f"[AGENT iter={i}] GUARD done — sfis_query result length={len(forced_result)} chars")
            final_answer = None
            continue

        if final_answer:
            print(f"[AGENT iter={i}] Final Answer produced — done.")
            yield {"type": "answer", "content": final_answer}
            _sfis_tools = {"sfis_query", "sfis_2a_defects", "sfis_pvs_query"}
            used_sfis = any(act.split("|")[0] in _sfis_tools for _, act, _ in history)
            if not used_sfis:
                try:
                    from src.memory import MemoryStore
                    MemoryStore().add(f"Task: {task}\nAnswer: {final_answer}", source="agent_result")
                except Exception:
                    pass
            break

        if action:
            print(f"[AGENT iter={i}] Tool call → {action!r}  input={action_input!r}")
            yield {
                "type": "tool_call",
                "name": action,
                "icon": TOOL_ICONS.get(action, "🔧"),
                "input": action_input,
            }
            tool = tool_map.get(action)
            if tool:
                try:
                    result = tool.run(action_input)
                except Exception as e:
                    result = f"Tool error: {e}"
            else:
                result = f"Unknown tool '{action}'. Available: {list(tool_map.keys())}"

            print(f"[AGENT iter={i}] Tool result ({len(result)} chars): {result[:200]}")
            yield {"type": "tool_result", "name": action, "output": result[:2000]}
            from src.rag import filter_observation
            filtered_result = filter_observation(task, result, tool_name=action)
            history.append((thought, f"{action}|{action_input}", filtered_result))
        else:
            yield {"type": "answer", "content": "I couldn't determine the next step. Please try rephrasing."}
            break
    else:
        yield {"type": "answer", "content": "Reached maximum iterations without a final answer."}

    yield {"type": "done"}


def run_agent(task: str, chat_history: list | None = None) -> str:
    """Run the reasoning agent on a task and return the final answer."""
    agent = get_agent()
    initial_state: AgentState = {
        "task": task,
        "history": [],
        "final_answer": None,
        "iterations": 0,
        "chat_history": chat_history,
        "_pending_action": None,
        "_pending_input": None,
    }
    result = agent.invoke(initial_state)
    answer = result.get("final_answer") or "Agent did not produce a final answer."

    # Persist to memory — skip SFIS results (must always be fetched live)
    _sfis_tools = {"sfis_query", "sfis_2a_defects", "sfis_pvs_query"}
    used_sfis = any(act.split("|")[0] in _sfis_tools for _, act, _ in result.get("history", []))
    if not used_sfis:
        try:
            from src.memory import MemoryStore
            MemoryStore().add(f"Task: {task}\nAnswer: {answer}", source="agent_result")
        except Exception:
            pass

    return answer
