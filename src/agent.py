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
from typing import TypedDict, List, Tuple, Optional, Annotated
import operator

from langgraph.graph import StateGraph, END

from src.config import MAX_ITERATIONS, AGENT_VERBOSE
from src.tools import ALL_TOOLS

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# State
# ------------------------------------------------------------------

class AgentState(TypedDict):
    task: str
    history: Annotated[List[Tuple[str, str, str]], operator.add]
    final_answer: Optional[str]
    iterations: int


# ------------------------------------------------------------------
# Prompt helpers
# ------------------------------------------------------------------

TOOL_DESCRIPTIONS = "\n".join(
    f"- {t.name}: {t.description}" for t in ALL_TOOLS
)

SYSTEM_PROMPT = f"""You are a smart manufacturing AI assistant running locally on Apple Silicon (mlx-lm).
You help engineers query the internal SFIS manufacturing system, search the web for technical information, read local files, and reason through complex problems.

## Your capabilities
{TOOL_DESCRIPTIONS}

## Key guidelines
- SFIS is the internal manufacturing database at http://10.52.1.9. Use sfis_query to look up any serial number (SN).
- sfis_query returns structured fields: Phase, Model, Config, SMT Line, Panel SN, SN position in panel, Failed Date, Lab In Time, Group Name, Failure Message, and List of Failing Tests.
- If the first sfis_query result doesn't have vendor detail, call it again with a component location appended (e.g. "SN123, R2251") to get Vendor/Lot/Date-Code data.
- Always check CONVERSATION HISTORY and RELEVANT MEMORIES (injected below) before searching the web or calling tools, to avoid repeating work.
- When searching the web, today's date is automatically appended to queries.
- Use memory_store to save concise facts you learn (e.g. "SN ABC123: Phase EVT2, failed FCT at burn_in station on 2024-05-10"). Keep stored facts short and specific.
- Use memory_recall before starting a task to surface relevant prior findings.
- Be concise and structured in your final answers. Use bullet points or tables for manufacturing data.

## Response format
Thought: <your reasoning>
Action: <tool_name>
Action Input: <input>

OR when done:

Thought: I have enough information.
Final Answer: <your answer>

Rules:
- Always start with Thought.
- One tool per step.
- Never repeat the same Action + Input.
- Always output "Final Answer:" when done.
"""


def _build_prompt(state: AgentState, chat_history: list | None = None) -> str:
    lines = [SYSTEM_PROMPT]

    # Inject recent conversation so the model can answer follow-ups
    if chat_history:
        lines.append("\n## Conversation History")
        for user_msg, assistant_msg in chat_history[-6:]:
            lines.append(f"User: {user_msg}")
            short = assistant_msg[:600] + "…" if len(assistant_msg) > 600 else assistant_msg
            lines.append(f"Assistant: {short}")
        lines.append("")

    # Auto-inject relevant memories on the first step (before any tool calls)
    if not state["history"]:
        try:
            from src.memory import MemoryStore
            memories = MemoryStore().search_text(state["task"], k=3)
            if memories and "No relevant memories" not in memories:
                lines.append("## Relevant Memories")
                lines.append(memories)
                lines.append("")
        except Exception:
            pass

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
        return thought_part, None, None, final_match.group(1).strip()

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

    if state["iterations"] >= MAX_ITERATIONS:
        return {**state, "final_answer": "Reached maximum iterations without a final answer."}

    prompt = _build_prompt(state)
    llm = get_llm()
    raw = llm.invoke(prompt, stop=["\nObservation:"])

    if AGENT_VERBOSE:
        logger.info("LLM raw output:\n%s", raw)

    thought, action, action_input, final_answer = _parse_llm_output(raw)

    if final_answer:
        return {**state, "final_answer": final_answer}

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

    if tool is None:
        observation = f"Unknown tool '{action}'. Available: {list(tool_map.keys())}"
    else:
        try:
            observation = tool.run(action_input)
        except Exception as e:
            observation = f"Tool error: {e}"

    if AGENT_VERBOSE:
        logger.info("Tool '%s' → %s", action, observation[:200])

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


def stream_agent(task: str, chat_history: list | None = None):
    """Generator yielding UI events as the agent works. Used by the web server."""
    from src.inference import get_llm

    tool_map = {t.name: t for t in ALL_TOOLS}
    history: List[Tuple[str, str, str]] = []
    llm = get_llm()

    yield {"type": "start", "task": task}

    for i in range(MAX_ITERATIONS):
        state: AgentState = {
            "task": task,
            "history": history,
            "final_answer": None,
            "iterations": i,
        }
        prompt = _build_prompt(state, chat_history)
        raw = llm.invoke(prompt, stop=["\nObservation:"])
        thought, action, action_input, final_answer = _parse_llm_output(raw)

        if thought:
            yield {"type": "thought", "content": thought}

        if final_answer:
            yield {"type": "answer", "content": final_answer}
            try:
                from src.memory import MemoryStore
                MemoryStore().add(f"Task: {task}\nAnswer: {final_answer}", source="agent_result")
            except Exception:
                pass
            break

        if action:
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

            yield {"type": "tool_result", "name": action, "output": result[:2000]}
            history.append((thought, f"{action}|{action_input}", result))
        else:
            yield {"type": "answer", "content": "I couldn't determine the next step. Please try rephrasing."}
            break
    else:
        yield {"type": "answer", "content": "Reached maximum iterations without a final answer."}

    yield {"type": "done"}


def run_agent(task: str) -> str:
    """Run the reasoning agent on a task and return the final answer."""
    agent = get_agent()
    initial_state: AgentState = {
        "task": task,
        "history": [],
        "final_answer": None,
        "iterations": 0,
    }
    result = agent.invoke(initial_state)
    answer = result.get("final_answer") or "Agent did not produce a final answer."

    # Persist the answer to memory for future runs
    try:
        from src.memory import MemoryStore
        mem = MemoryStore()
        mem.add(f"Task: {task}\nAnswer: {answer}", source="agent_result")
    except Exception:
        pass

    return answer
