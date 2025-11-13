from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from core.state import State
from core.prompts import (
    planner_system_prompt,
    summary_system_prompt,
    supervisor_system_prompt,
    validator_system_prompt,
)
from core.tools import web_search, describe_image, e2b_run_code
from typing import Dict, List, Any
import os


TOOLS = [web_search, describe_image, e2b_run_code]

def _ensure_defaults(state: Dict[str, Any]) -> State:
    return {
        "messages": state.get("messages", []),
        "plan": state.get("plan"),
        "draft": state.get("draft"),
        "validated": state.get("validated"),
        "summary": state.get("summary"),
        "validation_fail_count": state.get("validation_fail_count", 0),
    }


def planner_node(state: State, llm: ChatOpenAI) -> Dict[str, Any]:
    sys = SystemMessage(content=planner_system_prompt)
    print(sys, state['messages'], sep='\n')
    res = llm.invoke([sys] + state["messages"])
    steps = [s.strip("- •").strip() for s in (res.content or "").split("\n") if s.strip()]

    new_state = dict(state)
    new_state["messages"] = state["messages"] + [res]
    new_state["plan"] = steps[:8] or None
    
    print("\n--- PLANNER ---\n", steps[:8] or None)

    return _ensure_defaults(new_state)

def supervisor_node(state: State, llm: ChatOpenAI) -> Dict[str, Any]:
    def serialize_messages(messages: List[BaseMessage]):
        role_map = {"human": "user", "ai": "assistant", "system": "system"}
        serialized = []
        for m in messages:
            role = role_map.get(getattr(m, "type", "human"), "user")
            serialized.append({"role": role, "content": m.content})
        return serialized

    base_msgs = state["messages"][:2]
    
    supervisor_agent_graph = create_agent(
        model=llm, tools=TOOLS, system_prompt=supervisor_system_prompt, # debug=True
    )
    result = supervisor_agent_graph.invoke({"messages": serialize_messages(base_msgs)})
    draft = result["messages"][-1].content
    appended = result["messages"][2:] if len(result["messages"]) > 2 else []

    new_state = dict(state)
    new_state["messages"] = state["messages"] + appended
    new_state["draft"] = draft
    
    print("\n--- SUPERVISOR ---\n", draft)
    
    return _ensure_defaults(new_state)

def validator_node(state: State, llm: ChatOpenAI) -> Dict[str, Any]:
    draft = state.get("draft") or ""
    sys = SystemMessage(content=validator_system_prompt)
    res = llm.invoke([sys, HumanMessage(content=draft)])
    valid = "true" in (res.content or "").lower()

    count = state.get("validation_fail_count", 0)
    if not valid:
        count += 1

    new_state = dict(state)
    new_state["messages"] = state["messages"] + [AIMessage(content=f"[validator] {res.content}")]
    new_state["validated"] = valid
    new_state["validation_fail_count"] = count
    
    print("\n--- VALIDATOR ---\n", res.content)

    return _ensure_defaults(new_state)

def summarizer_node(state: State, llm: ChatOpenAI) -> Dict[str, Any]:
    history = str(state["messages"])
    sys = SystemMessage(content=summary_system_prompt.format(history=history))
    res = llm.invoke([sys])

    new_state = dict(state)
    new_state["messages"] = state["messages"] + [AIMessage(content=f"[summary] {res.content}")]
    new_state["summary"] = res.content
    return _ensure_defaults(new_state)