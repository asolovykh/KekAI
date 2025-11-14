import logging
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from core.state import State
from core.system_prompts import SIMPLE_PROMPTS, PRO_PROMTS
from core.tools import web_search, describe_image, code_execution, browse_page, arxiv_search
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


def _ensure_defaults(state: Dict[str, Any]) -> State:
    return {
        "messages": state.get("messages", []),
        "plan": state.get("plan"),
        "draft": state.get("draft"),
        "validated": state.get("validated"),
        "summary": state.get("summary"),
        "validation_fail_count": state.get("validation_fail_count", 0),
        "mode": state.get("mode"),
        "print_to": state.get("print_to")
    }


def planner_node(llm: ChatOpenAI, state: State) -> Dict[str, Any]:
    prompt = SIMPLE_PROMPTS.PLANNER.value if state.get("mode", "simple") == "simple" else PRO_PROMTS.PLANNER.value
    
    msg = SystemMessage(content=prompt)
    if state.get('print_to', False):
        info = f'Пользователь спросил: {state['messages'][-1].content}'
        state['print_to'].update(label=info, state='running')
        state['thoughts'].append(info)
    res = llm.invoke([msg] + state["messages"])
    logger.debug('State info:\n' + str(state))
    steps = [s.strip("- •").strip() for s in (res.content or "").split("\n") if s.strip()]

    new_state = dict(state)
    new_state["messages"] = state["messages"] + [res]
    new_state["plan"] = steps[:8] or None
    if state.get('print_to', False):
        info = f'📔 Планировщик предложил следующие шаги: {'\n'.join(new_state.get('plan', 'Empty plan'))}'
        state['print_to'].update(label=info, state='running')
        state['thoughts'].append(info)

    return _ensure_defaults(new_state)


def supervisor_node(llm: ChatOpenAI, state: State) -> Dict[str, Any]:
    prompt = SIMPLE_PROMPTS.SUPERVISOR.value if state.get("mode", "simple") == "simple" else PRO_PROMTS.SUPERVISOR.value
    
    def serialize_messages(messages: List[BaseMessage]):
        role_map = {"human": "user", "ai": "assistant", "system": "system"}
        serialized = []
        for m in messages:
            role = role_map.get(getattr(m, "type", "human"), "user")
            serialized.append({"role": role, "content": m.content})
        return serialized

    base_msgs = state["messages"][:2]
    
    TOOLS = []
    if state.get('mode', 'simple') == 'simple':
        TOOLS = [web_search(state.get('mode', 'simple'))]
    else:
        TOOLS = [web_search(state.get('mode', 'simple')), describe_image, code_execution, browse_page, arxiv_search]
    supervisor_agent_graph = create_agent(
        model=llm, tools=TOOLS, system_prompt=prompt
    )
    result = supervisor_agent_graph.invoke({"messages": serialize_messages(base_msgs)})
    logger.debug('State info:\n' + str(state))
    draft = result["messages"][-1].content
    appended = result["messages"][2:] if len(result["messages"]) > 2 else []

    new_state = dict(state)
    new_state["messages"] = state["messages"] + appended
    new_state["draft"] = draft
    if state.get('print_to', False):
        info = f'🔍 Супервизор нашел: {new_state.get('draft', "Empty draft")}'
        state['print_to'].update(label=info, state='running')
        state['thoughts'].append(info)
    
    return _ensure_defaults(new_state)


def validator_node(llm: ChatOpenAI, state: State) -> Dict[str, Any]:
    prompt = SIMPLE_PROMPTS.VALIDATOR.value if state.get("mode", "simple") == "simple" else PRO_PROMTS.VALIDATOR.value
    
    draft = state.get("draft") or ""
    sys = SystemMessage(content=prompt)
    res = llm.invoke([sys, HumanMessage(content=draft)])
    logger.debug('State info:\n' + str(state))
    valid = "true" in (res.content or "").lower()

    count = state.get("validation_fail_count", 0)
    if not valid:
        count += 1

    new_state = dict(state)
    new_state["messages"] = state["messages"] + [AIMessage(content=f"[validator] {res.content}")]
    new_state["validated"] = valid
    new_state["validation_fail_count"] = count
    
    if state.get('print_to', False):
        info = '⚖️ Валидация успешно пройдена!' if new_state.get('validated', False) else '⚖️ Валидация провалилась!'
        state['print_to'].update(label=info, state='running')
        state['thoughts'].append(info)

    return _ensure_defaults(new_state)

def summarizer_node(llm: ChatOpenAI, state: State) -> Dict[str, Any]:
    prompt = SIMPLE_PROMPTS.SUMMARIZER.value if state.get("mode", "simple") == "simple" else PRO_PROMTS.SUMMARIZER.value
    
    history = str(state["messages"])
    sys = SystemMessage(content=prompt.format(history=history))
    res = llm.invoke([sys])
    logger.debug('State info:\n' + str(state))

    new_state = dict(state)
    new_state["messages"] = state["messages"] + [AIMessage(content=f"[summary] {res.content}")]
    new_state["summary"] = res.content
    
    if state.get('print_to', False):
        info = f'Результат получен: {new_state.get('summary', "Empty summary")}'
        state['print_to'].update(label=info, state='complete')
        state['thoughts'].append(info)
    return _ensure_defaults(new_state)


def analyzer_node(llm: ChatOpenAI, state: State) -> Dict[str, Any]:
    prompt = SIMPLE_PROMPTS.ANALYZER.value if state.get("mode", "simple") == "simple" else PRO_PROMTS.ANALYZER.value