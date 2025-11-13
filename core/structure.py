from langgraph.graph import END, START, StateGraph
from core.state import State


def create_graph(state: State, **node_funcs) -> StateGraph:
    graph = StateGraph(state)
    graph.add_node('planner', node_funcs['planner'])
    graph.add_node('supervisor', node_funcs['supervisor'])
    graph.add_node('validator', node_funcs['validator'])
    graph.add_node('summarizer', node_funcs['summarizer'])
    
    graph.add_edge(START, 'planner')
    graph.add_edge('planner', 'supervisor')
    graph.add_edge('supervisor', 'validator')
    graph.add_edge("summarizer", END)
    
    return graph
