from langgraph.graph import END, START, StateGraph
from core.state import State


def create_graph(**node_funcs) -> StateGraph:
    graph = StateGraph(State)
    graph.add_node('planner', node_funcs['planner'])
    graph.add_node('supervisor', node_funcs['supervisor'])
    graph.add_node('validator', node_funcs['validator'])
    graph.add_node('summarizer', node_funcs['summarizer'])
    
    graph.add_edge(START, 'planner')
    graph.add_edge('planner', 'supervisor')
    graph.add_edge('supervisor', 'validator')
    
    def validation_routing(state: State) -> bool:
        # True -> summarizer, False -> supervisor (повтор)
        validated = state.get("validated", False)
        fail_count = state.get("validation_fail_count", 0)
        return True if (validated or fail_count >= 2) else False

    graph.add_conditional_edges("validator", validation_routing, {True: "summarizer", False: "supervisor"})
    graph.add_edge("summarizer", END)
    
    return graph
