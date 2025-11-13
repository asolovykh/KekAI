import os
import logging
from dotenv import load_dotenv
# from core.ui import run_ui
from langchain_core.messages import HumanMessage
from core.state import State
from core.structure import create_graph
from core.llm_connector import get_llm_client

logging.basicConfig(
    filename='KekAI.log',
    filemode='w',
    level=logging.INFO
)
logger = logging.getLogger(__name__)
load_dotenv()

DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
MAX_RESULTS = int(os.getenv("MAX_RESULTS", "5"))
API_KEY = os.getenv("API_KEY")
MODEL = os.environ.get("BASE_MODEL")
MODEL_SUPERVISOR = os.environ.get("BASE_MODEL")


if __name__ == '__main__':
    assert API_KEY is not None, "Insert your API_KEY into .env file!"
    query = input('Enter your question: ')
    init: State = {
        "messages": [HumanMessage(content=query)],
        "plan": None,
        "draft": None,
        "validated": None,
        "summary": None,
        "validation_fail_count": 0,
    }
    logger.info('State prepared')
    llm_nodes = {
        'planner': get_llm_client(MODEL, temperature=0.),
        'supervisor': get_llm_client(MODEL, temperature=0.1),
        'validator': get_llm_client(MODEL, temperature=0.1),
        'summarizer': get_llm_client(MODEL, temperature=0.1)
    }
    logger.info('LLM clients set')
    app = create_graph(State, **llm_nodes).compile()
    logger.info('Graph compiled')
    state = app.invoke(init)
    print("\n--- SUMMARY ---\n", state.get("summary"))
    # run_ui()