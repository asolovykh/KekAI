import os
import logging
from dotenv import load_dotenv
from core.ui import ResearchAssistantUI
from functools import partial
# from core.ui import run_ui
from core.nodes import *
from core.structure import create_graph
from core.llm_connector import get_llm_client

logging.basicConfig(
    filename='KekAI.log',
    filemode='w',
    level=logging.DEBUG
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
    llm_nodes = {
        'planner': partial(planner_node, get_llm_client(MODEL, temperature=0.)),
        'supervisor': partial(supervisor_node, get_llm_client(MODEL, temperature=0.1)),
        'validator': partial(validator_node, get_llm_client(MODEL, temperature=0.1)),
        'summarizer': partial(summarizer_node, get_llm_client(MODEL, temperature=0.1))
    }
    logger.info('LLM clients set')
    app = create_graph(**llm_nodes).compile()
    logger.info('Graph compiled')
    ui = ResearchAssistantUI(app)
    ui.run()
    # run_ui()