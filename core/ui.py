import streamlit as st
from langchain_core.messages import HumanMessage
from core.state import State
import logging

logger = logging.getLogger(__name__)


# === Класс UI ===
class ResearchAssistantUI:
    _instanse = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        if cls._instanse is None:
            cls._instanse = super().__new__(cls)
        
        return cls._instanse
    
    def __init__(
        self,
        agent,
        title: str = "Research Assistant",
        icon: str = "🔍"
    ):
        if not self._initialized:
            self.agent = agent
            self.title = title
            self.icon = icon

            self._setup_page()
            self._inject_css()
            logger.info('UI initialized')
            self._initialized = True

    def _setup_page(self):
        st.set_page_config(
            page_title=f"{self.icon} {self.title}",
            page_icon=self.icon,
            layout="wide",
            initial_sidebar_state="expanded"
        )
        st.session_state.selected_mode = "simple"
        logger.info('Page setup')

    def _inject_css(self):
        st.markdown("""
        <style>
        .main {
            background-color: #f0f2f5;
        }
        .stChatMessage {
            border-radius: 12px;
            padding: 12px;
            margin: 8px 0;
            max-width: 85%;
        }
        .user-message {
            background-color: #007bff;
            color: white;
            margin-left: auto;
            text-align: right;
        }
        .assistant-message {
            background-color: white;
            border: 1px solid #e0e0e0;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        }
        .stButton > button {
            background: linear-gradient(135deg, #1f77b4, #0d47a1);
            color: white;
            border: none;
            border-radius: 25px;
            padding: 10px 24px;
            font-weight: 500;
            transition: all 0.2s;
        }
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .stSidebar .stSelectbox label,
        .stSidebar .stButton > button {
            font-weight: bold;
        }
        .source-link {
            color: #1f77b4;
            text-decoration: none;
        }
        .source-link:hover {
            text-decoration: underline;
        }
        </style>
        """, unsafe_allow_html=True)

    def _render_sidebar(self) -> bool:
        with st.sidebar:
            st.header("⚙️ Настройки")
            mode_option = st.selectbox(
                "Режим поиска",
                options=["simple", "pro"],
                format_func=lambda x: {
                    "simple": "⚡ Simple (быстрый)",
                    "pro": "🔬 Pro (глубокий)"
                }[x],
                index=["simple", "pro"].index(st.session_state.selected_mode),
                help="Simple — быстрый ответ. Pro — глубокий анализ."
            )
            st.session_state.selected_mode = mode_option
            
            clear_chat = st.button("🗑️ Очистить чат", use_container_width=True)
            st.markdown("---")

            st.subheader("Примеры сложных вопросов (Pro)")
            st.markdown("""
            - "Какой фильм выиграл больше Оскаров: 'Титаник' или 'Бен-Гур'?"
            - "Сравни экономику Франции и Германии по ВВП на душу."
            - "Как теория относительности повлияла на GPS?"
            - "Сравни влияние Толстого и Достоевского."
            """)
            st.markdown("---")

            with st.expander("📊 Бенчмарки"):
                st.info("В разработке...")
            
            logger.info('Sidebar rendered')
            return clear_chat

    def _display_chat_history(self):
        for msg in st.session_state.messages:
            with st.chat_message(msg['role']):
                st.markdown(msg["content"], unsafe_allow_html=True)
                
                # if msg.get("sources"):
                #     with st.expander("Источники & Детали", expanded=False):
                #         for i, src in enumerate(msg["sources"][:5], 1):
                #             st.markdown(f"{i}. <a href='{src}' target='_blank' class='source-link'>{src}</a>", unsafe_allow_html=True)
                        
                #         if msg.get("mode") == Mode.PRO.value and msg.get("reasoning"):
                #             st.subheader("Шаги рассуждений")
                #             for step in msg["reasoning"]:
                #                 st.markdown(step, unsafe_allow_html=True)

    def run(self):
        st.title(f"{self.icon} {self.title}")
        st.markdown("Задавай вопросы. (История сохраняется)")

        # Инициализация сессии
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Сайдбар
        if self._render_sidebar():
            st.session_state.messages = []
            st.rerun()

        # История чата
        self._display_chat_history()

        # Ввод пользователя
        if prompt := st.chat_input("Что ты хочешь узнать?"):
            # Добавляем сообщение пользователя
            msg: State = {
                "messages": [HumanMessage(content=prompt)],
                "plan": None,
                "draft": None,
                "validated": None,
                "summary": None,
                "validation_fail_count": 0,
                "mode": st.session_state.selected_mode,
                "print_to": None,
            }
            user_msg = {"role": "user", "content": prompt}
            st.session_state.messages.append(user_msg)
            with st.chat_message("user"):
                st.markdown(prompt)

            # Определяем режим
            with st.chat_message("assistant"):
                if st.session_state.selected_mode == 'pro':
                    with st.status("Глубокий анализ (Pro)...", expanded=True) as status:
                        msg['print_to'] = status
                        res = self.agent.invoke(msg)
                else:
                    with st.spinner("Ищу быстрый ответ..."):
                        res = self.agent.invoke(msg)

                # Сохраняем полный ответ
                response_msg = {
                    "role": "assistant",
                    "content": res.get('summary', 'Empty string'),
                #     "sources": res.sources,
                #     "mode": res.mode,
                }
                st.session_state.messages.append(response_msg)
                st.rerun()

        # Футер
        st.markdown("---")
        st.markdown("*Powered by Qwen & Tavily. Настройте `.env` для API ключей.*")
