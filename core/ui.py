import streamlit as st


def run_ui():
    st.set_page_config(
        page_title="🔍 KekAI Research Assistant",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    # Кастомный CSS для стиля как у Grok: минималистичный, темный/светлый, чат-подобный
    st.markdown("""
    <style>
    .main {
        background-color: #f0f2f5;
    }
    .stChatMessage {
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
    }
    .user-message {
        background-color: #007bff;
        color: white;
        text-align: right;
    }
    .assistant-message {
        background-color: white;
        border: 1px solid #ddd;
    }
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border-radius: 20px;
        padding: 10px 20px;
    }
    .stSidebar .stSelectbox label {
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    st.title("🔍 Research Assistant")
    st.markdown("Задавай вопросы, я автоматически выберу режим (Simple для быстрых фактов, Pro для глубокого анализа). История чата сохраняется.")
    
    with st.sidebar:
        st.header("⚙️ Настройки")
        clear_chat = st.button("🗑️ Очистить чат")
        st.markdown("---")
        st.subheader("💡 Примеры сложных вопросов (Pro)")
        st.markdown("""
        - "Какой фильм выиграл больше Оскаров: 'Титаник' или 'Бен-Гур'?" (сравнение)
        - "Сравни экономику Франции и Германии по ВВП на душу населения." (анализ)
        - "Как теория относительности Эйнштейна повлияла на современную физику?" (цепочка фактов)
        - "Сравни влияние Толстого и Достоевского на русскую литературу." (глубокий обзор)
        """)
        st.markdown("---")
        benchmark_expander = st.expander("📊 Бенчмарки")
        run_benchmark = benchmark_expander.button("Запустить бенчмарки")
    # Инициализация сессии для чата
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # Кнопка очистки чата
    if clear_chat:
        st.session_state.messages = []
        st.rerun()
    # Отображение истории чата
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 Источники & Детали"):
                    for i, src in enumerate(message["sources"][:5], 1):
                        st.markdown(f"{i}. [{src}]({src})")
                    if message.get("mode") == "pro" and "reasoning" in message:
                        st.subheader("🧠 Шаги рассуждений (с источниками)")
                        for step in message["reasoning"]:
                            st.markdown(step)  # Markdown для кликабельных ссылок
                    if message.get("mode") == "pro" and "metrics" in message:
                        st.subheader("📈 Метрики")
                        metrics_df = pd.DataFrame(list(message["metrics"].items()), columns=["Метрика", "Значение"])
                        st.dataframe(metrics_df, hide_index=True, use_container_width=True)
    
    if prompt := st.chat_input("Что ты хочешь узнать?"):
        # Добавляем сообщение пользователя
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        # Автоматическое определение режима
        with st.chat_message("assistant"):
            with st.spinner("🤔 Определяю режим..."):
                mode = classify_mode(prompt)
                mode_display = "Simple (быстрый)" if mode == Mode.SIMPLE else "Pro (глубокий)"
                st.info(f"**Режим:** {mode_display}")
                st.markdown("---")
            # Обработка запроса
            if mode == Mode.SIMPLE:
                with st.spinner("🔍 Ищу ответ..."):
                    res = simple_mode(prompt)
                # Форматированный ответ
                full_response = f"**Ответ:** {res['answer']}\n\n⏱️ {res['time_estimate']}"
                st.markdown(full_response)
            else:
                # Для Pro: используем st.status для стабильного прогресса
                with st.status("🔍 Ищу ответ в Pro режиме...", expanded=True) as status:
                    res = pro_mode(prompt, sub_mode if sub_mode != "none" else None, status)
                # Улучшенный вывод: detailed_answer + таблица из key_metrics (если есть)
                full_response = f"**Ответ:** {res['answer']}\n\n{res.get('detailed_answer', '')}\n\n⏱️ {res['time_estimate']}"
                st.markdown(full_response)
                if res.get('key_metrics'):
                    # Динамическая таблица: предполагаем, что key_metrics — dict с метриками (адаптируется к вопросам)
                    metrics_data = {k: [v] for k, v in res['key_metrics'].items()}  # Преобразуем в список для DF
                    if len(list(res['key_metrics'].keys())) > 0:
                        df = pd.DataFrame(metrics_data)
                        st.subheader("🔍 Ключевые метрики")
                        st.dataframe(df, hide_index=True, use_container_width=True)
            # Сохраняем в сессию с доп. данными
            response_msg = {
                "role": "assistant",
                "content": full_response,
                "sources": res["sources"],
                "mode": res["mode"]
            }
            if mode == Mode.PRO:
                response_msg.update({
                    "reasoning": res.get("reasoning", []),
                    "metrics": res.get("metrics", {}),
                    "detailed_answer": res.get("detailed_answer", ""),
                    "key_metrics": res.get("key_metrics", {})
                })
            st.session_state.messages.append(response_msg)
            st.rerun()
    # Бенчмарки в сайдбаре
    if run_benchmark and benchmark_expander:
        with st.container():
            st.subheader("=== SimpleQA Benchmark ===")
            simple_answers = []
            for q in SIMPLEQA_QUESTIONS:
                res = simple_mode(q["q"])
                simple_answers.append(res["answer"])
            simple_acc = evaluate_simpleqa(simple_answers, [q["gt"] for q in SIMPLEQA_QUESTIONS])
            st.metric("Accuracy (%)", f"{simple_acc['accuracy']:.2f}")
            st.subheader("=== FRAMES Benchmark ===")
            frames_results = []
            for q in FRAMES_QUESTIONS:
                res = pro_mode(q["q"])
                frames_results.append(res)
                with st.expander(f"Q: {q['q'][:100]}..."):
                    st.write(f"**Answer:** {res['answer'][:150]}...")
            frames_metrics = evaluate_frames(frames_results, FRAMES_QUESTIONS)
            metrics_df = pd.DataFrame(list(frames_metrics.items()), columns=["Метрика", "Значение"])
            st.dataframe(metrics_df, hide_index=True, use_container_width=True)
    # Футер
    st.markdown("---")
    st.markdown("*Powered by Qwen & Tavily. Используй .env для ключей.*")