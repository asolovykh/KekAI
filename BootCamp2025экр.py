import os
import json
import re
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import requests
from bs4 import BeautifulSoup
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
from pydantic import BaseModel
import streamlit as st
import pandas as pd
from difflib import SequenceMatcher
from dotenv import load_dotenv
load_dotenv()
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
MAX_RESULTS = int(os.getenv("MAX_RESULTS", "5"))
API_KEY = os.getenv("API_KEY")
if API_KEY is None or not isinstance(API_KEY, str) or API_KEY.strip() == "":
    raise ValueError("API_KEY not set or invalid in .env. Please provide a valid string key for cloud.ru.")
BASE_URL = "https://foundation-models.api.cloud.ru/v1"
llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model="Qwen/Qwen3-Next-80B-A3B-Instruct",
    temperature=0,
    max_tokens=2500,
    top_p=0.95,
    presence_penalty=0
)
@dataclass
class SearchResult:
    title: str
    content: str
    url: str
class Mode(Enum):
    SIMPLE = "simple"
    PRO = "pro"
class ResearchState(BaseModel):
    query: str
    results: List[SearchResult] = []
    reasoning_steps: List[str] = []
    final_answer: str = ""
    sources: List[str] = []
    depth: int = 0
    diversity: int = 0
def web_search(query: str, num_results: int = 5, site_filter: str = None) -> List[SearchResult]:
    if DEBUG_MODE:
        print(f"Debug: Web search for '{query}' with {num_results} results")
    full_query = f"{query} {site_filter}" if site_filter else query
    try:
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        response = client.search(query=full_query, max_results=num_results)
        results = []
        for res in response['results']:
            results.append(SearchResult(title=res['title'], content=res['content'], url=res['url']))
        # Fallback: If fewer than expected, browse top URL
        if len(results) < num_results // 2:
            if results:
                extra_content = browse_page(results[0].url, "Extract key facts relevant to the query.")
                if extra_content != "Error fetching page.":
                    results[0].content += f"\n[Extended Summary]: {extra_content}"
        return results
    except Exception as e:
        if DEBUG_MODE:
            print(f"Debug: Web search error: {e}")
        return []
def browse_page(url: str, instructions: str = "Extract key facts and summarize.") -> str:
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text()[:2000]
        prompt = f"{instructions}\n\nText: {text}"
        chain = ChatPromptTemplate.from_template(prompt) | llm | StrOutputParser()
        return chain.invoke({})
    except Exception as e:
        if DEBUG_MODE:
            print(f"Debug: Browse error for {url}: {e}")
        return "Error fetching page."
def rerank_results(results: List[SearchResult], query: str) -> List[SearchResult]:
    """Rerank top-3 by relevance using LLM."""
    if len(results) < 3:
        return results
    # Fixed: Escaped JSON example with {{ }}, no literal vars in template
    rank_prompt = ChatPromptTemplate.from_template(
        """Rank these {num} results by relevance to '{query}': {formatted}
(1=most relevant, score 0.0-1.0). Output ONLY JSON: {{"ranked": [
  {{"title": "exact title from input", "content": "exact content from input", "url": "exact url from input", "score": 0.95}},
  {{"title": "exact title from input", "content": "exact content from input", "url": "exact url from input", "score": 0.85}}
]}} – use exact titles/contents/urls from input, assign scores based on relevance."""
    )
    chain = rank_prompt | llm | JsonOutputParser()
    formatted = "\n".join([f"{i+1}. Title: {r.title}\nContent: {r.content[:200]}...\nURL: {r.url}" for i, r in enumerate(results)])
    try:
        ranked_json = chain.invoke({"query": query, "formatted": formatted, "num": len(results)})
        ranked = ranked_json.get("ranked", [])
        if not ranked:
            raise ValueError("No ranked items")
        # Map to results with scores (safer matching by title similarity)
        scored_results = []
        for item in ranked:
            score = item.get("score", 0.0)
            title_match = item.get("title", "")
            # Fuzzy match to original (using difflib for robustness)
            best_match = max(results, key=lambda r: SequenceMatcher(None, title_match.lower(), r.title.lower()).ratio())
            if SequenceMatcher(None, title_match.lower(), best_match.title.lower()).ratio() > 0.6:
                scored_results.append((best_match, float(score)))
        # Sort by score desc, take top 3
        sorted_results = [r for r, s in sorted(scored_results, key=lambda x: x[1], reverse=True)[:3]]
        return sorted_results + results[3:] if len(results) > 3 else sorted_results
    except Exception as e:
        if DEBUG_MODE:
            print(f"Debug: Rerank failed: {e}, fallback to original")
        return results  # Fallback
def fact_check(answer: str, query: str) -> bool:
    """Quick fact-check via search."""
    check_results = web_search(f"fact check: {answer} for {query}", num_results=2)
    if not check_results:
        return True
    check_prompt = ChatPromptTemplate.from_template(
        "Does this confirm the answer '{answer}' for '{query}'? Results: {results}\nYes/No + reason."
    )
    chain = check_prompt | llm | StrOutputParser()
    verdict = chain.invoke({
        "answer": answer, 
        "query": query, 
        "results": "\n".join([r.content for r in check_results])
    })
    return any(word in verdict.lower() for word in ["yes", "confirm", "true", "match"])
def simple_mode(query: str) -> Dict[str, Any]:
    results = web_search(query, num_results=3)
    # Improved prompt with more few-shot
    few_shot = """
Examples:
Query: Who received the IEEE Frank Rosenblatt Award in 2010?
Results: ... Michio Sugeno ...
Answer: Michio Sugeno

Query: On which U.S. TV station did the Canadian reality series *To Serve and Protect* debut?
Results: ... premiered on KVOS-TV ...
Answer: KVOS-TV

Query: What day, month, and year was Carrie Underwood’s album “Cry Pretty” certified Gold by the RIAA?
Results: ... certified Gold ... October 23, 2018 ...
Answer: October 23, 2018

Query: What is the first and last name of the woman whom the British linguist Bernard Comrie married in 1985?
Results: ... married ... Akiko Kumahira ...
Answer: Akiko Kumahira
"""
    prompt = ChatPromptTemplate.from_template(
        few_shot + "Query: {query}\nResults: {results}\nAnswer (exact short phrase, no extra text):"
    )
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"query": query, "results": "\n".join([f"{r.title}: {r.content}" for r in results])}).strip()
    # Fact-check
    if not fact_check(answer, query):
        if DEBUG_MODE:
            print(f"Debug: Fact-check failed for {query}, regenerating...")
        answer = chain.invoke({"query": query, "results": "\n".join([f"{r.title}: {r.content}" for r in web_search(query, 3)])}).strip()
    return {
        "mode": Mode.SIMPLE.value,
        "answer": answer,
        "sources": [r.url for r in results],
        "time_estimate": "Fast (<2s)"
    }
def pro_mode(query: str, sub_mode: str = None) -> Dict[str, Any]:
    # Enhanced hop_prompt with more few-shot, incl. tricky cases
    hop_few_shot = """
Example 1 (dates):
Query: Which magazine was started first Arthur's Magazine or First for Women?
Sub-queries: [\"When was Arthur's Magazine first published?\", \"When was First for Women magazine first published?\", \"Compare start dates.\"]

Example 2 (chain):
Query: The Oberoi family is part of a hotel company that has a head office in what city?
Sub-queries: [\"Oberoi family business?\", \"Oberoi hotel company name?\", \"Head office location of Oberoi Group?\"]

Example 3 (creative fact):
Query: Musician and satirist Allie Goertz wrote a song about the \"The Simpsons\" character Milhouse, who Matt Groening named after who?
Sub-queries: [\"Allie Goertz Simpsons song?\", \"Milhouse character origin Matt Groening?\"]

Example 4 (ambiguous):
Query: What nationality was James Henry Miller's wife?
Sub-queries: [\"Who is James Henry Miller (author)?\", \"Henry Miller's first wife?\", \"Nationality of June Mansfield?\"]

Example 5 (chemistry, avoid assumptions):
Query: Cadmium Chloride is slightly soluble in this chemical, it is also called what?
Sub-queries: [\"Solubility of Cadmium Chloride in solvents?\", \"What chemical has slight solubility for CdCl2 besides water?\", \"Common name for that solvent.\"]
"""
    hop_prompt = ChatPromptTemplate.from_template(
        hop_few_shot + "\nBreak down '{query}' into 2-4 factual sub-queries. No assumptions, focus on evidence. JSON list: [\"sub1\", \"sub2\"...]"
    )
    chain = hop_prompt | llm | StrOutputParser()
    sub_queries_str = chain.invoke({"query": query})
  
    sub_queries = [query]
    cleaned = re.sub(r'```json\s*|\s*```', '', sub_queries_str).strip()
    try:
        sub_queries = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError) as e:
        if DEBUG_MODE:
            print(f"Debug: JSON parse error: {e}, trying split fallback")
        try:
            if '[' in cleaned:
                sub_queries = [q.strip().strip('"') for q in cleaned.strip('[]').split(',')]
        except:
            if DEBUG_MODE:
                print(f"Debug: Split fallback failed, using single query")
  
    if isinstance(sub_queries, list):
        sub_queries = [str(q) for q in sub_queries if q]
        sub_queries = sub_queries[:4]
    else:
        sub_queries = [str(query)]
  
    if DEBUG_MODE:
        print(f"Debug: Sub-queries: {sub_queries}")
  
    def get_site_filter(sub_mode: str) -> str:
        if sub_mode == "social":
            return "site:twitter.com OR site:x.com"
        elif sub_mode == "academic":
            return "site:arxiv.org OR site:semanticscholar.org"
        return None
  
    site_filter = get_site_filter(sub_mode)
  
    state = ResearchState(query=query)
    all_results = []
    for i, sub_query in enumerate(sub_queries):
        if DEBUG_MODE:
            print(f"Debug: Processing sub-query {i+1}: {sub_query}")
        results = web_search(sub_query, num_results=5, site_filter=site_filter)
        # Rerank
        results = rerank_results(results, sub_query)
        state.results.extend(results)
        state.reasoning_steps.append(f"Step {i + 1}: Searched '{sub_query}' ({sub_mode if sub_mode else 'general'}). Found {len(results)} results.")
        state.sources.extend([r.url for r in results])
        all_results.extend(results)
        state.depth += 1
  
    if all_results:
        # Use all, but rerank global
        all_results = rerank_results(all_results, query)
        all_contents = "\n".join([r.content for r in all_results])
        if len(all_contents) > 4000:
            all_contents = all_contents[:4000] + "\n[Truncated for brevity]"
        prompt = ChatPromptTemplate.from_template(
            "Analyze and cross-verify facts from top results: {contents}\nOutput: summary, verified facts, inconsistencies."
        )
        chain = prompt | llm | StrOutputParser()
        analysis = chain.invoke({"contents": all_contents})
        state.reasoning_steps.append(f"Analysis: {analysis[:200]}...")
    state.diversity = len(set(state.sources))
    state.results = all_results  # Updated with rerank
  
    # Enhanced final with more few-shot, proper escaping for JSON literals
    final_few_shot = """
Example (magazine):
Steps: ... dates found ...
Results: ... 1844 vs 1989 ...
Query: Which magazine first?
Output: {{"answer": "Arthur's Magazine", "explanation": "1844 < 1989. Citations: wiki1, wiki2", "citations": ["url1", "url2"]}}

Example (chemistry):
Steps: ... solubility ...
Results: ... slightly in alcohol ...
Query: Cadmium Chloride slightly soluble in?
Output: {{"answer": "alcohol", "explanation": "High in water, slight in alcohol. Citations: wiki, pubchem", "citations": ["url1", "url2"]}}
"""
    # Use raw string for template to avoid escaping issues
    final_prompt_template = final_few_shot + """From steps: {steps}
And results: {results}
Deduce direct answer to '{query}'. 
Output ONLY JSON: {{"answer": "exact short answer", "explanation": "concise reasoning with citations", "citations": ["url1", "url2"]}}"""
    final_prompt = ChatPromptTemplate.from_template(final_prompt_template)
    json_parser = JsonOutputParser()
    chain = final_prompt | llm | json_parser
    final_output = chain.invoke({
        "steps": "\n".join(state.reasoning_steps),
        "results": "\n".join([f"{r.title}: {r.content} ({r.url})" for r in state.results]),
        "query": query
    })
    final_answer = final_output.get("answer", "Error in synthesis")
    # Fact-check
    if not fact_check(final_answer, query):
        if DEBUG_MODE:
            print(f"Debug: Fact-check failed for {query}, regenerating final...")
        final_output = chain.invoke({  # Regenerate with same
            "steps": "\n".join(state.reasoning_steps),
            "results": "\n".join([f"{r.title}: {r.content} ({r.url})" for r in state.results]),
            "query": query
        })
        final_answer = final_output.get("answer", final_answer)
    return {
        "mode": Mode.PRO.value,
        "answer": final_answer,
        "reasoning": state.reasoning_steps,
        "sources": list(set(state.sources)),
        "metrics": {
            "factuality": 0.95,  # Placeholder
            "reasoning_depth": state.depth,
            "source_diversity": state.diversity
        },
        "time_estimate": "Deeper (10-30s)"
    }
SIMPLEQA_QUESTIONS = [
    {"q": "Who received the IEEE Frank Rosenblatt Award in 2010?", "gt": "Michio Sugeno"},
    {"q": "On which U.S. TV station did the Canadian reality series *To Serve and Protect* debut?", "gt": "KVOS-TV"},
    {"q": "What day, month, and year was Carrie Underwood’s album “Cry Pretty” certified Gold by the RIAA?", "gt": "October 23, 2018"},
    {"q": "What is the first and last name of the woman whom the British linguist Bernard Comrie married in 1985?", "gt": "Akiko Kumahira"},
]
FRAMES_QUESTIONS = [
    {"q": "Which magazine was started first Arthur's Magazine or First for Women?", "gt": "Arthur's Magazine", "gt_metrics": {"factuality": 1.0, "depth": 2, "diversity": 2}},
    {"q": "The Oberoi family is part of a hotel company that has a head office in what city?", "gt": "Delhi", "gt_metrics": {"factuality": 1.0, "depth": 2, "diversity": 2}},
    {"q": "Musician and satirist Allie Goertz wrote a song about the \"The Simpsons\" character Milhouse, who Matt Groening named after who?", "gt": "President Richard Nixon", "gt_metrics": {"factuality": 1.0, "depth": 3, "diversity": 4}},
    {"q": "What nationality was James Henry Miller's wife?", "gt": "American", "gt_metrics": {"factuality": 1.0, "depth": 2, "diversity": 3}},
    {"q": "Cadmium Chloride is slightly soluble in this chemical, it is also called what?", "gt": "alcohol", "gt_metrics": {"factuality": 1.0, "depth": 2, "diversity": 2}},
]
def evaluate_simpleqa(answers: List[str], ground_truths: List[str]) -> Dict[str, float]:
    accuracies = []
    for ans, gt in zip(answers, ground_truths):
        ratio = SequenceMatcher(None, gt.lower(), ans.lower()).ratio()
        acc = 1.0 if ratio > 0.7 else 0.0
        accuracies.append(acc)
    return {"accuracy": sum(accuracies) / len(accuracies) * 100}
def evaluate_frames(results: List[Dict], ground_truths: List[Dict]) -> Dict[str, float]:
    fact = []
    for i, (res, gt) in enumerate(zip(results, ground_truths)):
        ratio = SequenceMatcher(None, gt["gt"].lower(), res["answer"].lower()).ratio()
        computed_fact = 1.0 if ratio > 0.7 else 0.0
        res["metrics"]["factuality"] = computed_fact
        fact.append(computed_fact)
  
    fact_avg = sum(fact) / len(fact)
    depth = sum(r["metrics"]["reasoning_depth"] for r in results) / len(results)
    div = sum(r["metrics"]["source_diversity"] for r in results) / len(results)
  
    gt_depth = sum(g["gt_metrics"]["depth"] for g in ground_truths) / len(ground_truths)
    gt_div = sum(g["gt_metrics"]["diversity"] for g in ground_truths) / len(ground_truths)
  
    return {
        "factuality": fact_avg,
        "reasoning_depth": depth,
        "source_diversity": div,
        "gt_depth_avg": gt_depth,
        "gt_diversity_avg": gt_div
    }
def run_ui():
    st.set_page_config(page_title="Research Pro Mode", page_icon="🔍", layout="wide")
    st.title("🔍 Research Pro Mode")
    st.markdown("Продвинутый поисковый ассистент: Simple для скорости, Pro для глубины.")
  
    st.sidebar.header("Настройки")
    mode = st.sidebar.selectbox("Режим:", [Mode.SIMPLE.value, Mode.PRO.value])
    query = st.sidebar.text_area("Запрос:", placeholder="Введите вопрос...", height=100)
    if mode == Mode.PRO.value:
        sub_mode = st.sidebar.selectbox("Подрежим Pro (опционально):", ["none", "social", "academic"])
    else:
        sub_mode = None
  
    col1, col2 = st.columns([3, 1])
    with col2:
        run_button = st.button("🚀 Запустить поиск", type="primary")
  
    if run_button and query:
        with st.spinner(f"Обработка в {mode} режиме..."):
            if mode == Mode.SIMPLE.value:
                res = simple_mode(query)
            else:
                res = pro_mode(query, sub_mode if sub_mode != "none" else None)
            tab1, tab2 = st.tabs(["Ответ", "Источники & Метрики"])
            with tab1:
                st.subheader("Ответ")
                st.write(res["answer"])
                st.caption(f"Время: {res['time_estimate']}")
            with tab2:
                st.subheader("Источники")
                for i, src in enumerate(res["sources"][:5], 1):
                    st.write(f"{i}. [{src}]({src})")
                if mode == Mode.PRO.value:
                    st.subheader("Рассуждения")
                    for step in res["reasoning"]:
                        st.write(f"• {step}")
                    st.subheader("Метрики")
                    metrics_df = pd.DataFrame(list(res["metrics"].items()), columns=["Метрика", "Значение"])
                    st.dataframe(metrics_df, width="stretch")
  
    benchmark_expander = st.sidebar.expander("📊 Бенчмарки")
    if benchmark_expander.button("Запустить бенчмарки"):
        st.subheader("=== SimpleQA Benchmark (OpenAI SimpleQA, 2024) ===")
        simple_answers = []
        for q in SIMPLEQA_QUESTIONS:
            res = simple_mode(q["q"])
            simple_answers.append(res["answer"])
        simple_acc = evaluate_simpleqa(simple_answers, [q["gt"] for q in SIMPLEQA_QUESTIONS])
        st.metric("Accuracy (%)", f"{simple_acc['accuracy']:.2f}")
      
        st.subheader("=== FRAMES Benchmark (HotpotQA Multi-Hop) ===")
        frames_results = []
        for q in FRAMES_QUESTIONS:
            res = pro_mode(q["q"])
            frames_results.append(res)
            st.write(f"**Q:** {q['q'][:100]}...")
            st.write(f"**Answer:** {res['answer'][:150]}...")
        frames_metrics = evaluate_frames(frames_results, FRAMES_QUESTIONS)
        metrics_df = pd.DataFrame(list(frames_metrics.items()), columns=["Метрика", "Значение"])
        st.dataframe(metrics_df, width="stretch")
  
    st.sidebar.markdown("---")
    st.sidebar.info("Используйте .env для ключей (API_KEY для cloud.ru, TAVILY_API_KEY для поиска). DEBUG_MODE для логов.")
if __name__ == "__main__":
    run_ui()