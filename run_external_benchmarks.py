import json
import os
from functools import partial
import pandas as pd
import random
from difflib import SequenceMatcher
from langchain_core.messages import HumanMessage
from typing import List, Dict
from core.nodes import *
from core.structure import create_graph
from core.llm_connector import get_llm_client
from core.state import State

from dotenv import load_dotenv

MAX_RESULTS = int(os.getenv("MAX_RESULTS", "5"))
API_KEY = os.getenv("API_KEY")
MODEL = os.environ.get("BASE_MODEL")
MODEL_SUPERVISOR = os.environ.get("BASE_MODEL")
load_dotenv()

SIMPLEQA_DATASET_URL = "https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv"


def load_simpleqa_dataset(num_samples: int = None, seed: int = 42) -> List[Dict[str, str]]:
    print(f"Загружаю SimpleQA датасет из {SIMPLEQA_DATASET_URL}...")
    try:
        df = pd.read_csv(SIMPLEQA_DATASET_URL)
        print(f"   → Всего строк: {len(df)}")

        if 'problem' not in df.columns or 'answer' not in df.columns:
            raise ValueError(f"Ожидались 'problem' и 'answer', найдено: {list(df.columns)}")

        examples = [{"q": row["problem"], "gt": row["answer"]} for _, row in df.iterrows()]

        if num_samples:
            random.seed(seed)
            examples = random.sample(examples, min(num_samples, len(examples)))

        print(f"   → Выбрано {len(examples)} вопросов.")
        return examples

    except Exception as e:
        print(f"Ошибка загрузки: {e}")
        raise


def evaluate_external_simpleqa(answers: List[str], ground_truths: List[str]) -> Dict[str, float]:
    accuracies = []
    details = []
    for ans, gt in zip(answers, ground_truths):
        ans_norm = str(ans).lower().strip()
        gt_norm = str(gt).lower().strip()
        ratio = SequenceMatcher(None, ans_norm, gt_norm).ratio()
        acc = 1.0 if ratio > 0.7 else 0.0
        accuracies.append(acc)
        details.append({
            "ans": ans,
            "gt": gt,
            "ratio": ratio,
            "correct": bool(acc)
        })
    
    avg_acc = sum(accuracies) / len(accuracies) * 100
    return {
        "external_accuracy": avg_acc,
        "details": details
    }


def load_cache(mode_str: str) -> Dict:
    cache_file = f"cache_{mode_str}.json"
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(mode_str: str, cache: Dict):
    cache_file = f"cache_{mode_str}.json"
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def run_simpleqa_external(app, num_samples: int = 100, use_pro: bool = False):
    mode_str = "pro" if use_pro else "simple"
    print("\n" + "="*70)
    print(f"RUNNING OPENAI SIMPLEQA BENCHMARK ({mode_str.upper()} MODE)")
    print(f"Сэмпл: {num_samples} вопросов | Кэш: cache_{mode_str}.json")
    print("="*70)

    examples = load_simpleqa_dataset(num_samples)
    cache = load_cache(mode_str)
    answers = []
    new_computed = 0
    

    for i, item in enumerate(examples):
        q_hash = str(hash(item["q"]))
        if q_hash in cache:
            ans = cache[q_hash]
            print(f"\n[{i+1}/{len(examples)}] Q: {item['q'][:80]}... | GT: {item['gt']}")
            print(f"   → Answer (cached): {ans}")
            answers.append(ans)
            continue

        print(f"\n[{i+1}/{len(examples)}] Q: {item['q'][:80]}... | GT: {item['gt']}")
        
        try:
            msg: State = {
                "messages": [HumanMessage(content=item["q"])],
                "plan": None,
                "draft": None,
                "validated": None,
                "summary": None,
                "validation_fail_count": 0,
                "mode": 'pro' if use_pro else 'simple',
                "print_to": None,
                "thoughts": []
            }
            if use_pro:
                result = app.invoke(msg)
            else:
                result = app.invoke(msg) 
            ans = result["summary"]
            print(f"   → Answer: {ans}")
            
            cache[q_hash] = ans
            new_computed += 1
        except Exception as e:
            print(f"   Ошибка: {e}")
            ans = ""
        
        answers.append(ans)

        if (i + 1) % 10 == 0:
            print(f"   Прогресс: {i+1}/{len(examples)} (новых: {new_computed})")
            save_cache(mode_str, cache)

    save_cache(mode_str, cache)
    print(f"\n   Кэш сохранён: {new_computed} новых вычислений.")

    metrics = evaluate_external_simpleqa(answers, [ex["gt"] for ex in examples])
    errors = [d for d in metrics["details"] if not d["correct"]][:5]

    print("\n" + "—" * 50)
    print(f"SIMPLEQA ACCURACY: {metrics['external_accuracy']:.2f}%")
    print("—" * 50)
    if errors:
        print("Топ-5 ошибок:")
        for err in errors:
            print(f"   Ans: {err['ans'][:30]} | GT: {err['gt']} (ratio: {err['ratio']:.2f})")

    return {"metrics": metrics, "errors": errors}


def main():
    SIMPLE_N = 40
    PRO_N = 20

    print(f"Запуск бенчмарка: Simple Mode ({SIMPLE_N} вопросов), Pro Mode ({PRO_N} вопросов)")
    print("Модель: Qwen3-Next-80B | Поиск: Tavily\n")

    llm_nodes = {
        'analyzer': partial(analyzer_node, get_llm_client(MODEL), temperature=0.15),
        'planner': partial(planner_node, get_llm_client(MODEL, temperature=0.07)),
        'supervisor': partial(supervisor_node, get_llm_client(MODEL, temperature=0.05)),
        'validator': partial(validator_node, get_llm_client(MODEL, temperature=0)),
        'summarizer': partial(summarizer_node, get_llm_client(MODEL, temperature=0.1))
    }
    app = create_graph(**llm_nodes).compile()

    

    simple_results = run_simpleqa_external(app, num_samples=SIMPLE_N, use_pro=False)
    pro_results = run_simpleqa_external(app, num_samples=PRO_N, use_pro=True)

    # Итог
    print("\n" + "="*70)
    print(f"ИТОГОВАЯ СВОДКА (n={SIMPLE_N} + {PRO_N})")
    print("="*70)
    print(f"Simple Mode ({SIMPLE_N} q): {simple_results['metrics']['external_accuracy']:.2f}%")
    print(f"Pro Mode ({PRO_N} q):     {pro_results['metrics']['external_accuracy']:.2f}%")
    print(f"Цель (o1):                42.6%")

    # Сохранение
    summary = {
        "simple_n": SIMPLE_N,
        "pro_n": PRO_N,
        "simple_accuracy": simple_results['metrics']['external_accuracy'],
        "pro_accuracy": pro_results['metrics']['external_accuracy'],
        "simple_errors": simple_results['errors'],
        "pro_errors": pro_results['errors'],
        "note": "Fuzzy match >0.7 | Кэш: cache_simple.json / cache_pro.json"
    }
    filename = f"simpleqa_external_results_{SIMPLE_N}_{PRO_N}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Результаты сохранены в {filename}")


if __name__ == "__main__":
    main()