from __future__ import annotations

import argparse
import contextlib
import json
import multiprocessing
import os
import platform
import re
from functools import partial
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .api import _is_inaccessible, call_model

prompt_template = """你会看到一个参考资料和一些statement，请你判断对于参考资料来说statement是supported、unsupported、或者unknown，注意：
首先判断参考资料是否存在有效内容，如果参考资料中没有任何有效信息，如\"page not found\"页面，则认为所有statement的状态都是unknown。
除此之外，参考资料有效的情况下，对于一个statement来说，如果它包含的事实或数据在参考资料中可以全部或部分找到，就认为它是supported的（数据接受四舍五入）；如果statement中所有的事实和数据在参考资料中都找不到，认为它是unsupported的。

你应该返回json列表格式，列表中的每一项包含statement的序号和判断结果，例如：
[
    {{
        \"idx\": 1,
        \"result\": \"supported\"
    }},
    {{
        \"idx\": 2,
        \"result\": \"unsupported\"
    }}
]

下面是参考资料和statements：
<reference>
{reference}
</reference>

<statements>
{statements}
</statements>

下面开始判断，直接输出json列表，不要输出任何闲聊或解释。"""

prompt_template_en = """You will be provided with a reference and some statements. Please determine whether each statement is 'supported', 'unsupported', or 'unknown' with respect to the reference. Please note:
First, assess whether the reference contains any valid content. If the reference contains no valid information, such as a 'page not found' message, then all statements should be considered 'unknown'.
If the reference is valid, for a given statement: if the facts or data it contains can be found entirely or partially within the reference, it is considered 'supported' (data accepts rounding); if all facts and data in the statement cannot be found in the reference, it is considered 'unsupported'.

You should return the result in a JSON list format, where each item in the list contains the statement's index and the judgment result, for example:
[
    {{
        \"idx\": 1,
        \"result\": \"supported\"
    }},
    {{
        \"idx\": 2,
        \"result\": \"unsupported\"
    }}
]

Below are the reference and statements:
<reference>
{reference}
</reference>

<statements>
{statements}
</statements>

Begin the assessment now. Output only the JSON list, without any conversational text or explanations."""


_ALLOWED_RESULTS = {"supported", "unsupported", "unknown"}
_REFERENCE_MAX_CHARS = 6000
_WORKER_ENV = "BENCH2_VALIDATE_TOTAL_PROCESS"


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _unknown_results(facts: list[str]) -> list[dict[str, Any]]:
    return [{"idx": i, "result": "unknown"} for i in range(len(facts))]


def _coerce_results(response: str, facts: list[str]) -> list[dict[str, Any]]:
    fallback = _unknown_results(facts)
    try:
        payload = json.loads(response.replace("```json", "").replace("```", ""))
    except Exception:
        return fallback
    if not isinstance(payload, list):
        return fallback
    if len(payload) != len(facts):
        return fallback

    normalized: list[dict[str, Any]] = []
    for expected_idx, item in enumerate(payload):
        if not isinstance(item, dict):
            return fallback
        result = item.get("result")
        if result not in _ALLOWED_RESULTS:
            result = "unknown"
        idx = item.get("idx")
        if isinstance(idx, int):
            idx -= 1
        else:
            idx = expected_idx
        if idx != expected_idx:
            idx = expected_idx
        normalized.append({"idx": idx, "result": result})
    return normalized


def _build_prompt(ref: str, facts: list[str], lang: str) -> str:
    facts_str = "\n".join(f"{i + 1}. {fact}" for i, fact in enumerate(facts))
    if lang == "zh":
        return prompt_template.format(reference=ref, statements=facts_str)
    if lang == "en":
        return prompt_template_en.format(reference=ref, statements=facts_str)
    raise ValueError(f"Unsupported language: {lang}")


def _normalize_reference(ref: str) -> str:
    compact = re.sub(r"\s+", " ", ref).strip()
    return compact[:_REFERENCE_MAX_CHARS]


def _resolve_worker_count(n_total_process: int) -> int:
    if n_total_process > 0:
        return n_total_process
    env_value = os.getenv(_WORKER_ENV, "").strip()
    if not env_value:
        return 1
    try:
        return max(1, int(env_value))
    except ValueError:
        return 1


def validate(
    data: tuple[str, dict[str, Any]], id_to_lang_map: dict[str, str | None]
) -> dict[str, Any]:
    url = data[0]
    payload = data[1]
    ref = payload.get("url_content")
    facts = list(payload.get("facts") or [])
    article_id = payload.get("article_id")
    fallback = _unknown_results(facts)

    if not ref or not article_id:
        return {"url": url, "validate_res": fallback, "error": None}
    if not facts:
        return {"url": url, "validate_res": fallback, "error": None}
    if _is_inaccessible(ref):
        return {"url": url, "validate_res": fallback, "error": None}

    lang = id_to_lang_map.get(article_id)
    if lang not in {"zh", "en"}:
        return {"url": url, "validate_res": fallback, "error": None}

    try:
        response = call_model(_build_prompt(_normalize_reference(ref), facts, lang))
    except Exception:
        return {"url": url, "validate_res": fallback, "error": None}

    return {"url": url, "validate_res": _coerce_results(response, facts), "error": None}


def run_validation(
    *,
    raw_data_path: str | Path,
    output_path: str | Path,
    query_data_path: str | Path,
    n_total_process: int = 1,
) -> None:
    output = Path(output_path)
    raw_data = _load_jsonl(raw_data_path)
    query_data = _load_jsonl(query_data_path)
    id_to_lang_map = {item["id"]: item.get("language") for item in query_data if "id" in item}
    if not id_to_lang_map:
        raise ValueError("No valid language information found in query data")

    worker_count = _resolve_worker_count(n_total_process)
    if output.exists():
        processed = {d["id"] for d in _load_jsonl(output) if "id" in d}
        data_to_process = [d for d in raw_data if d.get("id") not in processed]
    else:
        data_to_process = raw_data

    print(f"Processing {len(data_to_process)} instances...", flush=True)

    for entry in tqdm(data_to_process):
        citations = [(k, v) for k, v in entry["citations_deduped"].items()]
        article_id = entry.get("id")
        if not article_id:
            continue

        for citation in citations:
            citation[1]["article_id"] = article_id

        if worker_count == 1:
            results = []
            total = len(citations)
            for index, citation in enumerate(citations, start=1):
                print(f"validate article={article_id} citation={index}/{total}", flush=True)
                results.append(validate(citation, id_to_lang_map))
        else:
            run_partial = partial(validate, id_to_lang_map=id_to_lang_map)
            with multiprocessing.Pool(processes=worker_count) as pool:
                results = pool.map(run_partial, citations)

        for res in results:
            entry["citations_deduped"][res["url"]]["validate_res"] = res["validate_res"]
            entry["citations_deduped"][res["url"]]["validate_error"] = res["error"]

        with output.open("a+", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    if platform.system() == "Darwin":
        with contextlib.suppress(RuntimeError):
            multiprocessing.set_start_method("spawn")
    else:
        with contextlib.suppress(RuntimeError):
            multiprocessing.set_start_method("fork")

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--raw_data_path", type=str, required=True)
    parser.add_argument("--query_data_path", type=str, required=True)
    parser.add_argument("--n_total_process", type=int, default=1)
    args = parser.parse_args()

    run_validation(
        raw_data_path=args.raw_data_path,
        output_path=args.output_path,
        query_data_path=args.query_data_path,
        n_total_process=args.n_total_process,
    )
