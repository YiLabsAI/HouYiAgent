"""HouYi compat override for deep_research_bench utils/extract.py.

Adds ``clean_fact`` post-processing after ``remove_urls`` to strip residual
URL fragments that the LLM occasionally embeds at the boundaries of extracted
facts (e.g. facts starting with ``html)...`` or ending with ``...https://www``).

These fragments arise when the report contains back-to-back Markdown links and
the LLM's sentence-boundary detection splits mid-URL.  The residual text is
meaningless to the validate LLM and causes the ``assert len(validate_res) ==
len(facts)`` check to fail (the validate LLM cannot judge a half-sentence),
which wastes all three retries and leaves ``validate_res=[]``.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import re
from functools import partial

prompt_template = """你会看到一篇研究报告，研究报告正文中会有一些对参考文献的引用。
正文中的引用可能以如下形式出现：
1. 一段文字+空格+数字，例如："李强基于收入、教育和职业构造了一个社会经济地位指数（SES），将社会划分为7个等级 15"
2. 一段文字+[（一个或多个)数字]，例如："李强基于收入、教育和职业构造了一个社会经济地位指数（SES），将社会划分为7个等级[15]"
3. 一段文字+[（一个或多个)数字†(一些行号等内容)]，例如："李强基于收入、教育和职业构造了一个社会经济地位指数（SES），将社会划分为7个等级[15†L10][5L23][7†summary][9summary]"
4. [引用来源](引用链接)，例如："根据[ChinaFile: A Guide to Social Class in Modern China](https://www.chinafile.com/reporting-opinion/media/guide-social-class-modern-china)'s分类，中国社会可分为九个阶层"

请从正文中找出**所有**引用了参考文献的地方，提取出(fact, ref_idx, url)三元组，提取的时候，注意以下事项：
1. 由于后续需要检验这些facts是否正确，你可能需要在引用的前后寻找一些上下文，以确保fact是完整可理解的，而不是简单的词组或短语
2. 如果一个fact引用了多个文献，那么它应该对应多个三元组，例如如果引用了2个文献，则应该是(fact, ref_idx_1, url_1)和(fact, ref_idx_2, url_2)
3. 对于第三种形式的引用，ref_idx仅考虑第一个数字部分，不考虑其他指示具体位置的内容；对于第四种形式的引用（即引用来源和链接直接出现在正文中）的情况，ref_idx统一设置为0
4. 如果正文中没有标出引用的具体位置（比如仅在文章结尾列出了参考文献列表，而没有在正文中标出），请返回空列表

你应该返回json列表格式，列表中的每一项是一个三元组，例如：
[
    {{
        "fact": "原文中的文本片段，注意中文引号要用全角, 英文引号前加单个反斜杠转义",
        "ref_idx": "该段文字引用的参考文献在参考文献列表中的索引",
        "url": "该段文字引用的参考文献链接（从研究报告结尾的参考文献列表或引用处的括号中提取）"
    }}
]

下面是研究报告的正文：
{report_text}

下面开始提取，直接输出json列表，不要输出任何闲聊或解释。"""


prompt_template_en = """You will be provided with a research report. The body of the report will contain some citations to references.

Citations in the main text may appear in the following forms:
1. A segment of text + space + number, for example: "Li Qiang constructed a socioeconomic status index (SES) based on income, education, and occupation, dividing society into 7 levels 15"
2. A segment of text + [number], for example: "Li Qiang constructed a socioeconomic status index (SES) based on income, education, and occupation, dividing society into 7 levels[15]"
3. A segment of text + [number†(some line numbers, etc.)], for example: "Li Qiang constructed a socioeconomic status index (SES) based on income, education, and occupation, dividing society into 7 levels[15†L10][5L23][7†summary]"
4. [Citation Source](Citation Link), for example: "According to [ChinaFile: A Guide to Social Class in Modern China](https://www.chinafile.com/reporting-opinion/media/guide-social-class-modern-china)'s classification, Chinese society can be divided into nine strata"

Please identify **all** instances where references are cited in the main text, and extract (fact, ref_idx, url) triplets. When extracting, pay attention to the following:
1. Since these facts will need to be verified later, you may need to look for some context before and after the citation to ensure that the fact is complete and understandable, rather than just a simple phrase or short expression.
2. If a fact cites multiple references, then it should correspond to two triplets: (fact, ref_idx_1, url_1) and (fact, ref_idx_2, url_2).
3. For the third form of citation (i.e., where the citation source and link appear directly in the text), the ref_idx should be uniformly set to 0.
4. If the main text does not specify the exact location of the citation (for example, only the reference list is listed at the end of the article, without specifying the citation point in the text), please return an empty list.

You should return a JSON list format, where each item in the list is a triplet, for example:
[
    {{
        "fact": "Text segment from the original document. Note that Chinese quotation marks should use full-width marks. And add a single backslash before the English quotation mark to make it a readable for python json module.",
        "ref_idx": "The index of the cited reference in the reference list for this text segment.",
        "url": "The URL of the cited reference for this text segment (extracted from the reference list at the end of the research report or from the parentheses at the citation point)."
    }}
]

Here is the main text of the research report:
{report_text}

Please begin the extraction now. Output only the JSON list directly, without any chitchat or explanations."""


_URL_TAIL_RE = re.compile(
    r"^"
    r"(?:"
    r"https?://[^\s)]*\)?"
    r"|[a-z][a-z0-9./_%-]*\)"
    r")"
    r"\s*",
    re.IGNORECASE,
)
_PURE_PATH_RE = re.compile(r"^[a-z][a-z0-9./_%-]{10,}$", re.IGNORECASE)
_REF_TOKEN_RE = re.compile(r"\[ref_[0-9a-f]+\]")
_URL_SCHEME_TAIL_RE = re.compile(r"\s*\[?https?://\S*$")
_MIN_FACT_CHARS = 15


def clean_urls(input_text: str) -> str:
    pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")

    def repl(match: re.Match) -> str:
        title = match.group(1)
        url = match.group(2)
        cut_idx = url.find("#:~:text=")
        if cut_idx != -1:
            url = url[:cut_idx]
        return f"[{title}]({url})"

    return pattern.sub(repl, input_text)


def remove_urls(input_text: str) -> str:
    pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
    return pattern.sub(r"[\1]", input_text)


def clean_escape(input_text: str) -> str:
    input_text = input_text.replace("\\>", ">")
    input_text = input_text.replace("\\<", "<")
    input_text = input_text.replace("\\+", "+")
    input_text = input_text.replace("\\~", "~")
    return input_text


def clean_fact(fact: str) -> str:
    """Remove residual URL fragments left at the start or end of a fact.

    When a HouYi report uses back-to-back Markdown links the LLM extractor
    sometimes splits a fact at an URL boundary, yielding strings like:

    * ``"html)。尽管整体收入水平..."``         – URL tail at the start
    * ``"com/news/.../)，但..."``              – URL path fragment at start
    * ``"...收入为43377元 [title](https://www"`` – truncated URL at the end
    * ``"[ref_5cfbcf88]。..."``               – bare ref token at start
    * ``"org/core/.../F1A646014C8B1E"``       – pure URL path, no content

    This function strips those fragments so the validate LLM receives a clean,
    assessable statement.
    """
    fact = fact.strip()

    # Strip leading bare ref tokens like [ref_abc123]
    fact = _REF_TOKEN_RE.sub("", fact).strip()

    # Strip leading URL fragment: full URL or path segment ending with ')'
    fact = _URL_TAIL_RE.sub("", fact).strip()

    # Strip leading punctuation that now has nothing before it
    fact = re.sub(r"^[)）。，,、；;：:\s]+", "", fact).strip()

    # Strip trailing truncated URL (no closing paren) e.g. "...[title](https://www"
    fact = _URL_SCHEME_TAIL_RE.sub("", fact).rstrip()
    # Strip trailing open [title with no closing paren
    fact = re.sub(r"\s*\[[^\]]*$", "", fact).strip()

    # Discard if what remains is a pure ASCII URL path with no natural language
    if _PURE_PATH_RE.match(fact):
        return ""

    return fact


def _parse_and_write(d: dict, response: str, output_path: str) -> bool:
    for _ in range(3):
        try:
            if response != "":
                response = response.replace("```json", "").replace("```", "")
                response = clean_escape(response)
                d["citations"] = json.loads(response)
                for c in d["citations"]:
                    c["fact"] = clean_fact(remove_urls(c["fact"]))
                d["citations"] = [
                    c for c in d["citations"] if len(c.get("fact", "")) >= _MIN_FACT_CHARS
                ]
            else:
                d["citations"] = "extraction failed"
            with open(output_path, "a+", encoding="utf-8") as f:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
            print(f">>>>>>>>>> generating {d['id']}-th instance...")
            return True
        except Exception as e:
            print(repr(e))
    return False


def run(data, output_path, id_to_lang_map):
    from .api import call_model as _call_model

    for i, d in enumerate(data):
        article_id = d.get("id")

        if not article_id:
            print("Article has no ID field, skipping extraction")
            continue

        if article_id not in id_to_lang_map:
            print(f"Language not found for article ID: {article_id}")
            continue

        lang = id_to_lang_map[article_id]
        if lang not in ["zh", "en"]:
            print(f"Unsupported language: {lang} for article ID: {article_id}")
            continue

        if lang == "zh":
            user_prompt = prompt_template.format(report_text=d["article"])
        else:
            user_prompt = prompt_template_en.format(report_text=d["article"])

        response = _call_model(user_prompt)
        if not _parse_and_write(d, response, output_path):
            print(
                f">>>>>>>>>> All attempts failed, article ID: {article_id}, cannot extract citations"
            )


if __name__ == "__main__":
    from .io_utils import load_jsonl  # bench2 runtime only

    multiprocessing.set_start_method("fork")

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--raw_data_path", type=str, required=True)
    parser.add_argument("--query_data_path", type=str, required=True)
    parser.add_argument("--n_total_process", type=int, default=1)
    args = parser.parse_args()

    output_path = args.output_path

    query_data = load_jsonl(args.query_data_path)
    id_to_lang_map = {
        item["id"]: item.get("language")
        for item in query_data
        if "id" in item and "language" in item
    }

    if not id_to_lang_map:
        raise ValueError("No valid language information found in query data")

    raw_data = load_jsonl(args.raw_data_path)

    if os.path.exists(output_path):
        processed = [d["id"] for d in load_jsonl(output_path)]
        data_to_process = [d for d in raw_data if d["id"] not in processed]
    else:
        data_to_process = raw_data

    if "openai" in args.raw_data_path:
        for d in data_to_process:
            d["article"] = clean_urls(d["article"])

    print(f"Processing {len(data_to_process)} instances...")

    n_total_process = args.n_total_process
    if n_total_process == 1:
        run(data_to_process, output_path, id_to_lang_map)
    elif n_total_process > 1:
        part_size = (len(data_to_process) + n_total_process - 1) // n_total_process
        data_splits = [
            data_to_process[i * part_size : (i + 1) * part_size] for i in range(n_total_process)
        ]
        run_partial = partial(run, output_path=output_path, id_to_lang_map=id_to_lang_map)
        with multiprocessing.Pool(processes=n_total_process) as pool:
            pool.map(run_partial, data_splits)
