from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import threading

from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class CleaningTimeoutError(RuntimeError):
    pass


class CleaningChunkingRequiredError(RuntimeError):
    pass


class ArticleCleaner:
    def __init__(self, clean_agent):
        self.clean_agent = clean_agent
        self.min_valid_length = 100

    def _get_clean_prompt(self, language="zh"):
        from prompt.clean_prompt import clean_article_prompt_en, clean_article_prompt_zh

        return clean_article_prompt_zh if language == "zh" else clean_article_prompt_en

    def _is_valid_result(self, text):
        return text and len(text.strip()) >= self.min_valid_length

    def _update_progress(self, pbar, pbar_lock):
        if pbar and pbar_lock:
            with pbar_lock:
                pbar.update(1)

    def _is_token_limit_error(self, error):
        error_str = str(error).lower()
        return "tokens" in error_str and "less than" in error_str

    def _clean_text(self, text, language="zh", max_retries=3):
        clean_prompt = self._get_clean_prompt(language)
        user_prompt = clean_prompt.format(article=text)

        for retry in range(max_retries):
            try:
                result = self.clean_agent.generate(user_prompt=user_prompt, system_prompt="")
                if self._is_valid_result(result):
                    return result
                logger.warning(f"Invalid cleaning result, retry #{retry + 1}")
            except TimeoutError as error:
                logger.error("API call timeout: %s", error)
                raise CleaningTimeoutError(str(error)) from error
            except Exception as error:
                logger.error(f"API call error: {error}")
                if self._is_token_limit_error(error):
                    raise CleaningChunkingRequiredError(str(error)) from error

        return None

    def chunk_clean_article(self, article, language="zh"):
        logger.info("Attempting to process article in 2 chunks")
        chunks = []
        chunk_size = len(article) // 2

        for i in range(2):
            start = i * chunk_size
            end = len(article) if i == 1 else chunk_size
            if i == 0:
                search_start = max(0, end - 200)
                for j in range(end, search_start, -1):
                    if j < len(article) and article[j] in [".", "?", "!", "。", "？", "！", "\n"]:
                        end = j + 1
                        break
            chunks.append(article[start:end])

        cleaned_chunks = []
        for i, chunk in enumerate(chunks):
            try:
                clean_result = self._clean_text(chunk, language)
                if clean_result is None and len(chunk) > 200000:
                    logger.error(f"Chunk {i + 1} too large, cannot process")
                    return None
                cleaned_chunks.append(clean_result)
            except CleaningTimeoutError:
                return None
            except CleaningChunkingRequiredError:
                logger.error(f"Chunk {i + 1} too large, cannot process")
                return None
            except Exception as error:
                logger.error(f"Failed to clean chunk {i + 1}/2: {error}")
                return None

        logger.info("All chunks processed, merging results")
        return "".join(cleaned_chunks)

    def clean_single(
        self,
        item,
        output_file=None,
        processed_ids=None,
        file_lock=None,
        pbar_lock=None,
        pbar=None,
        max_retries=5,
        language="zh",
    ):
        if not self.clean_agent:
            logger.error("No clean_agent provided, cannot clean article")
            self._update_progress(pbar, pbar_lock)
            return None

        try:
            data = item.copy()
            item_id = data.get("id")
            prompt = data.get("prompt", "")
            article = data.get("article", "")
            if not item_id or not prompt or not article:
                self._update_progress(pbar, pbar_lock)
                return None
            if processed_ids is not None and item_id in processed_ids:
                self._update_progress(pbar, pbar_lock)
                return None

            try:
                cleaned_article = self._clean_text(article, language, max_retries)
            except CleaningChunkingRequiredError:
                logger.info(f"ID: {item_id} - Article may be too long, trying chunked processing")
                cleaned_article = self.chunk_clean_article(article, language=language)
            except CleaningTimeoutError:
                logger.warning(
                    f"ID: {item_id} - Cleaning timed out, falling back to original article"
                )
                cleaned_article = article

            if not self._is_valid_result(cleaned_article):
                logger.error(f"ID: {item_id} - Failed to clean article after {max_retries} retries")
                self._update_progress(pbar, pbar_lock)
                return {"id": item_id, "error": "Failed to clean article"}

            result = {"id": item_id, "prompt": prompt, "article": cleaned_article}
            if output_file and file_lock and processed_ids is not None:
                self._write_result_to_file(result, output_file, file_lock, processed_ids, item_id)
                self._update_progress(pbar, pbar_lock)
                return item_id
            self._update_progress(pbar, pbar_lock)
            return result
        except Exception as error:
            self._update_progress(pbar, pbar_lock)
            logger.error(f"Error cleaning article {item.get('id', 'unknown')}: {error}")
            return None

    def _write_result_to_file(self, result, output_file, file_lock, processed_ids, item_id):
        with file_lock:
            with open(output_file, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            processed_ids.add(item_id)

    def clean_articles(
        self,
        model,
        raw_data_dir,
        cleaned_data_dir,
        max_workers=5,
        max_retries=5,
        limit=None,
        language="en",
    ):
        os.makedirs(cleaned_data_dir, exist_ok=True)
        input_file = os.path.join(raw_data_dir, f"{model}.jsonl")
        output_file = os.path.join(cleaned_data_dir, f"{model}.jsonl")
        if not os.path.exists(input_file):
            logger.warning(f"Input file for model {model} not found: {input_file}")
            return

        logger.info(f"=== Cleaning {model} articles ===")
        all_items = self._load_items(input_file)
        if limit is not None and limit > 0:
            all_items = all_items[:limit]
        processed_ids = self._load_processed_ids(output_file)
        to_process = [item for item in all_items if item.get("id") not in processed_ids]
        logger.info(
            f"Total: {len(all_items)} items, {len(to_process)} to process, {len(processed_ids)} already processed"
        )
        if not to_process:
            logger.info("All items already processed, no further action needed")
            return

        file_lock = threading.Lock()
        pbar_lock = threading.Lock()
        with (
            tqdm(
                total=len(all_items), desc=f"Cleaning {model} articles", initial=len(processed_ids)
            ) as pbar,
            concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor,
        ):
            futures = [
                executor.submit(
                    self.clean_single,
                    item,
                    output_file,
                    processed_ids,
                    file_lock,
                    pbar_lock,
                    pbar,
                    max_retries=max_retries,
                    language=language,
                )
                for item in to_process
            ]
            processed_count = 0
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    processed_count += 1

        logger.info(
            f"=== {model} model cleaning complete, cleaned {processed_count} new articles, total of {len(processed_ids)} articles processed ==="
        )

    def _load_items(self, input_file):
        all_items = []
        with open(input_file, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    try:
                        all_items.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning(f"Error parsing JSON in input file, line: {line.strip()}")
        return all_items

    def _load_processed_ids(self, output_file):
        processed_ids = set()
        if os.path.exists(output_file):
            logger.info(f"Found existing output file: {output_file}")
            with open(output_file, encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            if "id" in data:
                                processed_ids.add(data["id"])
                        except json.JSONDecodeError:
                            logger.warning("Invalid JSON line in output file, skipped")
            logger.info(f"Read {len(processed_ids)} already processed records from output file")
        else:
            open(output_file, "w", encoding="utf-8").close()
            logger.info(f"Created new output file: {output_file}")
        return processed_ids
