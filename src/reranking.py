import ipaddress
import os
import threading
from functools import lru_cache
from typing import Optional
from urllib.parse import urlparse
from dotenv import load_dotenv
import requests
import src.prompts as prompts
from concurrent.futures import ThreadPoolExecutor
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.api_requests import APIProcessor
from src.embedding_backend import _bgem3_encode_texts, _load_bgem3_model


def _get_env_value(*names, default=None):
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


def _should_bypass_env_proxy_for_url(base_url: str | None) -> bool:
    if not base_url:
        return False

    hostname = urlparse(base_url).hostname
    if not hostname:
        return False

    if hostname in {"localhost", "127.0.0.1", "::1"}:
        return True

    try:
        ip = ipaddress.ip_address(hostname)
        return ip.is_private or ip.is_loopback or ip.is_link_local
    except ValueError:
        return hostname.endswith(".local")


def _normalize_remote_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    if all(0.0 <= float(score) <= 1.0 for score in scores):
        return [float(score) for score in scores]
    return _minmax_normalize([float(score) for score in scores])


def _parse_devices(value: str | None) -> list[str]:
    if not value:
        return ["cuda:0"]

    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        return ["cuda:0"]

    devices: list[str] = []
    for part in parts:
        if part.isdigit():
            devices.append(f"cuda:{part}")
        elif part.startswith(("cuda:", "cpu", "mps", "npu:", "musa:")):
            devices.append(part)
        else:
            devices.append(f"cuda:{part}")
    return devices


def _env_positive_int(*names: str) -> Optional[int]:
    for name in names:
        raw_value = os.getenv(name)
        if raw_value is None or str(raw_value).strip() == "":
            continue
        try:
            value = int(str(raw_value).strip())
        except ValueError:
            continue
        if value > 0:
            return value
    return None


def _limit_devices(devices: list[str], *env_names: str) -> list[str]:
    max_devices = _env_positive_int(*env_names)
    if max_devices is None:
        return devices
    return devices[: max(1, max_devices)]


def _minmax_normalize(values: list[float]) -> list[float]:
    if not values:
        return []

    min_value = min(values)
    max_value = max(values)
    if max_value == min_value:
        return [1.0 if max_value > 0 else 0.0 for _ in values]

    scale = max_value - min_value
    return [float((value - min_value) / scale) for value in values]


_flag_reranker_load_lock = threading.Lock()


class FastTokenizerFlagRerankerModel:
    """Local reranker that keeps FlagEmbedding's model choice but uses the
    fast-tokenizer __call__ path instead of prepare_for_model + pad."""

    def __init__(
        self,
        model_name: str,
        device: str,
        use_fp16: bool,
        batch_size: int = 128,
        max_length: int = 512,
        trust_remote_code: bool = False,
    ):
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=False,
        )
        if use_fp16 and device.startswith("cuda"):
            self.model = self.model.half()
        self.model.to(device)
        self.model.eval()

    @staticmethod
    def _normalize_score(score: float) -> float:
        return float(torch.sigmoid(torch.tensor(score)).item())

    @torch.no_grad()
    def compute_score(self, sentence_pairs, normalize: bool = False):
        if not sentence_pairs:
            return []

        if isinstance(sentence_pairs[0], str):
            sentence_pairs = [sentence_pairs]

        indexed_pairs = list(enumerate(sentence_pairs))
        indexed_pairs.sort(key=lambda item: len(item[1][0]) + len(item[1][1]), reverse=True)
        scores = [0.0] * len(indexed_pairs)

        batch_size = self.batch_size
        start_index = 0
        while start_index < len(indexed_pairs):
            current_batch_size = min(batch_size, len(indexed_pairs) - start_index)
            current_batch = indexed_pairs[start_index:start_index + current_batch_size]
            queries = [pair[0] for _, pair in current_batch]
            passages = [pair[1] for _, pair in current_batch]

            try:
                inputs = self.tokenizer(
                    queries,
                    passages,
                    padding=True,
                    truncation="only_second",
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                logits = self.model(**inputs, return_dict=True).logits.view(-1).float().cpu().tolist()
            except (RuntimeError, torch.cuda.OutOfMemoryError):
                if current_batch_size == 1:
                    raise
                if self.device.startswith("cuda"):
                    torch.cuda.empty_cache()
                batch_size = max(1, current_batch_size * 3 // 4)
                continue

            for (original_index, _), score in zip(current_batch, logits):
                scores[original_index] = self._normalize_score(score) if normalize else float(score)
            start_index += current_batch_size

        return scores


@lru_cache(maxsize=4)
def _load_flag_reranker_model_cached(
    model_name: str,
    device: str,
    use_fp16: bool,
    batch_size: int,
    max_length: int,
    trust_remote_code: bool,
) -> FastTokenizerFlagRerankerModel:
    return FastTokenizerFlagRerankerModel(
        model_name=model_name,
        device=device,
        use_fp16=use_fp16,
        batch_size=batch_size,
        max_length=max_length,
        trust_remote_code=trust_remote_code,
    )


def _load_flag_reranker_model(
    model_name: str,
    device: str,
    use_fp16: bool,
    batch_size: int = 128,
    max_length: int = 512,
    trust_remote_code: bool = False,
) -> FastTokenizerFlagRerankerModel:
    # Serialise concurrent model initialisation so transformers does not race
    # through meta-tensor loading on first use under parallel question runs.
    with _flag_reranker_load_lock:
        return _load_flag_reranker_model_cached(
            model_name=model_name,
            device=device,
            use_fp16=use_fp16,
            batch_size=batch_size,
            max_length=max_length,
            trust_remote_code=trust_remote_code,
        )

class JinaReranker:
    def __init__(self):
        self.url = 'https://api.jina.ai/v1/rerank'
        self.headers = self.get_headers()
        
    def get_headers(self):
        load_dotenv()
        jina_api_key = os.getenv("JINA_API_KEY")    
        headers = {'Content-Type': 'application/json',
                   'Authorization': f'Bearer {jina_api_key}'}
        return headers
    
    def rerank(self, query, documents, top_n = 10):
        data = {
            "model": "jina-reranker-v2-base-multilingual",
            "query": query,
            "top_n": top_n,
            "documents": documents
        }

        response = requests.post(url=self.url, headers=self.headers, json=data)

        return response.json()
    
class FlagEmbeddingReranker:
    def __init__(self, model: str | None = None):
        load_dotenv()
        self.model_name = model or _get_env_value("RERANKING_MODEL", default="BAAI/bge-reranker-v2-m3")
        self.device = _get_env_value("RERANKING_DEVICE", default="cuda:0")
        self.devices = _limit_devices(_parse_devices(self.device), "RERANKING_MAX_DEVICES")
        use_fp16 = _get_env_value("RERANKING_USE_FP16", default="true").lower() in {"1", "true", "yes", "on"}
        trust_remote_code = _get_env_value("RERANKING_TRUST_REMOTE_CODE", default="false").lower() in {
            "1", "true", "yes", "on"
        }
        self.use_fp16 = use_fp16 and any(device.startswith("cuda") for device in self.devices)
        self.trust_remote_code = trust_remote_code
        self._fp16_runtime_disabled = False
        self.models = self._load_models(self.use_fp16)

    def _load_models(self, use_fp16: bool):
        # Keep one model per target device and shard work here. This also avoids
        # FlagEmbedding's prepare_for_model + tokenizer.pad path that triggers
        # the XLMRobertaTokenizerFast warning.
        return [
            _load_flag_reranker_model(
                model_name=self.model_name,
                device=device,
                use_fp16=use_fp16,
                trust_remote_code=self.trust_remote_code,
            )
            for device in self.devices
        ]

    @staticmethod
    def _is_fp16_cublas_failure(exc: Exception) -> bool:
        message = str(exc or "").lower()
        return "cublas_status_invalid_value" in message or "cublasgemmex" in message

    def _retry_with_fp32(self) -> None:
        if not self.use_fp16 or self._fp16_runtime_disabled:
            raise RuntimeError("fp16_fallback_unavailable")
        print(
            "Warning: FlagEmbedding reranker hit a CUDA fp16 GEMM failure. "
            "Reloading reranker in fp32 and retrying once."
        )
        self.models = self._load_models(False)
        self._fp16_runtime_disabled = True

    @staticmethod
    def _normalize_scores(scores):
        if isinstance(scores, (float, int)):
            return [float(scores)]
        return [float(score) for score in scores]

    def _compute_scores(self, pairs: list[list[str]]) -> list[float]:
        if len(self.models) == 1:
            return self._normalize_scores(self.models[0].compute_score(pairs, normalize=True))

        indexed_pairs = list(enumerate(pairs))
        chunks = [indexed_pairs[i::len(self.models)] for i in range(len(self.models))]

        def score_chunk(model, chunk):
            if not chunk:
                return []
            indices = [index for index, _ in chunk]
            chunk_pairs = [pair for _, pair in chunk]
            chunk_scores = self._normalize_scores(model.compute_score(chunk_pairs, normalize=True))
            return list(zip(indices, chunk_scores))

        indexed_scores = []
        with ThreadPoolExecutor(max_workers=len(self.models)) as executor:
            futures = [
                executor.submit(score_chunk, model, chunk)
                for model, chunk in zip(self.models, chunks)
            ]
            for future in futures:
                indexed_scores.extend(future.result())

        scores = [0.0] * len(pairs)
        for index, score in indexed_scores:
            scores[index] = score
        return scores

    def rerank_documents(self, query: str, documents: list, documents_batch_size: int = 4, llm_weight: float = 0.7):
        pairs = [[query, doc["text"]] for doc in documents]
        try:
            scores = self._compute_scores(pairs)
        except RuntimeError as exc:
            if self._is_fp16_cublas_failure(exc) and self.use_fp16 and not self._fp16_runtime_disabled:
                self._retry_with_fp32()
                scores = self._compute_scores(pairs)
            else:
                raise

        vector_weight = 1 - llm_weight
        results = []
        for doc, score in zip(documents, scores):
            item = doc.copy()
            item["relevance_score"] = round(float(score), 4)
            item["final_relevance_score"] = item["relevance_score"]
            item["combined_score"] = round(
                llm_weight * item["relevance_score"] + vector_weight * item["distance"], 4
            )
            results.append(item)

        results.sort(key=lambda x: x["combined_score"], reverse=True)
        self.last_debug = {
            "reranking_strategy": "single",
            "initial_candidate_pool_size": len(documents),
            "colbert_candidate_pool_size": None,
            "colbert_top_n": None,
            "final_reranking_backend": "flag_embedding",
        }
        return results


class VLLMApiReranker:
    def __init__(self, model: str | None = None):
        load_dotenv()
        self.model_name = (
            model
            or _get_env_value("RERANKING_MODEL", "QWEN_RERANKING_MODEL", "QWEN_MODEL", "LLM_MODEL")
            or "Qwen/Qwen3-Reranker-0.6B"
        )
        self.base_url = _get_env_value("RERANKING_BASE_URL", "QWEN_RERANKING_BASE_URL", "QWEN_BASE_URL", "LLM_BASE_URL")
        self.api_key = _get_env_value("RERANKING_API_KEY", "QWEN_RERANKING_API_KEY", "QWEN_API_KEY", "LLM_API_KEY")
        timeout_raw = _get_env_value("RERANKING_TIMEOUT", "RERANKING_API_TIMEOUT", default="300")
        self.timeout = float(timeout_raw) if timeout_raw else 300.0

    def _get_request_url(self) -> str:
        if not self.base_url:
            raise ValueError(
                "Missing reranking API base URL. Set RERANKING_BASE_URL "
                "(or fall back to QWEN_BASE_URL / LLM_BASE_URL)."
            )

        normalized_base = self.base_url.rstrip("/")
        if normalized_base.endswith("/chat/completions"):
            normalized_base = normalized_base[: -len("/chat/completions")]
        if normalized_base.endswith(("/rerank", "/v1/rerank", "/v2/rerank")):
            return normalized_base
        if normalized_base.endswith("/v1") or normalized_base.endswith("/v2"):
            return f"{normalized_base}/rerank"
        return f"{normalized_base}/v1/rerank"

    def _post(self, payload: dict) -> requests.Response:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        request_kwargs = {
            "url": self._get_request_url(),
            "headers": headers,
            "json": payload,
            "timeout": self.timeout,
        }

        if _should_bypass_env_proxy_for_url(self.base_url):
            with requests.Session() as session:
                session.trust_env = False
                return session.post(**request_kwargs)
        return requests.post(**request_kwargs)

    @staticmethod
    def _extract_scores(response_json: dict, total_documents: int) -> list[float]:
        results = response_json.get("results")
        if results is None:
            results = response_json.get("data")
        if not isinstance(results, list):
            raise ValueError(f"Unexpected rerank response payload: {response_json!r}")

        raw_scores = [0.0] * total_documents
        for item in results:
            if not isinstance(item, dict):
                continue
            index = item.get("index")
            score = item.get("relevance_score", item.get("score"))
            if index is None or score is None:
                continue
            try:
                normalized_index = int(index)
            except (TypeError, ValueError):
                continue
            if 0 <= normalized_index < total_documents:
                raw_scores[normalized_index] = float(score)

        return _normalize_remote_scores(raw_scores)

    def rerank_documents(self, query: str, documents: list, documents_batch_size: int = 4, llm_weight: float = 0.7):
        if not documents:
            self.last_debug = {
                "reranking_strategy": "single",
                "initial_candidate_pool_size": 0,
                "colbert_candidate_pool_size": None,
                "colbert_top_n": None,
                "final_reranking_backend": "vllm_api",
            }
            return []

        payload = {
            "model": self.model_name,
            "query": query,
            "documents": [doc.get("text", "") for doc in documents],
            "top_n": len(documents),
            "return_documents": False,
        }
        response = self._post(payload)
        if not response.ok:
            raise requests.HTTPError(
                f"{response.status_code} Client Error for url: {response.url}\n{response.text[:1000]}",
                response=response,
            )

        scores = self._extract_scores(response.json(), len(documents))
        vector_weight = 1 - llm_weight
        results = []
        for doc, score in zip(documents, scores):
            item = doc.copy()
            item["relevance_score"] = round(float(score), 4)
            item["final_relevance_score"] = item["relevance_score"]
            item["combined_score"] = round(
                llm_weight * item["relevance_score"] + vector_weight * float(item.get("distance", 0.0)),
                4,
            )
            results.append(item)

        results.sort(key=lambda item: item["combined_score"], reverse=True)
        self.last_debug = {
            "reranking_strategy": "single",
            "initial_candidate_pool_size": len(documents),
            "colbert_candidate_pool_size": None,
            "colbert_top_n": None,
            "final_reranking_backend": "vllm_api",
        }
        return results

class LLMReranker:
    def __init__(self, provider: str = "qwen", model: str = None):
        load_dotenv()
        self.provider = provider
        self.model = model or os.getenv("RERANKING_MODEL") or os.getenv("LLM_MODEL") or "Qwen3.5-35B-A3B-AWQ-4bit"
        self.processor = APIProcessor(provider=provider)
        self.system_prompt_rerank_single_block = prompts.RerankingPrompt.system_prompt_rerank_single_block
        self.system_prompt_rerank_multiple_blocks = prompts.RerankingPrompt.system_prompt_rerank_multiple_blocks
        self.schema_for_single_block = prompts.RetrievalRankingSingleBlock
        self.schema_for_multiple_blocks = prompts.RetrievalRankingMultipleBlocks
    
    def get_rank_for_single_block(self, query, retrieved_document):
        user_prompt = f'Here is the query:\n"{query}"\n\nHere is the retrieved text block:\n"""\n{retrieved_document}\n"""'

        return self.processor.send_message(
            model=self.model,
            temperature=0,
            system_content=self.system_prompt_rerank_single_block,
            human_content=user_prompt,
            is_structured=True,
            response_format=self.schema_for_single_block
        )

    def get_rank_for_multiple_blocks(self, query, retrieved_documents):
        formatted_blocks = "\n\n---\n\n".join([f'Block {i+1}:\n\n"""\n{text}\n"""' for i, text in enumerate(retrieved_documents)])
        user_prompt = (
            f"Here is the query: \"{query}\"\n\n"
            "Here are the retrieved text blocks:\n"
            f"{formatted_blocks}\n\n"
            f"You should provide exactly {len(retrieved_documents)} rankings, in order."
        )

        return self.processor.send_message(
            model=self.model,
            temperature=0,
            system_content=self.system_prompt_rerank_multiple_blocks,
            human_content=user_prompt,
            is_structured=True,
            response_format=self.schema_for_multiple_blocks
        )

    @staticmethod
    def _build_scored_document(doc: dict, relevance_score: float, llm_weight: float, vector_weight: float) -> dict:
        doc_with_score = doc.copy()
        numeric_score = float(relevance_score)
        doc_with_score["relevance_score"] = numeric_score
        doc_with_score["final_relevance_score"] = numeric_score
        doc_with_score["combined_score"] = round(
            llm_weight * numeric_score +
            vector_weight * float(doc.get("distance", 0.0)),
            4
        )
        return doc_with_score

    def _rerank_single_document(self, query: str, doc: dict, llm_weight: float, vector_weight: float) -> dict:
        ranking = self.get_rank_for_single_block(query, doc["text"])
        return self._build_scored_document(
            doc,
            ranking["relevance_score"],
            llm_weight=llm_weight,
            vector_weight=vector_weight,
        )

    def _fallback_batch_to_single_block(
        self,
        *,
        query: str,
        batch: list,
        llm_weight: float,
        vector_weight: float,
        reason: str,
    ) -> list[dict]:
        print(
            f"\nWarning: batch LLM reranking failed for {len(batch)} documents; "
            f"falling back to single-block reranking. Reason: {reason}"
        )
        return [
            self._rerank_single_document(query, doc, llm_weight=llm_weight, vector_weight=vector_weight)
            for doc in batch
        ]

    def rerank_documents(self, query: str, documents: list, documents_batch_size: int = 4, llm_weight: float = 0.7):
        """
        Rerank multiple documents using parallel processing with threading.
        Combines vector similarity and LLM relevance scores using weighted average.
        """
        # Create batches of documents
        doc_batches = [documents[i:i + documents_batch_size] for i in range(0, len(documents), documents_batch_size)]
        vector_weight = 1 - llm_weight
        
        if documents_batch_size == 1:
            def process_single_doc(doc):
                return self._rerank_single_document(
                    query,
                    doc,
                    llm_weight=llm_weight,
                    vector_weight=vector_weight,
                )

            # Process all documents in parallel using single-block method
            with ThreadPoolExecutor() as executor:
                all_results = list(executor.map(process_single_doc, documents))
                
        else:
            def process_batch(batch):
                texts = [doc['text'] for doc in batch]
                try:
                    rankings = self.get_rank_for_multiple_blocks(query, texts)
                except Exception as exc:
                    return self._fallback_batch_to_single_block(
                        query=query,
                        batch=batch,
                        llm_weight=llm_weight,
                        vector_weight=vector_weight,
                        reason=str(exc),
                    )

                results = []
                block_rankings = list(rankings.get('block_rankings', []) or [])

                if len(block_rankings) < len(batch):
                    print(f"\nWarning: Expected {len(batch)} rankings but got {len(block_rankings)}")
                    for i in range(len(block_rankings), len(batch)):
                        doc = batch[i]
                        print(f"Missing ranking for document on page {doc.get('page', 'unknown')}:")
                        print(f"Text preview: {doc['text'][:100]}...\n")

                for index, doc in enumerate(batch):
                    if index >= len(block_rankings):
                        results.append(
                            self._rerank_single_document(
                                query,
                                doc,
                                llm_weight=llm_weight,
                                vector_weight=vector_weight,
                            )
                        )
                        continue

                    rank = block_rankings[index] or {}
                    relevance_score = rank.get("relevance_score")
                    if relevance_score is None:
                        print(
                            f"\nWarning: batch ranking missing relevance_score for document on page "
                            f"{doc.get('page', 'unknown')}; falling back to single-block reranking."
                        )
                        results.append(
                            self._rerank_single_document(
                                query,
                                doc,
                                llm_weight=llm_weight,
                                vector_weight=vector_weight,
                            )
                        )
                        continue

                    results.append(
                        self._build_scored_document(
                            doc,
                            relevance_score,
                            llm_weight=llm_weight,
                            vector_weight=vector_weight,
                        )
                    )
                return results

            # Process batches in parallel using threads
            with ThreadPoolExecutor() as executor:
                batch_results = list(executor.map(process_batch, doc_batches))
            
            # Flatten results
            all_results = []
            for batch in batch_results:
                all_results.extend(batch)
        
        # Sort results by combined score in descending order
        all_results.sort(key=lambda x: x["combined_score"], reverse=True)
        self.last_debug = {
            "reranking_strategy": "single",
            "initial_candidate_pool_size": len(documents),
            "colbert_candidate_pool_size": None,
            "colbert_top_n": None,
            "final_reranking_backend": "llm_prompt",
        }
        return all_results


class ColBERTReranker:
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 16,
        query_max_length: int = 128,
        passage_max_length: int = 512,
    ):
        load_dotenv()
        resolved_model_name = model_name or _get_env_value("COLBERT_MODEL", "EMBEDDING_MODEL_NAME", default="BAAI/bge-m3")
        if not resolved_model_name:
            raise ValueError("A BGE-M3 model name or path is required for ColBERT-style reranking.")

        raw_device = device or _get_env_value("COLBERT_DEVICE", "RERANKING_DEVICE", default="cuda:0")
        self.device = _limit_devices(
            _parse_devices(raw_device),
            "COLBERT_MAX_DEVICES",
            "RERANKING_MAX_DEVICES",
        )[0]
        self.batch_size = max(1, int(batch_size))
        self.query_max_length = max(8, int(query_max_length))
        self.passage_max_length = max(16, int(passage_max_length))
        self.model_name = resolved_model_name
        use_fp16 = _get_env_value("COLBERT_USE_FP16", default="true").lower() in {"1", "true", "yes", "on"}
        self.use_fp16 = use_fp16 and str(self.device).startswith("cuda")
        self.model = _load_bgem3_model(
            model_name=self.model_name,
            device=self.device,
            use_fp16=self.use_fp16,
            batch_size=self.batch_size,
            query_max_length=self.query_max_length,
            passage_max_length=self.passage_max_length,
        )

    @staticmethod
    def _normalize_text(value: object) -> str:
        return " ".join(str(value or "").split())

    @staticmethod
    def _content_mask(input_ids: torch.Tensor, attention_mask: torch.Tensor, special_token_ids: list[int]) -> torch.Tensor:
        content_mask = attention_mask.bool()
        if not special_token_ids:
            return content_mask

        special_mask = torch.zeros_like(content_mask)
        for token_id in special_token_ids:
            special_mask |= input_ids == int(token_id)
        return content_mask & ~special_mask

    @staticmethod
    def _late_interaction_scores(
        query_embeddings: torch.Tensor,
        query_mask: torch.Tensor,
        doc_embeddings: torch.Tensor,
        doc_mask: torch.Tensor,
    ) -> torch.Tensor:
        similarity = torch.einsum("qd,bpd->bqp", query_embeddings, doc_embeddings)
        similarity = similarity.masked_fill(~doc_mask.unsqueeze(1), -1e4)
        max_similarity = similarity.max(dim=2).values
        max_similarity = max_similarity.masked_fill(~query_mask.unsqueeze(0), 0.0)
        return max_similarity.sum(dim=1)

    @staticmethod
    def _coerce_score(value) -> float:
        if hasattr(value, "item"):
            return float(value.item())
        return float(value)

    def _encode_colbert_query(self, query: str):
        normalized_query = self._normalize_text(query)
        if not normalized_query:
            raise ValueError("ColBERT query text cannot be empty.")
        outputs = _bgem3_encode_texts(
            self.model,
            [normalized_query],
            batch_size=1,
            max_length=self.query_max_length,
            return_dense=False,
            return_sparse=False,
            return_colbert_vecs=True,
        )
        colbert_vecs = list(outputs.get("colbert_vecs") or [])
        if not colbert_vecs:
            raise RuntimeError("BGE-M3 did not return query-side colbert_vecs for ColBERT-style reranking.")
        return colbert_vecs[0]

    def _encode_colbert_passages(self, texts: list[str]) -> list:
        normalized_texts = [self._normalize_text(text) for text in texts]
        outputs = _bgem3_encode_texts(
            self.model,
            normalized_texts,
            batch_size=self.batch_size,
            max_length=self.passage_max_length,
            return_dense=False,
            return_sparse=False,
            return_colbert_vecs=True,
        )
        colbert_vecs = list(outputs.get("colbert_vecs") or [])
        if len(colbert_vecs) != len(texts):
            raise RuntimeError(
                "BGE-M3 passage-side colbert_vecs count does not match the number of documents being reranked."
            )
        return colbert_vecs

    @torch.no_grad()
    def score_documents(self, query: str, documents: list[dict]) -> list[float]:
        if not documents:
            return []

        texts = [doc.get("text", "") for doc in documents]
        query_vecs = self._encode_colbert_query(query)
        passage_vecs = self._encode_colbert_passages(texts)
        return [
            self._coerce_score(self.model.colbert_score(query_vecs, doc_vecs))
            for doc_vecs in passage_vecs
        ]


class CascadeReranker:
    def __init__(
        self,
        *,
        colbert_model: str,
        colbert_device: Optional[str] = None,
        colbert_batch_size: int = 16,
        colbert_query_max_length: int = 128,
        colbert_passage_max_length: int = 512,
        cascade_candidate_pool_cap: int = 50,
        colbert_top_n: int = 10,
        final_reranking_batch_size: int = 2,
        final_reranker=None,
        final_reranking_backend: str = "flag_embedding",
        colbert_reranker: Optional[ColBERTReranker] = None,
    ):
        self.cascade_candidate_pool_cap = max(1, int(cascade_candidate_pool_cap))
        self.colbert_top_n = max(1, int(colbert_top_n))
        self.final_reranking_batch_size = max(1, int(final_reranking_batch_size))
        self.final_reranker = final_reranker
        self.final_reranking_backend = final_reranking_backend
        self.colbert_reranker = colbert_reranker or ColBERTReranker(
            model_name=colbert_model,
            device=colbert_device,
            batch_size=colbert_batch_size,
            query_max_length=colbert_query_max_length,
            passage_max_length=colbert_passage_max_length,
        )

    def rerank_documents(self, query: str, documents: list, documents_batch_size: int = 4, llm_weight: float = 0.7):
        if not documents:
            self.last_debug = {
                "reranking_strategy": "cascade",
                "initial_candidate_pool_size": 0,
                "colbert_candidate_pool_size": 0,
                "colbert_top_n": self.colbert_top_n,
                "final_reranking_backend": self.final_reranking_backend,
            }
            return []
        if self.final_reranker is None:
            raise ValueError("CascadeReranker requires a final_reranker.")

        sorted_documents = sorted(documents, key=lambda doc: float(doc.get("distance", 0.0)), reverse=True)
        initial_candidates = [doc.copy() for doc in sorted_documents[:self.cascade_candidate_pool_cap]]
        raw_scores = self.colbert_reranker.score_documents(query, initial_candidates)
        normalized_scores = _minmax_normalize(raw_scores)

        for doc, raw_score, normalized_score in zip(initial_candidates, raw_scores, normalized_scores):
            doc["distance_rrf"] = round(float(doc.get("distance", 0.0)), 4)
            doc["colbert_raw_score"] = round(float(raw_score), 4)
            doc["colbert_score"] = round(float(normalized_score), 4)
            doc["distance"] = doc["colbert_score"]
            doc["ranking_score"] = doc["colbert_score"]

        initial_candidates.sort(key=lambda doc: float(doc.get("colbert_score", 0.0)), reverse=True)
        colbert_candidates = initial_candidates[:self.colbert_top_n]

        final_results = self.final_reranker.rerank_documents(
            query=query,
            documents=colbert_candidates,
            documents_batch_size=self.final_reranking_batch_size,
            llm_weight=llm_weight,
        )
        for item in final_results:
            if "final_relevance_score" not in item and "relevance_score" in item:
                item["final_relevance_score"] = item["relevance_score"]

        self.last_debug = {
            "reranking_strategy": "cascade",
            "initial_candidate_pool_size": len(initial_candidates),
            "colbert_candidate_pool_size": len(colbert_candidates),
            "colbert_top_n": self.colbert_top_n,
            "final_reranking_backend": self.final_reranking_backend,
        }
        return final_results
    
