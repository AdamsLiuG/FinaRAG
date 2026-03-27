import os
from dotenv import load_dotenv
import requests
import src.prompts as prompts
from concurrent.futures import ThreadPoolExecutor
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.api_requests import APIProcessor


def _get_env_value(*names, default=None):
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


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
        self.devices = _parse_devices(self.device)
        use_fp16 = _get_env_value("RERANKING_USE_FP16", default="true").lower() in {"1", "true", "yes", "on"}
        trust_remote_code = _get_env_value("RERANKING_TRUST_REMOTE_CODE", default="false").lower() in {
            "1", "true", "yes", "on"
        }
        self.use_fp16 = use_fp16 and any(device.startswith("cuda") for device in self.devices)
        # Keep one model per target device and shard work here. This also avoids
        # FlagEmbedding's prepare_for_model + tokenizer.pad path that triggers
        # the XLMRobertaTokenizerFast warning.
        self.models = [
            FastTokenizerFlagRerankerModel(
                model_name=self.model_name,
                device=device,
                use_fp16=self.use_fp16,
                trust_remote_code=trust_remote_code,
            )
            for device in self.devices
        ]

    @staticmethod
    def _normalize_scores(scores):
        if isinstance(scores, (float, int)):
            return [float(scores)]
        return [float(score) for score in scores]

    def rerank_documents(self, query: str, documents: list, documents_batch_size: int = 4, llm_weight: float = 0.7):
        pairs = [[query, doc["text"]] for doc in documents]
        if len(self.models) == 1:
            scores = self._normalize_scores(self.models[0].compute_score(pairs, normalize=True))
        else:
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

        vector_weight = 1 - llm_weight
        results = []
        for doc, score in zip(documents, scores):
            item = doc.copy()
            item["relevance_score"] = round(float(score), 4)
            item["combined_score"] = round(
                llm_weight * item["relevance_score"] + vector_weight * item["distance"], 4
            )
            results.append(item)

        results.sort(key=lambda x: x["combined_score"], reverse=True)
        return results

class LLMReranker:
    def __init__(self, provider: str = "qwen", model: str = None):
        load_dotenv()
        self.provider = provider
        self.model = model or os.getenv("RERANKING_MODEL") or os.getenv("LLM_MODEL") or "Qwen/Qwen2.5-72B-Instruct"
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
                # Get ranking for single document
                ranking = self.get_rank_for_single_block(query, doc['text'])
                
                doc_with_score = doc.copy()
                doc_with_score["relevance_score"] = ranking["relevance_score"]
                # Calculate combined score - note that distance is inverted since lower is better
                doc_with_score["combined_score"] = round(
                    llm_weight * ranking["relevance_score"] + 
                    vector_weight * doc['distance'],
                    4
                )
                return doc_with_score

            # Process all documents in parallel using single-block method
            with ThreadPoolExecutor() as executor:
                all_results = list(executor.map(process_single_doc, documents))
                
        else:
            def process_batch(batch):
                texts = [doc['text'] for doc in batch]
                rankings = self.get_rank_for_multiple_blocks(query, texts)
                results = []
                block_rankings = rankings.get('block_rankings', [])
                
                if len(block_rankings) < len(batch):
                    print(f"\nWarning: Expected {len(batch)} rankings but got {len(block_rankings)}")
                    for i in range(len(block_rankings), len(batch)):
                        doc = batch[i]
                        print(f"Missing ranking for document on page {doc.get('page', 'unknown')}:")
                        print(f"Text preview: {doc['text'][:100]}...\n")
                    
                    for _ in range(len(batch) - len(block_rankings)):
                        block_rankings.append({
                            "relevance_score": 0.0, 
                            "reasoning": "Default ranking due to missing LLM response"
                        })
                
                for doc, rank in zip(batch, block_rankings):
                    doc_with_score = doc.copy()
                    doc_with_score["relevance_score"] = rank["relevance_score"]
                    doc_with_score["combined_score"] = round(
                        llm_weight * rank["relevance_score"] + 
                        vector_weight * doc['distance'],
                        4
                    )
                    results.append(doc_with_score)
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
        return all_results
    
