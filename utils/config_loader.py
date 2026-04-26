import os
import yaml

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_DIR)           # 得到 .../AcademicRAG
_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config", "config.yaml")

with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
    _cfg = yaml.safe_load(f)

# 导出配置变量
md5_path = _cfg["md5_path"]
collection_name = _cfg["collection_name"]
persist_directory = _cfg["persist_directory"]

chunk_size = _cfg["chunk_size"]
chunk_overlap = _cfg["chunk_overlap"]
separators = _cfg["separators"]
max_split_char_number = _cfg["max_split_char_number"]

similarity_threshold = _cfg["similarity_threshold"]

# 混合检索配置
hybrid_search_alpha = _cfg.get("hybrid_search_alpha", 0.6)
bm25_k = _cfg.get("bm25_k", 5)
rerank_top_k = _cfg.get("rerank_top_k", 3)

# 查询改写配置
enable_query_rewriting = _cfg.get("enable_query_rewriting", True)
query_rewrite_method = _cfg.get("query_rewrite_method", "rewrite")

# 重排序配置
enable_reranking = _cfg.get("enable_reranking", True)
reranker_type = _cfg.get("reranker_type", "cross_encoder")
reranker_model_name = _cfg.get("reranker_model_name", "BAAI/bge-reranker-v2-m3")
reranker_retrieve_k = _cfg.get("reranker_retrieve_k", 10)

# RAGAS评估配置
eval_test_data_path = _cfg.get("eval_test_data_path", "./data/eval_test_data.json")
eval_result_path = _cfg.get("eval_result_path", "./data/eval_results")
eval_metrics = _cfg.get("eval_metrics", ["faithfulness", "answer_relevancy", "context_precision", "context_recall"])

chat_model_name = _cfg["chat_model_name"]
chat_base_url = _cfg["chat_base_url"]
chat_api_key = _cfg["chat_api_key"]

embedding_model_name = _cfg["embedding_model_name"]
embedding_base_url = _cfg["embedding_base_url"]
embedding_api_key = _cfg["embedding_api_key"]

session_config = _cfg["session_config"]
