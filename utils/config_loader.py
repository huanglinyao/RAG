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

chat_model_name = _cfg["chat_model_name"]
chat_base_url = _cfg["chat_base_url"]
chat_api_key = _cfg["chat_api_key"]

embedding_model_name = _cfg["embedding_model_name"]
embedding_base_url = _cfg["embedding_base_url"]
embedding_api_key = _cfg["embedding_api_key"]

session_config = _cfg["session_config"]