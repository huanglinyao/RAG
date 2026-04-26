"""混合检索模块 - 结合向量检索(BM25)与关键词检索，提升召回质量"""

from typing import Optional
import numpy as np
from rank_bm25 import BM25Okapi
import jieba
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from utils import config_loader as config


class HybridRetriever:
    """混合检索器: 向量检索 + BM25关键词检索，加权融合"""

    def __init__(self, vector_store: Chroma, embedding: Embeddings):
        self.vector_store = vector_store
        self.embedding = embedding
        self.bm25: Optional[BM25Okapi] = None
        self.corpus_docs: list[tuple[str, dict]] = []  # [(text, metadata), ...]
        self._build_bm25_index()

    def _tokenize(self, text: str) -> list[str]:
        """中文分词"""
        return list(jieba.cut(text))

    def _build_bm25_index(self):
        """从ChromaDB中读取所有文档，构建BM25索引"""
        try:
            all_docs = self.vector_store.get(limit=10000)
            texts = all_docs.get("documents", []) or []
            metadatas = all_docs.get("metadatas", []) or []
            if texts:
                tokenized_corpus = [self._tokenize(t) for t in texts]
                self.bm25 = BM25Okapi(tokenized_corpus)
                self.corpus_docs = list(zip(texts, metadatas))
        except Exception:
            self.bm25 = None
            self.corpus_docs = []

    def rebuild_index(self):
        """重新构建BM25索引（在知识库更新后调用）"""
        self._build_bm25_index()

    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        alpha: Optional[float] = None
    ) -> list[Document]:
        """混合搜索: 加权融合向量检索和BM25检索结果

        Args:
            query: 查询字符串
            k: 返回结果数量
            alpha: 向量检索权重 (0~1), None则从配置读取

        Returns:
            排序后的Document列表
        """
        alpha = alpha if alpha is not None else config.hybrid_search_alpha

        # 1. 向量检索
        vector_results = self.vector_store.similarity_search_with_score(query, k=config.reranker_retrieve_k)
        vector_docs = []
        for doc, score in vector_results:
            # Chroma返回的是距离(越小越相关)，转为0~1相似度
            sim_score = 1.0 / (1.0 + score)
            vector_docs.append((doc.page_content, doc.metadata, sim_score))

        # 2. BM25检索
        keyword_docs = []
        if self.bm25 and self.corpus_docs:
            tokenized_query = self._tokenize(query)
            bm25_scores = self.bm25.get_scores(tokenized_query)
            # 取top-k BM25结果
            top_k_indices = np.argsort(bm25_scores)[::-1][:config.reranker_retrieve_k]
            for idx in top_k_indices:
                score = bm25_scores[idx]
                if score > 0:
                    text, metadata = self.corpus_docs[idx]
                    keyword_docs.append((text, metadata, score))

        # 3. 归一化 + 融合
        all_texts = {}
        # 向量分数归一化
        if vector_docs:
            v_scores = [s for _, _, s in vector_docs]
            v_max, v_min = max(v_scores), min(v_scores)
            v_range = v_max - v_min if v_max - v_min > 1e-8 else 1.0
            for text, meta, score in vector_docs:
                norm_score = (score - v_min) / v_range
                all_texts[text] = {
                    "metadata": meta,
                    "vector_score": norm_score,
                    "keyword_score": 0.0
                }

        # 关键词分数归一化
        if keyword_docs:
            k_scores = [s for _, _, s in keyword_docs]
            k_max, k_min = max(k_scores), min(k_scores)
            k_range = k_max - k_min if k_max - k_min > 1e-8 else 1.0
            for text, meta, score in keyword_docs:
                norm_score = (score - k_min) / k_range
                if text in all_texts:
                    all_texts[text]["keyword_score"] = norm_score
                else:
                    all_texts[text] = {
                        "metadata": meta,
                        "vector_score": 0.0,
                        "keyword_score": norm_score
                    }

        # 4. 加权融合排序
        scored_docs = []
        for text, scores in all_texts.items():
            fused = alpha * scores["vector_score"] + (1 - alpha) * scores["keyword_score"]
            scored_docs.append((text, scores["metadata"], fused))

        # 按融合分数降序排列，取top-k
        scored_docs.sort(key=lambda x: x[2], reverse=True)
        top_docs = scored_docs[:k]

        return [
            Document(page_content=text, metadata=meta)
            for text, meta, _ in top_docs
        ]

    def get_retriever(self, k: Optional[int] = None):
        """兼容LangChain Chain的retriever接口"""
        k = k or config.similarity_threshold

        def _retrieve(query: str) -> list[Document]:
            return self.hybrid_search(query, k=k)

        return _retrieve
