"""检索重排序模块 - 对检索结果进行精细化排序，提升最终生成质量"""

from typing import Optional
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from utils import config_loader as config


class Reranker:
    """重排序器，支持两种策略:
    - cross_encoder: 使用交叉编码器模型进行重排序
    - llm: 使用LLM对文档进行相关性评分
    """

    def __init__(self, llm: Optional[ChatOpenAI] = None):
        self.reranker_type = config.reranker_type
        self.llm = llm or ChatOpenAI(
            model=config.chat_model_name,
            base_url=config.chat_base_url,
            api_key=config.chat_api_key
        )
        self._cross_encoder = None

    def _get_cross_encoder(self):
        """延迟加载cross-encoder模型"""
        if self._cross_encoder is None:
            try:
                from sentence_transformers import CrossEncoder
                self._cross_encoder = CrossEncoder(
                    config.reranker_model_name,
                    trust_remote_code=True
                )
            except Exception as e:
                print(f"[Reranker] Cross-encoder模型加载失败: {e}，降级为LLM排序")
                self.reranker_type = "llm"
        return self._cross_encoder

    def _rerank_cross_encoder(self, query: str, docs: list[Document], top_k: int) -> list[Document]:
        """使用Cross-Encoder模型重排序"""
        model = self._get_cross_encoder()
        if model is None:
            return self._rerank_llm(query, docs, top_k)

        pairs = [[query, doc.page_content] for doc in docs]
        scores = model.predict(pairs)

        scored = list(zip(scores, docs))
        scored.sort(key=lambda x: x[0], reverse=True)

        return [doc for _, doc in scored[:top_k]]

    def _rerank_llm(self, query: str, docs: list[Document], top_k: int) -> list[Document]:
        """使用LLM对文档进行相关性评分并排序"""
        scored_docs = []
        for doc in docs:
            score = self._llm_judge_relevance(query, doc.page_content)
            scored_docs.append((score, doc))

        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs[:top_k]]

    def _llm_judge_relevance(self, query: str, document: str) -> float:
        """让LLM判断文档与查询的相关性 (0~5分)"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个文档相关性评估专家。请判断以下文档片段与用户问题的相关程度。"
                       "评分标准：0=完全不相关，1=略微相关，2=有点相关，3=中等相关，4=很相关，5=非常相关。"
                       "只返回一个数字评分，不要其他内容。"),
            ("human", "用户问题: {query}\n\n文档片段: {document}")
        ])
        chain = prompt | self.llm
        try:
            result = chain.invoke({"query": query, "document": document[:1500]}).content.strip()
            score = float(result)
            score = max(0.0, min(5.0, score))
        except (ValueError, TypeError):
            score = 0.0
        return score

    def rerank(self, query: str, docs: list[Document], top_k: Optional[int] = None) -> list[Document]:
        """对检索结果进行重排序

        Args:
            query: 原始用户查询
            docs: 待排序的文档列表
            top_k: 返回top-k个结果，默认从配置读取

        Returns:
            重排序后的Document列表
        """
        if not docs:
            return []

        top_k = top_k or config.rerank_top_k
        top_k = min(top_k, len(docs))

        if self.reranker_type == "cross_encoder":
            return self._rerank_cross_encoder(query, docs, top_k)
        else:
            return self._rerank_llm(query, docs, top_k)
