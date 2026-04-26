"""主RAG服务 - 集成查询改写、混合检索、重排序的完整RAG流水线"""

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory, RunnableLambda
from utils.file_history_store import get_history
from langchain_openai import OpenAIEmbeddings
from utils import config_loader as config
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from .vector_stores import VectorStoreService
from .query_rewriter import QueryRewriter
from .hybrid_retriever import HybridRetriever
from .reranker import Reranker
from .knowledge_base import register_bm25_rebuild_callback


def print_prompt(prompt):
    print("=" * 20)
    print(prompt.to_string())
    print("=" * 20)
    return prompt


class RagService(object):
    def __init__(self):
        embedding = OpenAIEmbeddings(
            model=config.embedding_model_name,
            base_url=config.embedding_base_url,
            api_key=config.embedding_api_key
        )

        self.vector_service = VectorStoreService(embedding=embedding)

        # 初始化各增强模块
        self.chat_model = ChatOpenAI(
            model=config.chat_model_name,
            base_url=config.chat_base_url,
            api_key=config.chat_api_key
        )
        self.query_rewriter = QueryRewriter(llm=self.chat_model)
        self.hybrid_retriever = HybridRetriever(
            vector_store=self.vector_service.vector_store,
            embedding=embedding
        )
        self.reranker = Reranker(llm=self.chat_model)

        # 注册BM25索引重建回调
        register_bm25_rebuild_callback(self.hybrid_retriever.rebuild_index)

        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "以我提供的已知参考资料为主，"
                 "简洁和专业的回答用户问题。参考资料:\n{context}。"),
                ("system", "并且我提供用户的对话历史记录，如下："),
                MessagesPlaceholder("history"),
                ("user", "请回答用户提问：{input}")
            ]
        )

        self.chain = self.__get_chain()

    def _retrieve_and_rerank(self, query: str) -> list[Document]:
        """执行完整的检索流水线: 改写 → 混合检索 → 重排序"""
        # 存储中间结果供UI调试展示
        self._last_rewritten_query = query
        self._last_raw_docs = []

        # Step 1: 查询改写
        if config.enable_query_rewriting:
            transformed = self.query_rewriter.transform(query)
            if isinstance(transformed, list):
                # multi_query: 对每个改写结果检索后合并
                self._last_rewritten_query = " | ".join(transformed)
                all_docs = []
                seen_texts = set()
                for q in transformed:
                    docs = self.hybrid_retriever.hybrid_search(q, k=config.reranker_retrieve_k)
                    for doc in docs:
                        if doc.page_content not in seen_texts:
                            seen_texts.add(doc.page_content)
                            all_docs.append(doc)
            else:
                self._last_rewritten_query = transformed
                all_docs = self.hybrid_retriever.hybrid_search(
                    transformed, k=config.reranker_retrieve_k
                )
        else:
            all_docs = self.hybrid_retriever.hybrid_search(
                query, k=config.reranker_retrieve_k
            )

        self._last_raw_docs = all_docs

        # Step 2: 重排序
        if config.enable_reranking and all_docs:
            ranked_docs = self.reranker.rerank(query, all_docs, top_k=config.similarity_threshold)
        else:
            ranked_docs = all_docs[:config.similarity_threshold]

        return ranked_docs

    def __get_chain(self):
        """获取最终的执行链"""
        def format_document(docs: list[Document]):
            if not docs:
                return "无相关参考资料"

            formatted_str = ""
            for i, doc in enumerate(docs, 1):
                formatted_str += f"[参考{i}] 文档片段：{doc.page_content}\n文档元数据：{doc.metadata}\n\n"
            return formatted_str

        def retrieve(value: dict) -> list[Document]:
            query = value["input"]
            return self._retrieve_and_rerank(query)

        def format_for_prompt_template(value):
            new_value = {}
            new_value["input"] = value["input"]["input"]
            new_value["context"] = value["context"]
            new_value["history"] = value["input"]["history"]
            return new_value

        chain = (
            {
                "input": RunnablePassthrough(),
                "context": RunnableLambda(retrieve) | format_document
            }
            | RunnableLambda(format_for_prompt_template)
            | self.prompt_template
            | print_prompt
            | self.chat_model
            | StrOutputParser()
        )

        conversation_chain = RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history",
        )

        return conversation_chain
