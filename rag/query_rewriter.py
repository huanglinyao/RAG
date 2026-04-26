"""查询改写模块 - 通过LLM对用户问题进行改写以提升检索效果"""

import re
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from utils import config_loader as config


class QueryRewriter:
    """查询改写器，支持三种策略:
    - rewrite: 将用户问题改写为更利于检索的形式
    - multi_query: 生成多个角度的查询
    - hyde: 生成假设文档，用文档内容做检索
    """

    def __init__(self, llm: Optional[ChatOpenAI] = None):
        self.llm = llm or ChatOpenAI(
            model=config.chat_model_name,
            base_url=config.chat_base_url,
            api_key=config.chat_api_key
        )
        self.method = config.query_rewrite_method

    def rewrite(self, query: str) -> str:
        """将用户问题改写为更清晰、更适合检索的形式"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个专业的查询改写助手。请将用户的提问改写为更清晰、更适合检索的形式。"
                       "要求：保持原意不变，去除口语化表达，使问题更加明确和结构化。"
                       "直接返回改写后的查询，不要加任何解释。"),
            ("user", "{query}")
        ])
        chain = prompt | self.llm
        result = chain.invoke({"query": query})
        return result.content.strip()

    def multi_query(self, query: str) -> list[str]:
        """从多个角度生成查询变体"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个专业的查询扩展助手。请从3个不同的角度对用户的问题进行改写，"
                       "以帮助从知识库中检索到更全面的相关信息。"
                       "每个角度一行，直接输出，不要序号和额外解释。"),
            ("user", "{query}")
        ])
        chain = prompt | self.llm
        result = chain.invoke({"query": query})
        lines = [line.strip() for line in result.content.strip().split('\n') if line.strip()]
        # 去重并返回
        seen = set()
        queries = [query]  # 包含原问题
        for line in lines:
            if line not in seen:
                seen.add(line)
                queries.append(line)
        return queries

    def hyde(self, query: str) -> str:
        """生成假设文档，用生成的文档内容作为检索查询"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个学术文档生成助手。请根据用户的问题，生成一段假设的学术文档内容。"
                       "内容应该像真实的学术论文段落一样详细、专业、包含关键术语。"
                       "这段内容将用于检索相似文档，所以请确保包含相关的技术术语和概念。"
                       "直接输出文档内容，不要加任何解释。"),
            ("user", "{query}")
        ])
        chain = prompt | self.llm
        result = chain.invoke({"query": query})
        return result.content.strip()

    def transform(self, query: str) -> str | list[str]:
        """根据配置的改写方法转换查询"""
        if self.method == "multi_query":
            return self.multi_query(query)
        elif self.method == "hyde":
            return self.hyde(query)
        else:
            return self.rewrite(query)
