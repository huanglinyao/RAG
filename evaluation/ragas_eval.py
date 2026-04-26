"""RAGAS评估模块 - 对RAG流水线进行定量评估

使用方法:
    python -m evaluation.ragas_eval

评估指标:
    - faithfulness: 生成答案是否忠实于检索到的上下文
    - answer_relevancy: 答案与问题的相关性
    - context_precision: 检索到的上下文中有多少是相关的
    - context_recall: 所有相关上下文中有多少被检索到
"""

import os
import json
import sys
from datetime import datetime
from typing import Optional

# 将项目根目录加入sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from datasets import Dataset
from utils import config_loader as config


class RagasEvaluator:
    """RAGAS评估器: 使用RAGAS库对RAG系统进行评估"""

    def __init__(self):
        self.chat_model = ChatOpenAI(
            model=config.chat_model_name,
            base_url=config.chat_base_url,
            api_key=config.chat_api_key
        )
        self.embeddings = OpenAIEmbeddings(
            model=config.embedding_model_name,
            base_url=config.embedding_base_url,
            api_key=config.embedding_api_key
        )
        self._init_ragas()

    def _init_ragas(self):
        """配置RAGAS使用本地模型"""
        try:
            from ragas.llms import LangchainLLMWrapper
            from ragas.embeddings import LangchainEmbeddingsWrapper

            ragas_llm = LangchainLLMWrapper(self.chat_model)
            ragas_emb = LangchainEmbeddingsWrapper(self.embeddings)

            self.llm = ragas_llm
            self.emb = ragas_emb
        except ImportError:
            raise ImportError(
                "请先安装ragas库: pip install ragas datasets"
            )

    def load_test_data(self, file_path: Optional[str] = None) -> list[dict]:
        """加载测试数据集

        测试数据格式 (JSON):
        [
            {
                "question": "问题",
                "ground_truth": "标准答案",
                "ground_truth_contexts": ["相关上下文1", "相关上下文2"]
            },
            ...
        ]
        """
        path = file_path or config.eval_test_data_path
        if not os.path.exists(path):
            print(f"测试数据文件不存在: {path}")
            print("请创建测试数据文件，格式参考:")
            print(json.dumps([
                {
                    "question": "什么是Transformer?",
                    "ground_truth": "Transformer是一种基于自注意力机制的神经网络架构...",
                    "ground_truth_contexts": ["相关段落1"]
                }
            ], ensure_ascii=False, indent=2))
            return []

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"加载了 {len(data)} 条测试数据")
        return data

    def run_pipeline(self, test_data: list[dict]) -> dict:
        """运行RAG流水线并收集评估结果

        对每条测试数据:
        1. 通过RAG系统获取答案
        2. 记录检索到的上下文
        3. 收集用于RAGAS评估的数据
        """
        # 延迟导入RAG服务（避免循环依赖）
        from rag.rag import RagService

        rag_service = RagService()

        questions = []
        answers = []
        contexts = []
        ground_truths = []

        for i, item in enumerate(test_data):
            question = item["question"]
            ground_truth = item["ground_truth"]

            print(f"[{i+1}/{len(test_data)}] 处理问题: {question[:50]}...")

            # 通过RAG链获取回答
            try:
                response = ""
                for chunk in rag_service.chain.stream(
                    {"input": question},
                    config={"configurable": {"session_id": "eval_session"}}
                ):
                    response += chunk
            except Exception as e:
                print(f"  处理失败: {e}")
                response = ""

            # 获取检索到的上下文
            retrieved_docs = getattr(rag_service, '_last_raw_docs', [])
            retrieved_texts = [doc.page_content for doc in retrieved_docs]

            questions.append(question)
            answers.append(response)
            contexts.append(retrieved_texts)
            ground_truths.append(ground_truth)

            print(f"  回答: {response[:80]}...")

        return {
            "questions": questions,
            "answers": answers,
            "contexts": contexts,
            "ground_truths": ground_truths,
        }

    def compute_metrics(self, eval_data: dict) -> dict:
        """使用RAGAS计算评估指标"""
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )

        # 构建HuggingFace Dataset
        dataset = Dataset.from_dict({
            "question": eval_data["questions"],
            "answer": eval_data["answers"],
            "contexts": eval_data["contexts"],
            "ground_truth": eval_data["ground_truths"],
        })

        # 配置指标使用本地模型
        metrics_to_use = []
        for metric_name in config.eval_metrics:
            if metric_name == "faithfulness":
                faithfulness.llm = self.llm
                metrics_to_use.append(faithfulness)
            elif metric_name == "answer_relevancy":
                answer_relevancy.llm = self.llm
                answer_relevancy.embeddings = self.emb
                metrics_to_use.append(answer_relevancy)
            elif metric_name == "context_precision":
                context_precision.llm = self.llm
                metrics_to_use.append(context_precision)
            elif metric_name == "context_recall":
                context_recall.llm = self.llm
                context_recall.embeddings = self.emb
                metrics_to_use.append(context_recall)

        print(f"计算指标: {[m.name for m in metrics_to_use]}")
        result = evaluate(dataset, metrics=metrics_to_use)

        # 转换为可序列化字典
        result_dict = {}
        for k, v in result.items():
            if hasattr(v, 'item'):
                result_dict[k] = v.item()
            elif isinstance(v, (list, tuple)):
                result_dict[k] = [float(x) if hasattr(x, 'item') else x for x in v]
            else:
                result_dict[k] = v

        return result_dict

    def save_results(self, results: dict):
        """保存评估结果"""
        output_dir = config.eval_result_path
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"eval_result_{timestamp}.json")

        output_data = {
            "timestamp": timestamp,
            "config": {
                "chat_model": config.chat_model_name,
                "embedding_model": config.embedding_model_name,
                "query_rewriting": config.enable_query_rewriting,
                "query_rewrite_method": config.query_rewrite_method,
                "hybrid_search_alpha": config.hybrid_search_alpha,
                "reranking": config.enable_reranking,
                "reranker_type": config.reranker_type,
                "top_k": config.similarity_threshold,
            },
            "metrics": {k: v for k, v in results.items() if k != "per_question"},
            "per_question": results.get("per_question", []),
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"\n评估结果已保存: {output_path}")
        return output_path


def main():
    """主入口: 运行RAGAS评估"""
    evaluator = RagasEvaluator()

    # 1. 加载测试数据
    test_data = evaluator.load_test_data()
    if not test_data:
        return

    # 2. 运行RAG流水线
    print("\n===== 开始运行RAG流水线 =====")
    eval_data = evaluator.run_pipeline(test_data)

    # 3. 计算RAGAS指标
    print("\n===== 开始计算RAGAS指标 =====")
    results = evaluator.compute_metrics(eval_data)

    # 4. 打印结果
    print("\n===== 评估结果 =====")
    for metric, value in results.items():
        if metric != "per_question":
            print(f"{metric}: {value:.4f}")

    # 5. 保存结果
    evaluator.save_results(results)


if __name__ == "__main__":
    main()
