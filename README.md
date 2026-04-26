<br />

<div align="center">
  <h3 align="center">AcademicRAG🎓：基于RAG的学术智能问答系统</h3>

  <p align="center">
    面向学术研究场景，基于检索增强生成（RAG）技术构建专业文献知识库，<br />
    实现"知识库构建 → 查询理解 → 混合检索 → 重排序 → 答案生成 → 效果评估"全链路闭环。
  </p>

</div>

---

## 目录

- [系统架构](#系统架构)
- [核心模块](#核心模块)
- [技术栈](#技术栈)
- [快速开始](#快速开始)
- [配置说明](#配置说明)

---

## 系统架构

### 离线知识库构建

```
PDF文档 → PyPDFLoader文本提取 → MD5去重 → RecursiveCharacterTextSplitter分块
→ OpenAIEmbeddings向量化 → ChromaDB向量存储 + BM25关键词索引
```

### 在线问答流水线

```
用户问题 → [查询改写] → [混合检索] → [重排序] → LLM生成 → 流式回答
```

---

## 核心模块

### 1. 查询改写（Query Rewriting）

支持三种策略，通过配置文件热切换：

| 策略 | 说明 |
|------|------|
| **rewrite** | LLM 将口语化问题改写为结构化检索查询 |
| **multi_query** | 从多个角度生成查询变体，合并检索结果 |
| **HyDE** | 生成假设文档片段作为检索查询，缩小语义鸿沟 |

### 2. 混合检索（Hybrid Search）

- **向量检索**：ChromaDB 稠密语义检索，捕获深层语义匹配
- **关键词检索**：rank-bm25 + jieba 中文分词，覆盖精确关键词匹配
- **加权融合**：`score = alpha × vector_score + (1-alpha) × keyword_score`，alpha 可配置

### 3. 重排序（Re-ranking）

- **Cross-Encoder**：BAAI/bge-reranker-v2-m3 对检索结果精细化评分重排
- **LLM-as-Judge**：使用 Qwen3 逐条评分，加载失败自动降级保障

### 4. 效果评估

基于 RAGAS 框架的定量评估体系：

| 指标 | 说明 |
|------|------|
| **Faithfulness** | 答案是否忠实于检索上下文 |
| **Answer Relevancy** | 答案与问题的相关性 |
| **Context Precision** | 检索结果中相关上下文的比例 |
| **Context Recall** | 所有相关上下文被检索到的比例 |

评估结果自动保存至 `data/eval_results/`，记录完整配置与指标。

---

## 技术栈

| 层级 | 选型 |
|------|------|
| 应用层 | Streamlit |
| 框架层 | LangChain LCEL |
| 向量库 | ChromaDB |
| 关键词索引 | rank-bm25 + jieba |
| 嵌入模型 | Qwen3-Embedding-4B（本地 API） |
| 生成模型 | Qwen3-8B（本地 API） |
| 重排序 | BAAI/bge-reranker-v2-m3 |
| 评估框架 | RAGAS |

---

## 快速开始

### 环境准备

```bash
conda create -n myenv python=3.10
conda activate myenv
pip install -r requirements.txt
```

### 配置模型

编辑 `config/config.yaml`，填入本地模型 API 地址：

```yaml
chat_model_name: "xx"
chat_base_url: "http://your-api"

embedding_model_name: "xx"
embedding_base_url: "http://your-api"
```

### 启动服务

```bash
# 启动知识库上传服务（Streamlit）
streamlit run app_file_uploader.py

# 启动问答服务（Streamlit）
streamlit run app_qa.py
```

### 运行评估

```bash
python -m evaluation.ragas_eval
```

---

## 配置说明

核心参数集中在 `config/config.yaml`：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `chunk_size` | 2000 | 文本分块大小 |
| `chunk_overlap` | 200 | 分块重叠大小 |
| `hybrid_search_alpha` | 0.6 | 混合检索向量权重 |
| `enable_query_rewriting` | true | 是否启用查询改写 |
| `query_rewrite_method` | rewrite | 改写策略 |
| `enable_reranking` | true | 是否启用重排序 |
| `reranker_type` | cross_encoder | 重排序方式 |
| `similarity_threshold` | 5 | 最终返回结果数量 |
