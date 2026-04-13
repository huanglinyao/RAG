<br />

<div align="center">
  <h3 align="center">AcademicRAG🎓：基于RAG的学术智能问答系统</h3>

  <p align="center">
    本系统面向学术研究场景，基于检索增强生成（RAG）技术，利用专业学术文献构建知识库，提升领域问答准确性，
    <br />
    有效抑制大语言模型在学术概念、实验数据、文献观点等方面可能产生的幻觉问题。
    <br />
  </p>

</div>

## 介绍

#### **离线部分：学术知识库构建**

- 📄 **文档上传解析**：支持上传 **PDF 格式**的**学术论文**、**技术报告**等，自动提取文本内容。
- ♻️ **去重与增量更新**：通过 **MD5 校验** 避免相同文献重复入库，支持知识库持续扩充。
- ✂️ **智能文本分割**：按段落、句子等自然边界分割长文本，兼顾语义完整性与检索粒度。
- 💾 **向量化与存储**：调用本地嵌入模型将文本块转为向量，存入 **Chroma 向量数据库**，并记录文献来源、时间等元数据便于溯源。

#### **在线部分：学术问答与对话**

- 💬 **多轮学术对话**：基于 **Streamlit** 构建聊天界面，支持连续追问，对话历史持久化存储。
- 🔍 **检索增强生成**：将用户问题向量化，检索最相关文献片段作为参考资料，结合对话历史，调用本地大语言模型生成专业回答，并流式输出。





---

## Getting Started 🚀

建议使用 `conda` 虚拟环境，以获得受控制的开发环境。请设置一个虚拟环境，并按照 `requirements.txt` 文件中的说明安装库。

```python
conda create -n myenv python=3.10
conda activate myenv
pip install -r requirements.txt
```

---



## 启动服务

对config.yaml的有关参数进行修改

```python
# ===== 聊天模型配置 =====
chat_model_name: "xxx"
chat_base_url: "xxx"
chat_api_key: "v"

# ===== 嵌入模型配置 =====
embedding_model_name: "xxx"
embedding_base_url: "xxx"
embedding_api_key: "xxx"
```

---

在终端激活虚拟环境，将文献pdf文件放置在 `./data/` 文件下，并运行以下命令激活 **学术知识库服务** 与 **学术问答对话服务**：

```python
# 激活虚拟环境
conda activate myenv

# 将路径切换到
cd /path/to/your/project/AcademicRAG

# 启动学术知识库服务 (streamlit webui)
streamlit run app_rag.py

# 启动学术学术问答对话服务  (streamlit webui)
streamlit run app.py
```

---



## 技术支持

- **python**
- **langchain**
- **chroma**
- **streamlit**

