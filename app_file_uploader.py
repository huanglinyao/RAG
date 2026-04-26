"""
基于Streamlit完成WEB网页上传服务
支持 PDF 文件 (使用 PyPDFLoader)
"""
import time
import tempfile
import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from rag.knowledge_base import KnowledgeBaseService

st.title("学术知识库更新服务")

uploader_file = st.file_uploader(
    "请上传 PDF 文件",
    type=['pdf'],
    accept_multiple_files=False,
)

if "service" not in st.session_state:
    st.session_state["service"] = KnowledgeBaseService()

if uploader_file is not None:
    file_name = uploader_file.name
    file_type = uploader_file.type
    file_size = uploader_file.size / 1024

    st.subheader(f"文件名：{file_name}")
    st.write(f"格式：{file_type} | 大小：{file_size:.2f} KB")

    # 将上传的PDF保存为临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploader_file.getvalue())
        tmp_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_path, mode='page')   # 按页返回Document
        pages = []
        for doc in loader.lazy_load():
            pages.append(doc.page_content)
        text = "\n".join(pages)    # 将所有页的文本合并
        if not text.strip():
            st.warning("⚠️ PDF中未提取到文本内容，请确认文件非扫描件或图像型PDF")
            st.stop()
    finally:
        # 清理临时文件
        os.unlink(tmp_path)

    with st.spinner("载入知识库中..."):
        time.sleep(1)
        result = st.session_state["service"].upload_by_str(text, file_name)
        st.write(result)