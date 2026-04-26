import time
from rag.rag import RagService
import streamlit as st
from utils import config_loader as config

# 标题
st.title("学术智能问答系统🎓")
st.divider()            # 分隔符

if "message" not in st.session_state:
    st.session_state["message"] = [{"role": "assistant", "content": "你好，有什么可以帮助你？"}]

if "rag" not in st.session_state:
    st.session_state["rag"] = RagService()

# 侧边栏：显示检索状态和配置
with st.sidebar:
    st.subheader("当前配置")
    st.write(f"- 查询改写: {'✅ 开启' if config.enable_query_rewriting else '❌ 关闭'} ({config.query_rewrite_method})")
    st.write(f"- 混合检索: alpha={config.hybrid_search_alpha} (1=纯向量)")
    st.write(f"- 重排序: {'✅ 开启' if config.enable_reranking else '❌ 关闭'} ({config.reranker_type})")
    st.write(f"- 返回结果数: k={config.similarity_threshold}")
    st.divider()

    if st.button("清空对话历史"):
        st.session_state["message"] = [{"role": "assistant", "content": "你好，有什么可以帮助你？"}]
        st.rerun()

for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

# 在页面最下方提供用户输入栏
prompt = st.chat_input()

if prompt:

    # 在页面输出用户的提问
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})

    ai_res_list = []
    with st.spinner("思考中..."):

        # 调用RAG链生成回答
        res_stream = st.session_state["rag"].chain.stream({"input": prompt}, config.session_config)

        def capture(generator, cache_list):
            for chunk in generator:
                cache_list.append(chunk)
                yield chunk

        st.chat_message("assistant").write_stream(capture(res_stream, ai_res_list))
        full_response = "".join(ai_res_list)
        st.session_state["message"].append({"role": "assistant", "content": full_response})

    # 显示检索过程的调试信息（折叠面板）
    rag_svc = st.session_state["rag"]
    with st.expander("🔍 查看检索详情", expanded=False):
        st.markdown("**改写后的查询:**")
        st.code(rag_svc._last_rewritten_query if hasattr(rag_svc, '_last_rewritten_query') else prompt)

        st.markdown("**检索到的文档:**")
        if hasattr(rag_svc, '_last_raw_docs') and rag_svc._last_raw_docs:
            for i, doc in enumerate(rag_svc._last_raw_docs[:config.similarity_threshold], 1):
                st.markdown(f"**参考{i}** (来源: {doc.metadata.get('source', '未知')})")
                st.text(doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""))
                st.divider()
        else:
            st.write("无检索结果")
