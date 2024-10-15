import os
import gradio as gr
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
os.environ['HTTP_PROXY'] = '127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = '127.0.0.1:7890'

proxy = 'http://127.0.0.1:7890'


def initialize_sales_bot(vector_store_dir: str = "orange_sales_vector_database"):
    # 加载向量数据库
    db = FAISS.load_local("orange_sales_vector_database", OpenAIEmbeddings(),
                          allow_dangerous_deserialization=True)

    # 定义检索器
    retriever = db.as_retriever(search_type="similarity_score_threshold",
                                search_kwargs={"score_threshold": 0.75})

    # 定义大模型
    llm = ChatOpenAI(model="gpt-4o-mini",
                     openai_proxy='http://127.0.0.1:7890')

    # 根据历史记录重构用户问题prompt
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # 根据上下文回答问题
    system_prompt = (
        "你是一个卖橘子的农民，你的名字叫 橘小莲 。"
        "你的任务是卖橘子，并且耐心回答用户问题 。 "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know. "
        "Use three sentences maximum and keep the answer concise."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(
        history_aware_retriever, question_answer_chain)

    # 管理聊天记录
    store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    # 带记忆和上下文的销售顾问
    global SALES_BOT
    SALES_BOT = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return SALES_BOT


def sales_chat(message, history):
    print(f"[message]{message}")
    print(f"[history]{history}")
    # TODO: 从命令行参数中获取
    enable_chat = True

    ans = SALES_BOT.invoke(
        {"input": message},
        config={
            "configurable": {"session_id": "abc123"}
        },  # constructs a key "abc123" in `store`.
    )
    # 如果检索出结果，或者开了大模型聊天模式
    # 返回 RetrievalQA combine_documents_chain 整合的结果
    if ans["context"] or enable_chat:
        print(f"[result]{ans['answer']}")
        print(f"[source_documents]{ans['context']}")
        return ans["answer"]
    # 否则输出套路话术
    else:
        return "这个问题我要问问领导"


def launch_gradio():
    demo = gr.ChatInterface(
        fn=sales_chat,
        title="修炼家的橘子屋",
        # retry_btn=None,
        # undo_btn=None,
        chatbot=gr.Chatbot(height=600),
    )

    demo.launch(share=True)


if __name__ == "__main__":
    # 初始化橘子销售机器人sales_bot
    initialize_sales_bot()
    # 启动 Gradio 服务
    launch_gradio()
