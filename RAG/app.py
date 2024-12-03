import warnings
warnings.filterwarnings("ignore")

import streamlit as st

import RAG.RAG_pipeline as rp

from langchain_core.messages import AIMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

chunk_type = "Recursive"
chunk_size = 800
chunk_overlap = 100

SESSION_ID = "1234"
B_INST, E_INST = "<s>[INST]", "[/INST]</s>"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"



def context_aware_chain(retriever):
    llm_sum = rp.initialize_llm(temperature = 0.0)

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is.")

    hist_aware_retr = rp.create_rephrase_history_chain(llm_sum, retriever, contextualize_q_system_prompt)

    return hist_aware_retr


def ans_chain (retriever):
    # we increase the temperature here as we dont want the model to directly return the retrieved documents but return some creative answers 
    llm_ans = rp.initialize_llm(temperature = 0.5)

    system_prompt = (
        "You are an expert in Star Wars movies and characters who answers questions of new fans of the saga. "
        "You will be given some information extracted from Wikipedia that can be useful to answer the question, use them how you want. "
        "If you don't know the answer, say \'The information given are not enough to answer to the question\'. "
        "Use three sentences maximum and keep the answer concise."
        "\n\n"
        "ADDITIONAL INFORMATION: {context}")

    full_rag_chain_with_history = rp.create_qa_RAG_chain_history(llm_ans, retriever, system_prompt)

    return full_rag_chain_with_history


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.chat_store:
        st.session_state.chat_store[session_id] = ChatMessageHistory()
    return st.session_state.chat_store[session_id]


def get_response(full_chain, session_id, user_query):
    """
    Prompts the llm and returns the answer to the user_query using a full chain considering the history as well.
    """
    answer = full_chain.invoke({"input": user_query}, config={
        "configurable": {"session_id": session_id}
    })

    return answer["answer"]

def folder_added():
    with st.spinner("Creating vector retriever, please wait..."):
        # st.session_state.disabled_sum_model = False
        st.session_state.retriever = rp.initialize_retriever(st.session_state.files_folder)

def update_summarization_model():
    # st.session_state.disabled_ans_model = False
    st.session_state.retriever_chain = context_aware_chain(st.session_state.retriever)

def update_answer_model():
    # model_name = st.session_state.ans_model
    # ollama_model_name = re.search("(.*)  Size:", model_name).group(1)

    st.session_state.retriever_answer_chain = ans_chain (st.session_state.retriever_chain)

    st.session_state.final_chain = RunnableWithMessageHistory(
        st.session_state.retriever_answer_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

st.set_page_config(page_title="Chat with a Star Wars expert", page_icon="🦜")
st.title("_Chat_ with :blue[A Star Wars Expert] 🤖")

# # Initially, the dropdown menus for the answering and summarizarion model are disabled.
# if "disabled_ans_model" not in st.session_state:
#     st.session_state.disabled_ans_model = True

# if "disabled_sum_model" not in st.session_state:
#     st.session_state.disabled_sum_model = True

# Instantiates a sidebar
with st.sidebar:
    # # Returns the list of ollama models available in ollama on the device
    # models_ollama = ollama.list()["models"]
    # # extract name and size of the model (in GB)
    # model_name = [m['name'] for m in models_ollama]
    # model_size = [float(m["size"]) for m in models_ollama]
    # name_detail = zip(model_name, model_size)
    # # Sort the models based on their size, in ascending order. Faster (smaller models) first
    # name_detail = sorted(name_detail, key=lambda x: x[1])
    # model_info = [f"{name}  Size: {size/(10**9):.2f}GB" for name, size in name_detail]

    model_info = ["mixtral-8x7b-32768"]
    st.text_input("Insert the folder containing the .html files to be used for RAG", on_change=folder_added, key="files_folder")
    # if st.button("Start"):
    #     print("done")
    #     update_summarization_model()
    #     update_answer_model()
    
    # st.selectbox("Choose a model for context summarization", model_info, index=None, on_change=update_summarization_model, placeholder="Select model", key="sum_model")
    # st.selectbox("Choose a model for answering", model_info, index=None, on_change=update_answer_model, placeholder="Select model", key="ans_model")    
    st.selectbox("Choose a model for context summarization", model_info, index=None, on_change=update_summarization_model, placeholder="Select model", key="sum_model")
    st.selectbox("Choose a model for answering", model_info, index=None, on_change=update_answer_model, placeholder="Select model", key="ans_model")
    # st.caption("You will see here the models downloaded from Ollama. For installing ollama: https://ollama.com/download.  \nFor the models available: https://ollama.com/library.  \n⚠️ Remember: Heavily quantized models will perform slightly worse but much faster.")


if "disabled" not in st.session_state:
    st.session_state.disabled = False

if "chat_store" not in st.session_state:
    st.session_state.chat_store = {}
def disable():
    st.session_state.disabled = True
def enable():
    st.session_state.disabled = False

if SESSION_ID not in st.session_state.chat_store:
    st.session_state.chat_store[SESSION_ID] = ChatMessageHistory()

for message in st.session_state.chat_store[SESSION_ID].messages:
    MESSAGE_TYPE = "AI" if isinstance(message, AIMessage) else "Human"
    if isinstance(message, AIMessage):
        prefix = "AI"
    else:
        prefix = "Human"

    with st.chat_message(MESSAGE_TYPE):
        st.write(message.content)

if user_query := st.chat_input("Type your message ✍", disabled=st.session_state.disabled, on_submit=disable):
    # Returns an error if any of the fields on the left is unfilled
    # if "retriever" not in st.session_state:
    #     st.error("Retriever, summarization model and answering model were not set.")
    # elif "retriever_chain" not in st.session_state:
    #     st.error("Summarization model and answering model were not set.")
    # elif "retriever_answer_chain" not in st.session_state:
    #     st.error("Answering model was not set.")
    # If all is good, then a spinner will be shown on screen telling the answer in being generated and the chat input will be disabled until the generation is done.
    if user_query is not None and user_query != "":
        with st.spinner("The model is generating an answer, please wait"):
            response = get_response(st.session_state.final_chain, SESSION_ID, user_query)
            st.session_state.disabled = False
            st.rerun()