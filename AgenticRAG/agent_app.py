import streamlit as st
from RAG_chat_bot import ChatBot

def chat(query) : 
    agent = ChatBot()
    resp = agent.query(query)
    return resp.response


st.title("DeepSolv RAG task")


def show_ui(prompt_to_user="How may I help you?"):
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "You are an expert in data analysis ", "content": prompt_to_user}]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User-provided prompt
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chat(prompt)
                st.markdown(response)
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)

def run():
    show_ui("What would you like to know?")

run()