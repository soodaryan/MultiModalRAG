import streamlit as st
from RAG_chat_bot import ChatBot

def chat(query) : 
    """Agent caller. Returns the ouput to Query by calling the RAG agent."""
    agent = ChatBot()
    resp = agent.query(query)
    return resp.response

def show_ui(prompt_to_user="How may I help you?"):

    """The function for Streamlit"""

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "You are an expert in data analysis ", "content": prompt_to_user}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chat(prompt)
                st.markdown(response)
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)

def run():
    
    """Wrapper Function to start UI"""
    
    show_ui("What would you like to know?")


st.title("DeepSolv RAG task")
run()