import os 

from pathlib import Path
from utils import get_doc_tools

from llama_index.core import VectorStoreIndex, Settings

from llama_index.core.agent import AgentRunner
from llama_index.core.objects import ObjectIndex
from llama_index.core.agent import FunctionCallingAgentWorker

from utils import initialize_settings

def ChatBot() : 
    files = [
        "data/document.pdf"
    ]
    initialize_settings()

    files_to_tools_dict = {}
    for file in files:
        print(f"Getting tools for file: {file}")
        vector_tool, summary_tool = get_doc_tools(file, Path(file).stem)
        files_to_tools_dict[file] = [vector_tool, summary_tool]
    print(files_to_tools_dict)

    llm = Settings.llm
    all_tools = [t for file in files for t in files_to_tools_dict[file]]

    # defining an "object" index and retriever over these tools
    obj_index = ObjectIndex.from_objects(
        all_tools,
        index_cls=VectorStoreIndex,
    )
    obj_retriever = obj_index.as_retriever(similarity_top_k=2)
    agent_worker = FunctionCallingAgentWorker.from_tools(
        tool_retriever=obj_retriever,
        llm=llm, 
        system_prompt=
        """
        You are an intelligent assistant specializing in retrieving and answering queries based on provided documents. 
        Use the retrieved context to generate accurate and concise answers. 
        If the information is not in the document, explicitly state, "The document does not provide this information." 
        Ensure all answers are derived only from the retrieved content and are highly relevant to the query.
        Example Queries:

        **Query** : What is the e-Tender notice number and its purpose?
        **Answer** : The e-Tender notice number is NITJ/DRC/PUR/TT/36/2024. Its purpose is the fabrication of a machine for continuous production of textile waste-based composite materials for the Department of Textile Technology.
        
        **Query** : The technical bid is evaluated to ensure compliance with essential eligibility criteria, submission of EMD and Tender Fee, completion of required documents, adherence to equipment specifications, and validity of service and warranty policies. Only technically qualified bids proceed to the financial evaluation stage.
        """,
        verbose=False
    )
    agent = AgentRunner(agent_worker)
    return agent


def chat(query) : 
    agent = ChatBot()
    resp = agent.query(query)
    return resp.response


def test(agent,query):
    resp = agent.query(query)
    return resp.response


def main() : 
    queries = [
        "What is the e-Tender notice number and the purpose of the tender mentioned in the document?",
        "What are the eligibility criteria for bidders to participate in this tender?",
        "What are the deadlines for submitting the online bids and physically submitting the tender fee and EMD?",
        "What is the role of Annexure-G in determining supplier eligibility, and how is local content defined?",
        "What is the payment structure for the successful supplier as mentioned in the document?",
        "What are the warranty obligations for suppliers as outlined in Annexure-F?",
        "How is the technical bid evaluated, and what criteria are used for shortlisting bidders?",
        "What penalties are imposed for delays in delivery or non-performance by the supplier?",
        "What does Annexure-E specify about the blacklisting or debarment of suppliers?",
        "What does Annexure-D require from suppliers regarding manufacturer authorization?"
    ]
    responses = {}
    agent = ChatBot()
    for query in queries : 
        response = test(agent, query)
        responses[query] = response
        print(f"Q) {query}")
        print(f"A) {response}")


if __name__ == "__main__": 
    main()