import uuid
from langchain_chroma import Chroma
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_huggingface import HuggingFaceEmbeddings

def create_retriever (id_key = "doc_id") :

    # vectorstore for child chunks 
    vectorstore = Chroma(
        collection_name="multi_modal_rag",
        embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )

    # storage layer for parent docs
    store = InMemoryStore()

    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )
    return id_key, retriever

def makeVectoreStore (texts, 
                      tables, 
                      images,
                      text_summaries,
                      table_summaries,
                      image_summaries
                      ) : 

    id_key, retriever = create_retriever()

    doc_ids = [str(uuid.uuid4()) for _ in texts]
    summary_texts = [
        Document(page_content=summary, metadata={id_key: doc_ids[i]}) for i, summary in enumerate(text_summaries)
    ]
    retriever.vectorstore.add_documents(summary_texts)
    retriever.docstore.mset(list(zip(doc_ids, texts)))

    # Add tables
    table_ids = [str(uuid.uuid4()) for _ in tables]
    summary_tables = [
        Document(page_content=summary, metadata={id_key: table_ids[i]}) for i, summary in enumerate(table_summaries)
    ]
    retriever.vectorstore.add_documents(summary_tables)
    retriever.docstore.mset(list(zip(table_ids, tables)))

    # Add image summaries
    img_ids = [str(uuid.uuid4()) for _ in images]
    summary_img = [
        Document(page_content=summary, metadata={id_key: img_ids[i]}) for i, summary in enumerate(image_summaries)
    ]
    retriever.vectorstore.add_documents(summary_img)
    retriever.docstore.mset(list(zip(img_ids, images)))
    return retriever


if __name__ == "__main__" : 
    # importing this here as its not needed anywhere else so shudnt be imported unneccesarily
    from DataExtraction.dataPreprocessing import text_table_img

    path = 'data/document.pdf'
    
    texts, tables, images, text_summaries, table_summaries, image_summaries = text_table_img(path)
    retriever = makeVectoreStore (texts, tables, images, text_summaries, table_summaries, image_summaries) 
    docs = retriever.invoke( "what is the document about?")
    print (docs)