# 🚀 Document Processing with RAG - CE TASK

Hi there! 👋 This is my implementation of RAG-based pipeline to extract structured information from the given document. I explored two different workflows—Agentic and Multimodal—to tackle this challenge, and I’m excited to share my work with you! 😊

---

## ✨ What I Did  

### 🧠 **Agentic Workflow (Workflow 1)**  
In this workflow, I built an **agentic pipeline** using LlamaIndex. Here’s how it works:  
- I gave the agent two tools:  
  - **Vector Indexer**: Breaks the document into chunks and creates a vector representation for retrieval.  
  - **Summarizer**: Summarizes the chunks for easier comprehension.  
- I combined these tools using the **object retriever** function, enabling efficient query-based information retrieval.  

### 🌟 **Multimodal Workflow (Workflow 2)**  
In this pipeline I focused on **data preprocessing** to handle multimodal content:  
- I used the **Unstructured** library to extract:  
  - Tables 📊  
  - Text 📝  
  - Images 🖼️  
- For images, I used the **BLIP** model to generate captions.  
- I summarized the text and tables and then created a **vector index** to map the summaries back to their original nodes.  
- To make it even better, I implemented **title-wise chunking** to preserve context in the nodes, which made the retrieval much more accurate.  

---

## 🧐 Why I Prefer the Multimodal Workflow  
While both workflows are solid, I found the **Multimodal Workflow** has a clear edge because:  
1. It handles **multimodal data** (images, tables, text) effortlessly.  
2. Better preprocessing leads to more accurate and meaningful retrieval.  

---

## 🔧 How to Set It Up  

1. Clone the repo:  
   ```bash  
   git clone https://github.com/soodaryan/MultiModalRAG
   cd MultiModalRAG  
   ```  

2. Install the dependencies:  
   ```bash  
   pip install -r requirements.txt  
   ```  

3. Download the LLM models:  
   I used **Groq LLMs** because they’re free and work great even on CPU-only systems. 

---

## 🚀 How to Run  

1. **Agentic Workflow**:  
   ```bash  
   python3 AgenticRAG/RAG_chat_bot.py
   ```  

2. **Multimodal Workflow**:  
   ```bash  
   python3 -m RAG.RAG_pipeline
   ``` 

---

## 💡 Some Observations  

- **Agentic Workflow**: Best for text-heavy documents and straightforward use cases.  
- **Multimodal Workflow**: Handles complex documents with tables, images, and text like a pro! 🙌 Title-wise chunking really improved the results here.  

---

📁 Sample Outputs
Check the examples/ directory for outputs from both workflows.



## 🏗️ Architecture Overview  

1. **Agentic Workflow**:  
   - Uses LlamaIndex tools (Vector Indexer and Summarizer) with an agentic structure.  

2. **Multimodal Workflow**:  
   - Processes text, tables, and images separately.  
   - Maps summaries back to original nodes with title-wise chunking for better context retention.  

---

## 🤔 Why I Used Groq LLMs  
I went with **Groq LLMs** because:  
- They’re free! 🎉  
- They work seamlessly on CPU-only systems, keeping things scalable and efficient.  

--- 

## 🚀 Future Scopes 
- I am currently working on a pipeline for RAG evaluation to quantitively evaluate bot the workflows
- We can integrate the multimodal part into a basic multiagentic workflow where in we can have different agent specialized in differnt domains.

--- 
Feel free to dive in, raise issues, or suggest improvements. Let’s make document processing smarter together! 💪✨  
