from llama_index.readers.file.pymu_pdf.base import PyMuPDFReader
from pathlib import Path  

loader = PyMuPDFReader()

file_path = Path("/Users/aryansood/Github/deepsolv-task/DeepSolv-RAG-Task/data/Apple_Vision_Pro_Privacy_Overview.pdf")
files = [
    "data/Apple_Vision_Pro_Privacy_Overview.pdf",
    "data/apple-privacy-policy-en-ww.pdff",
    "data/law-enforcement-guidelines-us.pdf",
]


files_to_tools_dict = {}
for file in files:
    print(f"Getting tools for file: {file}")
    vector_tool, summary_tool = get_doc_tools(file, Path(file).stem)
    file_to_tools_dict[file] = [vector_tool, summary_tool]

# Load the data from the PDF
documents = loader.load_data(file_path=file_path, metadata=True)

print(documents)
