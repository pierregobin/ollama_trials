#!/usr/bin/env python3

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

# -----------------------
# CONFIG
# -----------------------
PDF_PATH = "GORIUS_Jean-michel.pdf"  # <-- change this
MODEL_NAME = "phi3"

# -----------------------
# 1. Load PDF
# -----------------------
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

# -----------------------
# 2. Split text
# -----------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,   # smaller for CPU
    chunk_overlap=100
)
chunks = splitter.split_documents(documents)

# -----------------------
# 3. Create embeddings (local)
# -----------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------
# 4. Vector store
# -----------------------
vectorstore = FAISS.from_documents(chunks, embeddings)

# -----------------------
# 5. Local LLM (Phi-3)
# -----------------------
llm = Ollama(
    model=MODEL_NAME,
    num_ctx=2048
)

# -----------------------
# 6. Retrieval QA chain
# -----------------------
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)

# -----------------------
# 7. Interactive Q&A
# -----------------------
print("\nPDF loaded. Ask questions (type 'exit' to quit).")

while True:
    query = input("\nQuestion: ")
    if query.lower() == "exit":
        break

    result = qa(query)

    print("\nAnswer:\n", result["result"])

    print("\nSources:")
    for doc in result["source_documents"]:
        print(
            f"- Page {doc.metadata.get('page')} | "
            f"{doc.metadata.get('source')}"
        )

