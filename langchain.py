from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.llms import Ollama
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
import uvicorn


llm = Ollama(model="llama2")
loader = WebBaseLoader(["https://www.oscars.org/", "https://en.wikipedia.org/wiki/96th_Academy_Awards/","https://www.oscars.org/oscars/ceremonies/2024"])
docs = loader.load()
embeddings = OllamaEmbeddings()
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)
prompt = ChatPromptTemplate.from_template("""Answer the following question based onl8000y on the provided context:

<context>
{context}
</context>

Question: {input}""")

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)


app = FastAPI()

class Item(BaseModel):
    prompt: str

@app.post("/process")
async def process_prompt(item: Item):
    print(item.dict())
    return {"status": "received"}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=9001)