from operator import itemgetter
from typing import List, Tuple

from langchain.llms.openai import OpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import format_document
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda

from dotenv import load_dotenv; load_dotenv()

from api.app.utils import output_parser

def load_retriever():
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores.faiss import FAISS
    
    model_name = "BAAI/bge-base-en-v1.5"
    model_kwargs = { "device": "cuda" }
    encode_kwargs = { "normalize_embeddings": True } # use cosine similarity
    embedding = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    f_search = "faiss"
    store_name = f"api/db/{f_search}"

    docsearch = FAISS.load_local(store_name, embeddings=embedding)

    retriever = docsearch.as_retriever()
    return retriever

_TEMPLATE_FOR_QUERY = """I collected several guide documents for configuring Cisco or Juniper switches,
In order for a language model-based retriever to extract meaningful documents, you must create the right query.
Please refer to the information below and create an appropriate query.

- Previous chatting history: {chat_history}
- Current question: {question}"""
PROMPT_FOR_QUERY = PromptTemplate.from_template(_TEMPLATE_FOR_QUERY)

_TEMPLATE_FOR_ANSWER = """I have the following Network Topology.

- Network Topology: {network_topology}

And, the following information was exchanged through previous chatting.

- Previous Chatting History: {chat_history}

The problems we are currently trying to solve are as follows.

- Current question: {question}

Below is a document that may help solve the problem.

- document : {document}

{format_instructions}
"""
PROMPT_FOR_ANSWER = PromptTemplate.from_template(
    _TEMPLATE_FOR_ANSWER,
    partial_variables={"format_instructions": output_parser.get_format_instructions()}
)

def _combine_documents(docs, document_separator="\n\n") -> str:
    doc_strings = [format_document(doc, PromptTemplate.from_template(template="{page_content}")) for doc in docs]
    return document_separator.join(doc_strings)

def _format_chat_history(chat_history: List[Tuple]) -> str:
    """Format chat history into a string."""
    buffer = ""
    for dialogue_turn in chat_history:
        human = "Human: " + dialogue_turn[0]
        ai = "Assistant: " + dialogue_turn[1]
        buffer += "\n" + "\n".join([human, ai])
    if buffer == "":
        buffer = "None"
    return buffer

retriever = load_retriever()

chain = (
    {
        "network_topology": itemgetter("topology"),
        "chat_history": itemgetter("chat_history") | RunnableLambda(_format_chat_history),
        "question": itemgetter("question"),
        "document": {
            "chat_history": itemgetter("chat_history") | RunnableLambda(_format_chat_history),
            "question": itemgetter("question"),
        } 
        | PROMPT_FOR_QUERY
        | OpenAI(model="gpt-3.5-turbo-instruct", temperature=0)
        | StrOutputParser()
        | retriever
        | _combine_documents,
    }
    | PROMPT_FOR_ANSWER 
    | OpenAI(model="gpt-3.5-turbo-instruct", temperature=0)
    | output_parser
)