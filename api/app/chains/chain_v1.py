
from operator import itemgetter
from typing import List, Tuple

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import format_document
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableLambda
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS

from dotenv import load_dotenv; load_dotenv()

from api.app.utils import output_parser

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


# _TEMPLATE = """Given the following conversation and a follow up question, rephrase the 
# follow up question to be a standalone question, in its original language.
# Chat History:{chat_history}

# Follow Up Input: {question}
# Standalone question:"""
_TEMPLATE = """Your are a network operator and you are asked to configure a network device. Check the following network topology formatted with JSON, and consider the following chat history. Then, rephrase the follow up question to be a standalone question, in its original language.

Network topology: {topology}

Chat History:{chat_history}

Follow Up Input: {question}
Standalone question:"""

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_TEMPLATE)

ANSWER_TEMPLATE = """You are an expert at configuring network opeartion.
Get context with network topology and suggest only CLI command for achieveing question's requirements. (Never include plantext answer)
{format_instructions}

Question: {question} 

Network topology: {topology}

Context: {context}

Answer:"""
ANSWER_PROMPT = ChatPromptTemplate.from_template(
    ANSWER_TEMPLATE,
    partial_variables={"format_instructions": output_parser.get_format_instructions()}
)

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")


def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    """Combine documents into a single string."""
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
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


chain = (
    RunnableParallel({
        "standalone_question": RunnableParallel({
            "chat_history": itemgetter("chat_history") | RunnableLambda(_format_chat_history),
            "question": itemgetter("question"),
            "topology": itemgetter("topology"),
        }) 
        | CONDENSE_QUESTION_PROMPT
        | ChatOpenAI(temperature=0)
        | StrOutputParser(),
        "topology": itemgetter("topology"),
    }) 
    | {
        "context": itemgetter("standalone_question") | retriever | _combine_documents,
        "question": itemgetter("standalone_question"),
        "topology": itemgetter("topology"),
    }
    | ANSWER_PROMPT
    | ChatOpenAI(temperature=0)
    | output_parser
)