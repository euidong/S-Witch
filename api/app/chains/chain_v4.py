#! python

# For getting each device's CLI command.
# 1. Prompt를 구조화한다.
# 1-1. chatting history를 만든다.
# 1-2. question은 그대로 유지한다.
# 1-3. topology는 그대로 유지하며, 몇 개의 device가 존재하는지 확인한다.
# 1-4. Device의 수만큼 Command를 생성하기 위해서 그 수 만큼의 Prompt를 생성한다.
# 2. 모델에 각 Prompt를 전달한다.
# 3. Output을 구조화한다.
# 3-1. Json Array 형태로 각 device의 command를 저장한다.
from operator import itemgetter
from typing import List, Tuple

from langchain.llms.openai import OpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import format_document
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda

from dotenv import load_dotenv; load_dotenv()

from api.app.utils import output_parser, network_topology_parser

_TEMPLATE_FOR_ALLOCATE_IP = """I have the following Network Topology.

- Network Topology: {network_topology}

And, the following information was exchanged through previous chatting.

- Previous Chatting History: {chat_history}

The problems we are currently trying to solve are as follows.

- Current question: {question}

Can you allocate IP addresses and subnet masks to each device's port? and, add this information to the "Network Topology" formatted with JSON?

{format_instructions}

- Network Topology:"""

PROMPT_FOR_ALLOCATE_IP = PromptTemplate.from_template(
    _TEMPLATE_FOR_ALLOCATE_IP,
    partial_variables={
        "format_instructions": network_topology_parser.get_format_instructions(),
    }
)


_TEMPLATE_FOR_ANSWER = """I have the following Network Topology.

- Network Topology: {network_topology}

And, the following information was exchanged through previous chatting.

- Previous Chatting History: {chat_history}

The problems we are currently trying to solve are as follows.

- Current question: {question}

{format_instructions}

{output_examples}
"""

PROMPT_FOR_ANSWER = PromptTemplate.from_template(
    _TEMPLATE_FOR_ANSWER,
    partial_variables={
        "format_instructions": output_parser.get_format_instructions(),
        "output_examples": """Output Examles:

- {"device": "R1", "command": "enable\nconfigure terminal\ninterface GigabitEthernet0/0\nip address 192.168.0.1\nno shutdown\nexit\nexit\ncopy running-config startup-config\nexit", "comment": "set the ip address of Router1 the interface GigabitEthernet0/0"}
- {"device": "R2", "command": "enable\nconfigure terminal\nrouter ospf 1\nnetwork 192.168.0.4  0.0.255.255 area 0.0.0.0\nnetwork 192.168.0.3 0.0.0.0 area 23\nexit\nexit\ncopy running-config startup-config\nexit", "comment": "configure Router2's routing protocol with OSPF"}
- {"device": "PC1", "command": "ip 192.168.0.2 /24 192.168.0.1", "comment": "Configure PC1 ip address"}
- {"device": "R3", "command": "", "comment": "No command is required."}""",
    }
)

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

from langchain_community.chat_models import ChatOpenAI

chain = (
    {
        "network_topology": {
            "network_topology": itemgetter("topology"),
            "chat_history": itemgetter("chat_history") | RunnableLambda(_format_chat_history),
            "question": itemgetter("question"),
        } | PROMPT_FOR_ALLOCATE_IP | ChatOpenAI(model="gpt-3.5-turbo", temperature=0) | network_topology_parser,
        "chat_history": itemgetter("chat_history") | RunnableLambda(_format_chat_history),
        "question": itemgetter("question"),
    }
    | PROMPT_FOR_ANSWER 
    | ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    | output_parser
)
first_chain = (
    {
        "network_topology": itemgetter("topology"),
        "chat_history": itemgetter("chat_history") | RunnableLambda(_format_chat_history),
        "question": itemgetter("question"),
    }
    | PROMPT_FOR_ALLOCATE_IP 
    | ChatOpenAI(model="gpt-3.5-turbo", temperature=0) 
    | network_topology_parser
)

second_chain = (
    {
        "network_topology": itemgetter("topology"),
        "chat_history": itemgetter("chat_history") | RunnableLambda(_format_chat_history),
        "question": itemgetter("question"),
    }
    | PROMPT_FOR_ANSWER 
    | ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    | output_parser
)