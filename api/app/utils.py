from typing import List, Tuple
from langserve.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

# User input
class ChatRequest(BaseModel):
    """Chat history with the bot."""
    chat_history: List[Tuple[str, str]] = Field(
        ...,
        extra={"widget": {"type": "chat", "input": "question"}},
    )
    topology: str
    question: str

class ChatRequestWrapper(BaseModel):
    input: ChatRequest
    config: dict = Field(
        {},
        description="Additional configuration for the chain.",
    )
    kwargs: dict = Field(
        {},
        description="Additional keyword arguments for the chain.",
    )

# User output
class ChatResponse(BaseModel):
    """Chat response from the bot."""

    device: str = Field(description="The device name.")
    command: str = Field(description="Each device's CLI Command to solve the user's problem. It cannot contain any content other than CLI commands.")
    comment: str = Field(description="Description of the command.")

# Set up a parser + inject instructions into the prompt template.
output_parser = JsonOutputParser(pydantic_object=ChatResponse)

class PortIdentification(BaseModel):
    """Port identification."""
    device: str = Field(description="The device name.")
    port: str = Field(description="The port name.")
    ip: str = Field(description="The ip address.")
    subnet: str = Field(description="The subnet mask.")

class PortIdentificationWrapper(BaseModel):
    port_info: List[PortIdentification] = Field(description="The result of port identification.")

# Set up a parser + inject instructions into the prompt template.
port_identification_parser = JsonOutputParser(pydantic_object=PortIdentificationWrapper)

class NetworkTopology(BaseModel):
    """Network topology."""
    node_info: List[dict] = Field(
        ...,
        description="The information of each device.",
    )
    link_info: List[dict] = Field(
        ...,
        description="The information of each link.",
    )

network_topology_parser = JsonOutputParser(pydantic_object=NetworkTopology)