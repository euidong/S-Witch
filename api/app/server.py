

from fastapi import FastAPI
from langserve import add_routes
from fastapi.middleware.cors import CORSMiddleware

from api.app.utils import ChatRequest, ChatResponse, ChatRequestWrapper
from api.app.chains.chain_v1 import chain as chain_v1
from api.app.chains.chain_v2 import chain as chain_v2
from api.app.chains.chain_v3 import chain as chain_v3
from api.app.chains.chain_v4 import chain as chain_v4
from api.app.chains.chain_v5 import invoke as chain_v5_invoke

from api.app.chains.chain_v4 import first_chain as first_chain_v4, second_chain as second_chain_v4

from dotenv import load_dotenv; load_dotenv()

chain_v1 = chain_v1.with_types(input_type=ChatRequest, output_type=ChatResponse)
chain_v2 = chain_v2.with_types(input_type=ChatRequest, output_type=ChatResponse)
chain_v3 = chain_v3.with_types(input_type=ChatRequest, output_type=ChatResponse)
chain_v4 = chain_v4.with_types(input_type=ChatRequest)

app = FastAPI(
    title="S Witch API Server",
    version="0.1.0",
    description="S Witch API Server",
)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Adds routes to the app for using the chain under:
# /invoke
# /batch
# /stream
add_routes(app, chain_v1, path="/v1", enable_feedback_endpoint=True)
add_routes(app, chain_v2, path="/v2", enable_feedback_endpoint=True)
add_routes(app, chain_v3, path="/v3", enable_feedback_endpoint=True)

from langserve import APIHandler
api_handler = APIHandler(chain_v4, path="/v4")

from fastapi import Request, Response
import json
import asyncio


# ip 받기
# 이를 활용해서, 전체적으로 어떻게 설정해야할지 조언을 얻기
# 조언과 ip, topology에 기반하여 각 각의 device를 위한 설정 방법 얻기.
@app.post("/v4/invoke")
async def invoke_v4(request: ChatRequestWrapper) -> Response:
    request = request.input
    topology = json.loads(request.topology)
    node_info = topology["node_info"]
    async_list = []
    first_request = request.dict()
    first_output = first_chain_v4.invoke(first_request)
    request["topology"] = first_output["output"]
    for node in node_info:
        node_request = request.dict()
        node_request["question"] += f" for this purpose, how can I configure device {node['name']}?"
        node_response = second_chain_v4.ainvoke(node_request)
        async_list.append(node_response)
    finished, _ = await asyncio.wait(async_list, return_when=asyncio.ALL_COMPLETED)
    return { "output": [f.result() for f in finished] } 

@app.post("/v5/invoke")
async def invoke_v5(request: ChatRequestWrapper) -> Response:
    request = request.input
    output = await chain_v5_invoke(request)
    return { "output": output }

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)