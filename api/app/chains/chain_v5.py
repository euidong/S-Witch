# 0. setup
from dotenv import load_dotenv; load_dotenv()

# 1. Create Template

INSTRUCTION = """You are a network operator, and I will request network configuration from you through chat. 
In the request process, I will convey the network topology, the content of the conversation so far, and my requirements. 
To process the request, you will follow the process in four steps.

- In the first stage, the previous conversation is received as input and goes through a process of summarizing it. If there is no conversation content, the step is skipped.
- In the second stage, For a given network topology, provide an overall design(blueprint) that can meet the requirements without adding links or new devices. For example, determine routing protocol, Access List, VLAN allocation, etc.
- In step 3, the task of assigning an IP address and subnet mask to every connected port of the topology is performed according to the overall design.
- In step 4, the task of creating a CLI command for each device is performed.
"""


FIRST_STEP_TEMPLATE = INSTRUCTION + '\n' + """You are currently in stage 1.

If the content of a previous conversation is as follows, summarize the content in one paragraph.

- Previous conversation: {prev_conversation}"""

SECOND_STEP_TEMPLATE = INSTRUCTION + '\n' + """You are currently in stage 2.

A summary of the previous conversation follows:

- Summary of previous conversation: {prev_conversation_summary}

The current network topology is as follows.

- network topology : {network_topology}

Currently my requirements are:

- Current conversation content: {question}

At this time, please suggest how to design the overall network."""

THIRD_STEP_TEMPLATE = INSTRUCTION + '\n' + """You are currently in stage 3.

Details on the overall network design are as follows.

- Network design: {design_of_network}

The current network topology is as follows.

- network topology : {network_topology}

Assign an appropriate IP address and subnet mask to the every connected port of each device.

You should consider the following points.

```
When setting an IP (Internet Protocol) address and subnet mask for a device in a network, it's important to consider several factors to ensure proper network configuration and performance. Here are some key points to keep in mind:

1. Network Size and Structure: Understand the size of your network and how many devices will be connected. This will influence how you design your IP addressing scheme.
Choose a subnet mask that provides an appropriate number of host addresses for your network. A subnet with too few addresses can limit growth, while too many can waste address space.

2. Subnetting and CIDR: Consider using subnetting or Classless Inter-Domain Routing (CIDR) to divide your larger network into smaller, more manageable subnets. This can improve performance, security, and network management.
Make sure the subnet mask reflects the subnetting scheme, with the correct number of bits allocated for the network and host portions of the address.

3. Avoiding IP Conflicts: Ensure that each device on your network has a unique IP address within the subnet. IP conflicts can lead to network connectivity issues.
If using DHCP, configure it correctly to avoid assigning an IP address that is already in use as a static IP.

4. Network Hardware and Topology: Understand how your network devices (routers, switches, etc.) use IP addresses and subnet masks. Incorrect settings can lead to routing issues or isolation of parts of your network.
Consider how your network topology (the arrangement and connection of network devices) influences your IP addressing scheme.
```

{format_instructions}"""

FOURTH_STEP_TEMPLATE = INSTRUCTION + '\n' + """You are currently in stage 4.

Details on the overall network design are as follows.

- Network design: {design_of_network}

The current network topology is as follows.

- network topology: {network_topology}
- ip address / subnet mask : {port_identification}

The node_name that you need to configure is {device_name}.
Create a CLI command according to the above design. In this step, you should assign the ip address and subnet mask to the every port of the device.

{format_instructions}

{example_command}"""

OUTPUT_EXAMPLE_COMMAND = """Below are examples of creation.

- {"device": "R1", "command": "enable\nconfigure terminal\ninterface GigabitEthernet1/0\nip address 192.168.1.1 255.255.255.0\nno shutdown\ninterface GigabitEthernet2/0\nip address 192.168.2.1 255.255.255.0\nno shutdown\nexit\nrouter ospf 1\nnetwork 192.168.1.0 0.0.0.255 area 0\nnetwork 192.168.2.0 0.0.0.255 area 0\nexit\nexit\ncopy running-config startup-config\nexit", " comment": "set the ip address of Router1 the interface GigabitEthernet0/1, 0/2 and configure OSPF routing."}
- {"device": "PC1", "command": "ip 192.168.1.2 /24 192.168.1.1", "comment": "configure PC1 ip address and default gateway address"}
- {"device": "PC2", "command": "ip 192.168.2.2 /24 192.168.2.1", "comment": "configure PC2 ip address and default gateway address"}
- {"device": "R3", "command": "", "comment": "No command is required."}"""


# 2. Create Prompt
from langchain.prompts.prompt import PromptTemplate
from api.app.utils import output_parser, port_identification_parser

first_step_prompt = PromptTemplate.from_template(FIRST_STEP_TEMPLATE)
second_step_prompt = PromptTemplate.from_template(SECOND_STEP_TEMPLATE)
third_step_prompt = PromptTemplate.from_template(THIRD_STEP_TEMPLATE, partial_variables={"format_instructions": port_identification_parser.get_format_instructions()})
forth_step_prompt = PromptTemplate.from_template(FOURTH_STEP_TEMPLATE, partial_variables={"format_instructions": output_parser.get_format_instructions(), "example_command": OUTPUT_EXAMPLE_COMMAND})

# 3. Create Chain
from langchain_community.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser

first_step_chain = first_step_prompt | ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0) | StrOutputParser()
second_step_chain = second_step_prompt | ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0) | StrOutputParser()
third_step_chain = third_step_prompt | ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0) | port_identification_parser
fourth_step_chain = forth_step_prompt | ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0) | output_parser


## 4. Create Invoke Function

import asyncio
import json
from api.app.utils import ChatRequest

async def invoke(input: ChatRequest):
    if len(input.chat_history) != 0:
        prev_conversation_summary = first_step_chain.invoke({ "prev_conversation": input.chat_history })
    else:
        prev_conversation_summary = "It dose not exist."
    input.topology = json.loads(input.topology)
    design_of_network = second_step_chain.invoke({ "prev_conversation_summary": prev_conversation_summary, "network_topology": input.topology, "question": input.question })
    port_identification = third_step_chain.invoke({ "design_of_network": design_of_network, "network_topology": input.topology })
    
    node_info = input.topology["node_info"]
    async_list = []
    for node in node_info:
        # if node["node_type"] == "ethernet_switch":
        #     continue
        node_response = fourth_step_chain.ainvoke({ "design_of_network": design_of_network, "network_topology": input.topology, "port_identification": port_identification, "device_name": node["node_name"] })
        async_list.append(node_response)
    finished, _ = await asyncio.wait(async_list, return_when=asyncio.ALL_COMPLETED)
    print([f.result() for f in finished])
    return [f.result() for f in finished]

# 5. Test

DUMMY_INPUT_1 = ChatRequest(
    chat_history = [],
    topology = json.dumps({"node_info":[{"node_id":"42c32716-bd61-4fb2-a731-21852caa657b","name":"PC1","node_type":"vpcs","ports":[{"name":"Ethernet0","port_number":0,"link_type":"ethernet"}]},{"node_id":"36221bf8-e63c-4deb-b026-f95c7ae49ba2","name":"PC2","node_type":"vpcs","ports":[{"name":"Ethernet0","port_number":0,"link_type":"ethernet"}]},{"node_id":"8c4ca0f9-55a2-4517-a6c7-e89b8519b38a","name":"PC3","node_type":"vpcs","ports":[{"name":"Ethernet0","port_number":0,"link_type":"ethernet"}]},{"node_id":"4e78a684-6c7b-485a-9c62-36ecc9c026b6","name":"PC4","node_type":"vpcs","ports":[{"name":"Ethernet0","port_number":0,"link_type":"ethernet"}]},{"node_id":"43307983-5526-49ad-885a-e52e93f20a4e","name":"PC5","node_type":"vpcs","ports":[{"name":"Ethernet0","port_number":0,"link_type":"ethernet"}]},{"node_id":"087f3f0a-4605-4893-bd6a-9be111f04be2","name":"R1","node_type":"dynamips","ports":[{"name":"FastEthernet0/0","port_number":0,"link_type":"ethernet"},{"name":"GigabitEthernet1/0","port_number":0,"link_type":"ethernet"},{"name":"GigabitEthernet2/0","port_number":0,"link_type":"ethernet"},{"name":"GigabitEthernet3/0","port_number":0,"link_type":"ethernet"},{"name":"GigabitEthernet4/0","port_number":0,"link_type":"ethernet"},{"name":"GigabitEthernet5/0","port_number":0,"link_type":"ethernet"},{"name":"GigabitEthernet6/0","port_number":0,"link_type":"ethernet"}]},{"node_id":"fcee6830-3517-4954-bf6f-5d1847ea4734","name":"R2","node_type":"dynamips","ports":[{"name":"FastEthernet0/0","port_number":0,"link_type":"ethernet"},{"name":"GigabitEthernet1/0","port_number":0,"link_type":"ethernet"},{"name":"GigabitEthernet2/0","port_number":0,"link_type":"ethernet"},{"name":"GigabitEthernet3/0","port_number":0,"link_type":"ethernet"},{"name":"GigabitEthernet4/0","port_number":0,"link_type":"ethernet"},{"name":"GigabitEthernet5/0","port_number":0,"link_type":"ethernet"},{"name":"GigabitEthernet6/0","port_number":0,"link_type":"ethernet"}]},{"node_id":"857eaa69-d763-46be-9d76-b4d5135e1da0","name":"R3","node_type":"dynamips","ports":[{"name":"FastEthernet0/0","port_number":0,"link_type":"ethernet"},{"name":"GigabitEthernet1/0","port_number":0,"link_type":"ethernet"},{"name":"GigabitEthernet2/0","port_number":0,"link_type":"ethernet"},{"name":"GigabitEthernet3/0","port_number":0,"link_type":"ethernet"},{"name":"GigabitEthernet4/0","port_number":0,"link_type":"ethernet"},{"name":"GigabitEthernet5/0","port_number":0,"link_type":"ethernet"},{"name":"GigabitEthernet6/0","port_number":0,"link_type":"ethernet"}]},{"node_id":"d6af792d-ac6f-4d8d-9d09-25695ce052dc","name":"Switch1","node_type":"ethernet_switch","ports":[{"name":"Ethernet0","port_number":0,"link_type":"ethernet"},{"name":"Ethernet1","port_number":1,"link_type":"ethernet"},{"name":"Ethernet2","port_number":2,"link_type":"ethernet"},{"name":"Ethernet3","port_number":3,"link_type":"ethernet"},{"name":"Ethernet4","port_number":4,"link_type":"ethernet"},{"name":"Ethernet5","port_number":5,"link_type":"ethernet"},{"name":"Ethernet6","port_number":6,"link_type":"ethernet"},{"name":"Ethernet7","port_number":7,"link_type":"ethernet"}]},{"node_id":"78f4a9d4-4e22-4bf8-a2b4-ee4d956a3f68","name":"Switch2","node_type":"ethernet_switch","ports":[{"name":"Ethernet0","port_number":0,"link_type":"ethernet"},{"name":"Ethernet1","port_number":1,"link_type":"ethernet"},{"name":"Ethernet2","port_number":2,"link_type":"ethernet"},{"name":"Ethernet3","port_number":3,"link_type":"ethernet"},{"name":"Ethernet4","port_number":4,"link_type":"ethernet"},{"name":"Ethernet5","port_number":5,"link_type":"ethernet"},{"name":"Ethernet6","port_number":6,"link_type":"ethernet"},{"name":"Ethernet7","port_number":7,"link_type":"ethernet"}]}],"link_info":[{"link_id":"0158c60f-af87-4ac2-a860-0f3f0204dad0","link_type":"ethernet","nodes":[{"node_id":"087f3f0a-4605-4893-bd6a-9be111f04be2","port_number":0},{"node_id":"fcee6830-3517-4954-bf6f-5d1847ea4734","port_number":0}]},{"link_id":"be91c023-200a-47d2-b780-020bbae18516","link_type":"ethernet","nodes":[{"node_id":"087f3f0a-4605-4893-bd6a-9be111f04be2","port_number":0},{"node_id":"857eaa69-d763-46be-9d76-b4d5135e1da0","port_number":0}]},{"link_id":"c2b75669-f4ca-4ac5-947d-394717622c42","link_type":"ethernet","nodes":[{"node_id":"857eaa69-d763-46be-9d76-b4d5135e1da0","port_number":0},{"node_id":"fcee6830-3517-4954-bf6f-5d1847ea4734","port_number":0}]},{"link_id":"27e0d4a2-beca-4e2e-b588-1a48a37be8e9","link_type":"ethernet","nodes":[{"node_id":"8c4ca0f9-55a2-4517-a6c7-e89b8519b38a","port_number":0},{"node_id":"fcee6830-3517-4954-bf6f-5d1847ea4734","port_number":0}]},{"link_id":"27a20665-f4bb-4559-903d-b96a18e2f85f","link_type":"ethernet","nodes":[{"node_id":"857eaa69-d763-46be-9d76-b4d5135e1da0","port_number":0},{"node_id":"78f4a9d4-4e22-4bf8-a2b4-ee4d956a3f68","port_number":0}]},{"link_id":"c9d5315f-4324-4940-a844-fe2d47e69bd2","link_type":"ethernet","nodes":[{"node_id":"087f3f0a-4605-4893-bd6a-9be111f04be2","port_number":0},{"node_id":"d6af792d-ac6f-4d8d-9d09-25695ce052dc","port_number":0}]},{"link_id":"3b939529-0d64-46d3-ba80-45647ea084b6","link_type":"ethernet","nodes":[{"node_id":"42c32716-bd61-4fb2-a731-21852caa657b","port_number":0},{"node_id":"d6af792d-ac6f-4d8d-9d09-25695ce052dc","port_number":1}]},{"link_id":"4b56ad8a-6010-4303-b13c-523aa4dddb6c","link_type":"ethernet","nodes":[{"node_id":"36221bf8-e63c-4deb-b026-f95c7ae49ba2","port_number":0},{"node_id":"d6af792d-ac6f-4d8d-9d09-25695ce052dc","port_number":2}]},{"link_id":"0134633c-506f-4486-9c0f-33bab2156da5","link_type":"ethernet","nodes":[{"node_id":"4e78a684-6c7b-485a-9c62-36ecc9c026b6","port_number":0},{"node_id":"78f4a9d4-4e22-4bf8-a2b4-ee4d956a3f68","port_number":1}]},{"link_id":"926ac61b-2b4b-4eb6-8cbc-d7bea56fdda9","link_type":"ethernet","nodes":[{"node_id":"43307983-5526-49ad-885a-e52e93f20a4e","port_number":0},{"node_id":"78f4a9d4-4e22-4bf8-a2b4-ee4d956a3f68","port_number":2}]}]}),
    question = "connect PC1 to PC2.",
)

DUMMY_INPUT_2 = ChatRequest(
    chat_history = [["connect PC1 to PC2.", "Overall network design: configure PC1 and PC2's ip address and subnet mask.\n\nDevice: PC1\nCommand: ip 192.168.0.1 /24\nComment: set up ip address and subnet mask\n\nDevice: PC2\nCommand: ip 192.168.0.2\nComment: set up ip address and subnet mask"]],
    topology = json.dumps({"node_info":[{"node_id":"42c32716-bd61-4fb2-a731-21852caa657b","name":"PC1","node_type":"vpcs","ports":[{"name":"Ethernet0","port_number":0,"link_type":"ethernet"}]},{"node_id":"36221bf8-e63c-4deb-b026-f95c7ae49ba2","name":"PC2","node_type":"vpcs","ports":[{"name":"Ethernet0","port_number":0,"link_type":"ethernet"}]},{"node_id":"8c4ca0f9-55a2-4517-a6c7-e89b8519b38a","name":"PC3","node_type":"vpcs","ports":[{"name":"Ethernet0","port_number":0,"link_type":"ethernet"}]},{"node_id":"4e78a684-6c7b-485a-9c62-36ecc9c026b6","name":"PC4","node_type":"vpcs","ports":[{"name":"Ethernet0","port_number":0,"link_type":"ethernet"}]},{"node_id":"43307983-5526-49ad-885a-e52e93f20a4e","name":"PC5","node_type":"vpcs","ports":[{"name":"Ethernet0","port_number":0,"link_type":"ethernet"}]},{"node_id":"087f3f0a-4605-4893-bd6a-9be111f04be2","name":"R1","node_type":"dynamips","ports":[{"name":"FastEthernet0/0","port_number":0,"link_type":"ethernet"},{"name":"GigabitEthernet1/0","port_number":0,"link_type":"ethernet"},{"name":"GigabitEthernet2/0","port_number":0,"link_type":"ethernet"},{"name":"GigabitEthernet3/0","port_number":0,"link_type":"ethernet"},{"name":"GigabitEthernet4/0","port_number":0,"link_type":"ethernet"},{"name":"GigabitEthernet5/0","port_number":0,"link_type":"ethernet"},{"name":"GigabitEthernet6/0","port_number":0,"link_type":"ethernet"}]},{"node_id":"fcee6830-3517-4954-bf6f-5d1847ea4734","name":"R2","node_type":"dynamips","ports":[{"name":"FastEthernet0/0","port_number":0,"link_type":"ethernet"},{"name":"GigabitEthernet1/0","port_number":0,"link_type":"ethernet"},{"name":"GigabitEthernet2/0","port_number":0,"link_type":"ethernet"},{"name":"GigabitEthernet3/0","port_number":0,"link_type":"ethernet"},{"name":"GigabitEthernet4/0","port_number":0,"link_type":"ethernet"},{"name":"GigabitEthernet5/0","port_number":0,"link_type":"ethernet"},{"name":"GigabitEthernet6/0","port_number":0,"link_type":"ethernet"}]},{"node_id":"857eaa69-d763-46be-9d76-b4d5135e1da0","name":"R3","node_type":"dynamips","ports":[{"name":"FastEthernet0/0","port_number":0,"link_type":"ethernet"},{"name":"GigabitEthernet1/0","port_number":0,"link_type":"ethernet"},{"name":"GigabitEthernet2/0","port_number":0,"link_type":"ethernet"},{"name":"GigabitEthernet3/0","port_number":0,"link_type":"ethernet"},{"name":"GigabitEthernet4/0","port_number":0,"link_type":"ethernet"},{"name":"GigabitEthernet5/0","port_number":0,"link_type":"ethernet"},{"name":"GigabitEthernet6/0","port_number":0,"link_type":"ethernet"}]},{"node_id":"d6af792d-ac6f-4d8d-9d09-25695ce052dc","name":"Switch1","node_type":"ethernet_switch","ports":[{"name":"Ethernet0","port_number":0,"link_type":"ethernet"},{"name":"Ethernet1","port_number":1,"link_type":"ethernet"},{"name":"Ethernet2","port_number":2,"link_type":"ethernet"},{"name":"Ethernet3","port_number":3,"link_type":"ethernet"},{"name":"Ethernet4","port_number":4,"link_type":"ethernet"},{"name":"Ethernet5","port_number":5,"link_type":"ethernet"},{"name":"Ethernet6","port_number":6,"link_type":"ethernet"},{"name":"Ethernet7","port_number":7,"link_type":"ethernet"}]},{"node_id":"78f4a9d4-4e22-4bf8-a2b4-ee4d956a3f68","name":"Switch2","node_type":"ethernet_switch","ports":[{"name":"Ethernet0","port_number":0,"link_type":"ethernet"},{"name":"Ethernet1","port_number":1,"link_type":"ethernet"},{"name":"Ethernet2","port_number":2,"link_type":"ethernet"},{"name":"Ethernet3","port_number":3,"link_type":"ethernet"},{"name":"Ethernet4","port_number":4,"link_type":"ethernet"},{"name":"Ethernet5","port_number":5,"link_type":"ethernet"},{"name":"Ethernet6","port_number":6,"link_type":"ethernet"},{"name":"Ethernet7","port_number":7,"link_type":"ethernet"}]}],"link_info":[{"link_id":"0158c60f-af87-4ac2-a860-0f3f0204dad0","link_type":"ethernet","nodes":[{"node_id":"087f3f0a-4605-4893-bd6a-9be111f04be2","port_number":0},{"node_id":"fcee6830-3517-4954-bf6f-5d1847ea4734","port_number":0}]},{"link_id":"be91c023-200a-47d2-b780-020bbae18516","link_type":"ethernet","nodes":[{"node_id":"087f3f0a-4605-4893-bd6a-9be111f04be2","port_number":0},{"node_id":"857eaa69-d763-46be-9d76-b4d5135e1da0","port_number":0}]},{"link_id":"c2b75669-f4ca-4ac5-947d-394717622c42","link_type":"ethernet","nodes":[{"node_id":"857eaa69-d763-46be-9d76-b4d5135e1da0","port_number":0},{"node_id":"fcee6830-3517-4954-bf6f-5d1847ea4734","port_number":0}]},{"link_id":"27e0d4a2-beca-4e2e-b588-1a48a37be8e9","link_type":"ethernet","nodes":[{"node_id":"8c4ca0f9-55a2-4517-a6c7-e89b8519b38a","port_number":0},{"node_id":"fcee6830-3517-4954-bf6f-5d1847ea4734","port_number":0}]},{"link_id":"27a20665-f4bb-4559-903d-b96a18e2f85f","link_type":"ethernet","nodes":[{"node_id":"857eaa69-d763-46be-9d76-b4d5135e1da0","port_number":0},{"node_id":"78f4a9d4-4e22-4bf8-a2b4-ee4d956a3f68","port_number":0}]},{"link_id":"c9d5315f-4324-4940-a844-fe2d47e69bd2","link_type":"ethernet","nodes":[{"node_id":"087f3f0a-4605-4893-bd6a-9be111f04be2","port_number":0},{"node_id":"d6af792d-ac6f-4d8d-9d09-25695ce052dc","port_number":0}]},{"link_id":"3b939529-0d64-46d3-ba80-45647ea084b6","link_type":"ethernet","nodes":[{"node_id":"42c32716-bd61-4fb2-a731-21852caa657b","port_number":0},{"node_id":"d6af792d-ac6f-4d8d-9d09-25695ce052dc","port_number":1}]},{"link_id":"4b56ad8a-6010-4303-b13c-523aa4dddb6c","link_type":"ethernet","nodes":[{"node_id":"36221bf8-e63c-4deb-b026-f95c7ae49ba2","port_number":0},{"node_id":"d6af792d-ac6f-4d8d-9d09-25695ce052dc","port_number":2}]},{"link_id":"0134633c-506f-4486-9c0f-33bab2156da5","link_type":"ethernet","nodes":[{"node_id":"4e78a684-6c7b-485a-9c62-36ecc9c026b6","port_number":0},{"node_id":"78f4a9d4-4e22-4bf8-a2b4-ee4d956a3f68","port_number":1}]},{"link_id":"926ac61b-2b4b-4eb6-8cbc-d7bea56fdda9","link_type":"ethernet","nodes":[{"node_id":"43307983-5526-49ad-885a-e52e93f20a4e","port_number":0},{"node_id":"78f4a9d4-4e22-4bf8-a2b4-ee4d956a3f68","port_number":2}]}]}),
    question = "connect PC1 to PC2.",
)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(invoke(DUMMY_INPUT_1))
    loop.close()
