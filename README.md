# S-Witch (Switch Witch) : Switch Configuration Assistant

<div align="center">
  
  ![python](https://img.shields.io/badge/python-3.11-brightgreen)
  ![pytorch](https://img.shields.io/badge/langchain-0.0.352-orange)
  ![openai](https://img.shields.io/badge/openai-0.28.1-blueviolet)
  ![gns3](https://img.shields.io/badge/gns3-2.2.44-blue)
  
</div>

In modern network structures that become more complex and emphasize flexibility, the demand for the automation of network management and Intent Driven Network continues to increase. In response, technology utilizing virtualization and control plane separation has developed, and research on network automation based on this is being actively conducted. However, limited studies focus on building an automated network in environments comprised of traditional switches that lack support for these advanced functionalities. Consequently, this study presents a technology proposal that creates the CLI command for existing commercial switches by incorporating user requests conveyed through natural language. For this purpose, we applied LLM for generating and Network Digital Twin for verification environment.

## Architecture

### Overall Architecture

Our system consists of 3 components: S-Witch, Digital Twin, and Web UI.

<div align="center">
  <img src="./imgs/overall-architecture.png" width="500px" />
</div>

### Chaining Architecture

Our connection strategy is as follows.

<div align="center">
  <img src="./imgs/llm-chaining.png" width="800px" />
</div>

## Result

### Web View (Chatting)

<div align="center">
  <img src="./imgs/web-ui.png" width="500px" />
</div>
