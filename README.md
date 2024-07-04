# S-Witch (Switch Witch) : Switch Configuration Assistant

<div align="center">
  
  ![python](https://img.shields.io/badge/python-3.11-brightgreen)
  ![pytorch](https://img.shields.io/badge/langchain-0.0.352-orange)
  ![openai](https://img.shields.io/badge/openai-0.28.1-blueviolet)
  ![gns3](https://img.shields.io/badge/gns3-2.2.44-blue)
  
</div>

<div align="center">

  ![thumbnail](/imgs/S-Witch.png)

</div>


In modern network structures that become more complex and emphasize flexibility, the demand for the automation of network management and Intent Driven Network continues to increase. In response, technology utilizing virtualization and control plane separation has developed, and research on network automation based on this is being actively conducted. However, limited studies focus on building an automated network in environments comprised of traditional switches that lack support for these advanced functionalities. Consequently, this study presents a technology proposal that creates the CLI command for existing commercial switches by incorporating user requests conveyed through natural language. For this purpose, we applied LLM for generating and Network Digital Twin for verification environment.

## Published at

[_E. -D. Jeong, H. -G. Kim, S. Nam, J. -H. Yoo and J. W. -K. Hong, "S-Witch: Switch Configuration Assistant with LLM and Prompt Engineering," NOMS 2024-2024 IEEE Network Operations and Management Symposium, Seoul, Korea, Republic of, 2024, pp. 1-7, doi: 10.1109/NOMS59830.2024.10575007. keywords: {Automation;Natural languages;Buildings;Digital twins;Proposals;Virtualization;Network Configuration Automation;Intent Driven Network;Large Language Model;Prompt Engineering}_](https://ieeexplore.ieee.org/document/10575007)

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

## How to run

1. download and run [GNS3 Server-2.2.44](https://github.com/GNS3/gns3-server).
2. go to [api directory](/api) and run S-Witch component.
3. go to [GNS3 Web UI](https://github.com/euidong/gns3-web-ui) and run GNS3 Web UI.
   - you must consider [connection with S-Witch and GNS3 Server Component](https://github.com/euidong/S-Witch/README.md#overall-architecture).
4. make a topology and test. happy hacking~🤗
