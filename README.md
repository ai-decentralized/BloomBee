<p align="center">  
    <img src="figures/bloombee.jpg" alt="Bloombee Logo" /><br>  
    Run large language models in a heterogeneous decentralized environment with offloading.<br>
    <br>
    <a href="https://pypi.org/project/bloombee/"><img src="https://img.shields.io/pypi/v/bloombee.svg?label=PyPI&color=green"></a>
    <a href="https://github.com/ai-decentralized/bloombee/actions"><img src="https://img.shields.io/github/actions/workflow/status/ai-decentralized/bloombee/pylint.yml?branch=main&label=Build"></a>
    <a href="https://discord.gg/Ypexx2rxt9"><img src="https://img.shields.io/discord/1267714065166241813?label=Discord&logo=discord&logoColor=white"></a>
</p>  

The rapid rise of generative AI has boosted demand for large language model (LLM) inference and fine-tuining services. While proprietary models are still favored, advancements in open-source LLMs have made them competitive. However, high costs and limited GPU resources hinder deployment. This work introduces BloomBee, a decentralized offline serving system that leverages idle GPU resources to provide cost-effective access to LLMs.

We rely on global GPU sharing, which includes more consumer-grade GPUs. If your GPU can only manage a small portion of a large language model, like the Llama3.1 (405B) model, you can connect to a network of servers that load different parts of the model. In this network, you can request inference or fine-tuning services.

<p align="center">
    🚀 &nbsp;<b><a href="https://colab.research.google.com/drive/1BZn0KrEGaNA2dlzmCTtTIjJKx3bNzOMs#scrollTo=1Qhi4I2PSGgg">Try now in Colab</a></b>
</p>

## Installation

#### From Pypi
```
pip install bloombee
```
#### From Source
```bash  
git clone https://github.com/ai-decentralized/BloomBee.git  
cd BloomBee  
pip install .
```
## How to use BloomBee(<a href="https://colab.research.google.com/drive/1pENMOEoEV01DqBImZzuX_4jTV3fNwNga#scrollTo=oyCFDemCZsRs">Try now in Colab</a>)
#### 1. Start the main server 
```
python -m bloombee.cli.run_dht --host_maddrs /ip4/0.0.0.0/tcp/31340 --identity_path bootstrapp1.id 

```
Now you will get the BloomBee's main server location: 
```
Mon 00 01:23:45.678 [INFO] Running a DHT instance. To connect other peers to this one, use --initial_peers /ip4/YOUR_IP_ADDRESS/tcp/31340/p2p/QmefxzDL1DaJ7TcrZjLuz7Xs9sUVKpufyg7f5276ZHFjbQ
```  
You can provide this address as --initial_peers to workers or other backbone servers.

If you want your swarm to be accessible outside of your local network, ensure that you have a **public IP address** or set up **port forwarding** correctly, so that your peer is reachable from the outside.

#### 2. Connect the workers to the main bloombee server  
Here is the BloomBee Server location:
```
export BBSERVER=/ip4/10.52.2.249/tcp/31340/p2p/QmefxzDL1DaJ7TcrZjLuz7Xs9sUVKpufyg7f5276ZHFjbQ  

```
Start one worker to hold 16 blocks (16 tranformer layers)
```
python -m bloombee.cli.run_server huggyllama/llama-7b --initial_peers $BBSERVER --num_blocks 16  --identity_path bootstrap_1.id
```
Start second worker to hold another 16 blocks (16 tranformer layers)
```
python -m bloombee.cli.run_server huggyllama/llama-7b --initial_peers $BBSERVER --num_blocks 16  --identity_path bootstrap_1.id
```

#### 3. Run inference or finetune jobs

#### Inference   
```
cd Bloombee/
python benchmarks/benchmark_inference.py --model huggyllama/llama-7b  --initial_peers $BBSERVER --torch_dtype float32 --seq_len 128
```

#### Finetune 

```
cd Bloombee/
python benchmarks/benchmark_training.py --model huggyllama/llama-7b  --initial_peers $BBSERVER --torch_dtype float32  --n_steps 20 --batch_size 32 --seq_len 128
```


## Acknowledgements  

Bloombee is built upon a few popular libraries: 

  - [Hivemind](https://github.com/learning-at-home/hivemind) - A PyTorch library for decentralized deep learning across the Internet.  
  - [FlexLLMGen](https://github.com/FMInference/FlexLLMGen) - An offloading-based system running on weak GPUs.  
  - [Petals](https://github.com/bigscience-workshop/petals) - A library for decentralized LLMs fine-tuning and inference without offloading.

