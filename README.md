# LLM Reading List
List of Articles, video for reading and keeping progress of each. 

## Prompt Engineering

## Papers

| Paper | Abstract | Importance | Link | Read or Not|
|--------|----------|------------|------|------|
| LIMA |The relative importance of these two stages by training LIMA, a 65B parameter LLaMa language model fine-tuned with the standard supervised loss on only 1,000 carefully curated prompts and responses, without any reinforcement learning or human preference modeling.| No Re-inforcement Learning | [Link](https://arxiv.org/abs/2305.11206) | Not read |
| Distilling Step by Step | Train smaller task-specific models by either finetuning with human labels or distilling using LLM-generated labels. However, finetuning and distillation require large amounts of training data to achieve comparable performance to LLMs. We introduce Distilling step-by-step, a new mechanism that (a) trains smaller models that outperform LLMs, and (b) achieves so by leveraging less training data needed by finetuning or distillation | Smaller task specific models, not Distillation techniquens | [Link](https://arxiv.org/abs/2305.02301) | Not read |
|Dr. LLaMa | Small Language Models (SLMs) are known for their efficiency but often encounter difficulties in tasks with limited capacity and training data, particularly in domain-specific scenarios.| Method that improves SLMs through generative data augmentation utilizing LLMs, using LLMs for create a smaller models with Limited data scenarios | [Link](https://arxiv.org/abs/2305.07804)| Not read |
|FrugalGPT | Paper outlines and discusses three types of strategies that users can exploit to reduce the inference cost associated with using LLMs: 1) prompt adaptation, 2) LLM approximation, and 3) LLM cascade | LLM Graph or which LLM to use for which query | [Link](https://arxiv.org/abs/2305.05176)| Not read |
|Explanibility via Language Model | Using GPT4 to explain GPT2 results|[Link](https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html) | Not Read |
|RLHF Blog| RL with Human Feedback, how does it work, what it takes | [Link](https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html) | Not Read |
|Super ICL| Super Incontext learning which allows black box LLMs (OpenAI) to work with locally fine-tuned smaller models| Best of both worlds | [Link](https://arxiv.org/abs/2305.08848) | Not Read |
|LoRA| Low-Rank Adaptation of Large Language models - greatly reducing the number of parameters to train| One of the most popular techniques for training LLMs | [Link](https://arxiv.org/abs/2106.09685) | Not Read|

## Training

1. [LLM Bootcamp](https://fullstackdeeplearning.com/llm-bootcamp/)



## Open LLMs
1. [Open LLM](https://github.com/eugeneyan/open-llms)
2. [LLM Studio](https://github.com/h2oai/h2o-llmstudio)
3. [Number Every ML Developer should Know](https://github.com/ray-project/llm-numbers), [Numbers Every Engineer should know](http://brenocon.com/dean_perf.html)
4. [Mosaic ML - MPU Models](https://www.mosaicml.com/blog/mpt-7b) - Large Models, with huge context length, and smaller models like LLaMA
5. [Autonomous Agets](https://python.langchain.com/en/latest/use_cases/autonomous_agents.html)
