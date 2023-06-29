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
|Explanibility via Language Model | Using GPT4 to explain GPT2 results||[Link](https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html) | Not Read |
|RLHF Blog| RL with Human Feedback, how does it work, what it takes | |[Link](https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html) | Not Read |
|Super ICL| Super Incontext learning which allows black box LLMs (OpenAI) to work with locally fine-tuned smaller models| Best of both worlds | [Link](https://arxiv.org/abs/2305.08848) | Not Read |
|LoRA| Low-Rank Adaptation of Large Language models - greatly reducing the number of parameters to train| One of the most popular techniques for training LLMs | [Link](https://arxiv.org/abs/2106.09685) | Not Read|
|ALiBi|  We therefore introduce a simpler and more efficient position method, Attention with Linear Biases (ALiBi). ALiBi does not add positional embeddings to word embeddings; instead, it biases query-key attention scores with a penalty that is proportional to their distance.| how does a model achieve extrapolation at inference time for sequences that are longer than it saw during training|[Link](https://arxiv.org/abs/2108.12409)|Not Read|
|Self-Supervise Learning Bible| "dark matter of intelligence| A good take on lot of things related to training the self-supervised techniques|[Link](https://arxiv.org/abs/2304.12210)|Not Read|
|ToolFormers|Toolformer: Language Models Can Teach Themselves||[Link](https://arxiv.org/abs/2302.04761)|Not Read|
|QLoRA|QLoRA : Efficient Finetuning of Quantized LLMs|An efficient fine-tuning approach that reduces memor usage enough to to finetune a 65B,QLoRA backpropagates gradients through a frozen, 4-bit quantized pretrained language model into Low Rank Adapters~(LoRA). Our best model family, which we name Guanaco, outperforms all previous openly released models on the Vicuna benchmark|[Link](https://arxiv.org/abs/2305.14314)|Not Read|
|Scaling Laws |Scaling Laws for Neural Language Models | Simple equations govern the dependence of overfitting on model/dataset size and the dependence of training speed on model size. These relationships allow us to determine the optimal allocation of a fixed compute budget.|[Link](https://arxiv.org/pdf/2001.08361.pdf)|Not Read|
##
## Training
1. [LLM Bootcamp](https://fullstackdeeplearning.com/llm-bootcamp/)
2. [QLora Blog](https://huggingface.co/blog/4bit-transformers-bitsandbytes)

## Inferencing
1. [HuggingFace Inference Engine](https://github.com/huggingface/text-generation-inference)

## Tools/ Generic
1. [Semantic Kernals](https://learn.microsoft.com/en-us/semantic-kernel/overview/) 
2. [Autonomous Agets](https://python.langchain.com/en/latest/use_cases/autonomous_agents.html)
3. [LLM Economics](https://towardsdatascience.com/llm-economics-chatgpt-vs-open-source-dfc29f69fec1) 
4. [Prompt Engineering](https://github.com/brexhq/prompt-engineering)
5. [Transformers](https://e2eml.school/transformers.html#resources)


## Open LLMs
1. [Open LLM](https://github.com/eugeneyan/open-llms)
2. [LLM Studio](https://github.com/h2oai/h2o-llmstudio)
3. [Number Every ML Developer should Know](https://github.com/ray-project/llm-numbers), [Numbers Every Engineer should know](http://brenocon.com/dean_perf.html)
4. [Mosaic ML - MPU Models](https://www.mosaicml.com/blog/mpt-7b) - Large Models, with huge context length, and smaller models like LLaMA
5. [Big Code](https://huggingface.co/bigcode) - Brilliant pipeline code for analysis
6. [PEFT](https://smashinggradient.com/2023/04/11/summary-of-adapter-based-performance-efficient-fine-tuning-peft-techniques-for-large-language-models/)
7.  [List of Good books to read on ML](https://aman.ai/read/#python-for-computational-science-and-engineering)

## Questions on LLMs:

1. What is the best way to train a domain specific transformer - assuming low data and low compute available?
2. What are the typical things in pipeline one must check when doing unsupervised training
3. When to use MLM, CLM, PFT, LoRA 
4. Model Optimizing - Quantization
