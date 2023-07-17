# To fine-tune or Not

1. Imitation Models - Llama style of fine-tuning
2. Overfitting
3. catastrohic forgetting

## Issues with Imitation Models
### To Imitate or Dynamic Prompt Selection:
1. [Example Selector](https://python.langchain.com/docs/modules/model_io/prompts/example_selectors/)
### Prompting LLM to plan and execute actions over Long documents
2. [PEARL](https://arxiv.org/format/2305.14564) - [Git Repo](https://github.com/SimengSun/pearl)

## Paper Link
[The False Promise of Imitating Proprietary LLMs](https://arxiv.org/abs/2305.15717)
"Depending on the complexity of your task, attempting to imitate the outputs of GPT or any sophisticated model with a weaker model may result in poor model performance."

## Conclusions/Take Away
1. Depending on the complexity of your task, attempting to imitate the outputs of GPT or any sophisticated model with a weaker model may result in poor model performance.
2. In-Context Learning with dynamic example loading may achieve the same results as fine tuning without substantial additional costs that would come from a managed service.
3. Breaking a task into smaller subsequent problems can help simplify a larger problem into more manageable pieces. You can also use these smaller tasks to solve bottlenecks related to model limitations.

### [Reference Article](https://towardsdatascience.com/thinking-about-fine-tuning-an-llm-heres-3-considerations-before-you-get-started-c1f483f293)
