# 📊 Evaluation Tips

DeSTA2.5-Audio is a highly flexible model designed to handle a wide range of natural language instructions. It supports customization of both the system prompt and the user prompt, allowing it to adapt to various evaluation tasks — from simple auditory perception to complex multi-step reasoning.

Because the model is built on an instruction-tuned language model, it may exhibit behaviors such as:

- Providing explanations or justifications in its responses
- Starting with conversational phrases (e.g., “Sure!” or “Let me help”)
- Producing fluent, natural-sounding output instead of strictly formatted answers

These characteristics showcase the strength of a well-aligned audio language model. However, they can introduce challenges in format-sensitive evaluations like **ASR or Multiple-choice questions**, where strict output formatting is essential.

To keep things fair and consistent, our [paper](https://arxiv.org/abs/2507.02768) uses a simple and standardized setup with minimal prompt engineering. More advanced prompts, like chain-of-thought or task-specific templates, could improve results and are worth exploring in future work. 

Please consider using the following prompts for your evaluation:

| Question type     | System Prompt                                                                 | User Prompt                                                                                                      |
|-------------------|--------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| Default           | Focus on the audio clips and instruction.                                     | <\|AUDIO\|> {instruction}                                                                                        |
| Multiple-choice   | Focus on the audio clips and instruction. Choose one of the options without any explanation. | <\|AUDIO\|> {instruction}\nThe answer could be: "{option_1}", \"{option_2}\", \"{option_3}\" or \"{option4}\"           |
| Multiple-choice (2) | Focus on the audio clip and instruction. Put your answer in the format \"The correct answer is: \"___\" \". | <\|AUDIO\|> {instruction}\nChoose from the following options: \"{option1}\", \"{option2}\", \"{option3}\" or \"{option4}\"
| Dialogue-based    | You are a voice assistant. Act as if you are a natural language partner       | <\|AUDIO\|>                                                                                                      |
| ASR (word-level metrics) | Focus on the audio clips and instruction. Respond directly without any other words | <\|AUDIO\|> {instruction}                                                                                        |
