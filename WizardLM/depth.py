# 基础指令
base_instruction = "我希望你扮演一个提示改写者。\r\n \
					你的目标是将给定的提示改写成更复杂的版本，使那些著名的人工智能系统（例如doubao thinking 和chatGPT）更难处理。\r\n \
					但改写后的提示必须合理且人类能够理解和回应。\r\n \
					请勿省略非文本部分（如表格和代码），也请勿省略#目前的提示词#中的输入内容。\r\n \
					你应当通过以下方法复杂化给定的提示：\r\n\
					{}\r\n\
					改写后的提示只能比原提示增加10至20个词，且不得冗长。\r\n\
					#新的的提示词# 只包括提示词内容，不得包含中不得包含'#目前的提示词#'、'#新的的提示词#'、'given prompt'或'新的的提示词'字样\n\n"

def createConstraintsPrompt(instruction,answer=None,thinking=None):
	prompt = base_instruction.format("请在#目前的提示词#中添加一个额外的约束条件或要求")
	prompt += "#目前的提示词#:\r\n{}\r\n".format(instruction)
	# if thinking:
	# 	prompt += "#目前提示词的标准答案#: \r\n {} \n\n 所以最终答案是：{} \r\n\n".format(thinking, answer)
	# else:
	# 	prompt += "#目前提示词的标准答案#: \r\n {} \n\n".format(answer)
	
	prompt += "请输出#新的提示词#\r\n"
	return prompt

def createDeepenPrompt(instruction,answer=None,thinking=None):
	prompt = base_instruction.format("如果#目前的提示词#包含对特定问题的询问，可扩展其深度和广度")
	prompt += "#目前的提示词#:\r\n{}\r\n".format(instruction)
	# if thinking:
	# 	prompt += "#目前提示词的标准答案#: \r\n {} \n\n 所以最终答案是：{} \r\n\n".format(thinking, answer)
	# else:
	# 	prompt += "#目前提示词的标准答案#: \r\n {} \n\n".format(answer)
	
	prompt += "请输出#新的提示词#\r\n"
	return prompt

def createConcretizingPrompt(instruction,answer=None,thinking=None):
	prompt = base_instruction.format("请将一般概念替换为更具体的概念")
	prompt += "#目前的提示词#:\r\n{}\r\n".format(instruction)
	# if thinking:
	# 	prompt += "#目前提示词的标准答案#: \r\n {} \n\n 所以最终答案是：{} \r\n\n".format(thinking, answer)
	# else:
	# 	prompt += "#目前提示词的标准答案#: \r\n {} \n\n".format(answer)
	
	prompt += "请输出#新的提示词#\r\n"
	return prompt

def createReasoningPrompt(instruction,answer=None,thinking=None):
	prompt = base_instruction.format("若#目前的提示词#可通过简单思维过程解决，可改写为显式要求多步骤推理")
	prompt += "#目前的提示词#:\r\n{}\r\n".format(instruction)
	# if thinking:
	# 	prompt += "#目前提示词的标准答案#: \r\n {} \n\n 所以最终答案是：{} \r\n\n".format(thinking, answer)
	# else:
	# 	prompt += "#目前提示词的标准答案#: \r\n {} \n\n".format(answer)
	
	prompt += "请输出#新的提示词#\r\n"
	return prompt