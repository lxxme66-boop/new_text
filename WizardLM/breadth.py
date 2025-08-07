base_instruction = '''我希望你扮演一位提示词创作者的角色。\
    你的目标是从#目前的提示词# ,和#提示词的标准回答#中汲取灵感，创造出一个全新的提示词。\
    这个新提示必须与#目前的提示词#属于同一领域，\
    但要更加独特罕见。#目前的提示词#的长度和复杂程度需与#给定的提示#保持相近。\
    该提示必须合情合理，能够被人类理解并作出有效回应。\
    此外，#新的提示词# 需要简洁明了，只包括提示词内容，不得包含 \"目前的提示词\"、\"答案是......\", \"#新的提示词#\"、\"given prompt\"、\"created prompt\"等字样。'''
def createBreadthPrompt(instruction, thinking=None, answer=None):
	prompt = base_instruction
	prompt += "\n\n#目前的提示词#:\r\n {} \r\n\n".format(instruction)
	# if thinking:
	# 	prompt += "#目前提示词的标准答案#: \r\n {} \n\n 所以最终答案是：{} \r\n\n".format(thinking, answer)
	# else:
	# 	prompt += "#目前提示词的标准答案#: \r\n {} \n\n".format(answer)
	prompt += "#新的提示词#:\r\n"
	return prompt