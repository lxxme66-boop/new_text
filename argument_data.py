import base64
import argparse
import os
# 通过 pip install volcengine-python-sdk[ark] 安装方舟SDK
from volcenginesdkarkruntime import Ark, AsyncArk

import os
import re
from Doubao.prompts_conf import system_prompt, user_prompts
import asyncio
import json
from typing import List, Dict, Any

ark_url = "https://ark.cn-beijing.volces.com/api/v3"
api_key = "ecb26efc-05e7-4d58-8d40-0dca61ccb4e9" # my own api
model = "doubao-1.5-thinking-pro-m-250428"


class ArgumentDataProcessor:
    """数据增强处理器类"""
    
    def __init__(self, api_key: str = api_key, model: str = model):
        self.api_key = api_key
        self.model = model
        self.ark_url = ark_url
    
    async def enhance_qa_data(self, qa_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """增强QA数据"""
        enhanced_data = []
        
        # 处理每个QA对
        for idx, qa_item in enumerate(qa_data):
            try:
                # 使用现有的modify_sft_response函数进行增强
                enhanced_item = await self._enhance_single_qa(qa_item, idx)
                enhanced_data.append(enhanced_item)
            except Exception as e:
                print(f"Error enhancing QA item {idx}: {e}")
                enhanced_data.append(qa_item)  # 如果增强失败，保留原始数据
        
        return enhanced_data
    
    async def _enhance_single_qa(self, qa_item: Dict[str, Any], index: int) -> Dict[str, Any]:
        """增强单个QA项"""
        # 选择合适的prompt模板
        prompt_template = user_prompts['rewrite_prompt']
        
        # 调用现有的modify_sft_response函数
        enhanced_response = await modify_sft_response([qa_item], 0, prompt_template)
        
        if enhanced_response:
            return enhanced_response
        else:
            return qa_item
async def modify_sft_response(responses, index,prompt_template):

    response = responses[index]
    question = response.get('question', '')
    answer = response.get('answer', '')
    choices = response.get('choices', [])
    reasoning = response.get('reasoning', '')
    lecture = response.get('lecture', '')
    context = response.get('context', '')
    imageDescription = response.get('imageDescription', '')
    AnalysisResult = response.get('analysisResults', '')
    RelatedKnowledge = response.get('relatedKnowledge', '')
    image_path=response.get('image_path', '')
    prompt = prompt_template.format(
        question=question,
        answer=answer,
        choices=choices,
        reasoning=reasoning,
        lecture=lecture,
        context=context,
        imageDescription=imageDescription,
        analysisResults=AnalysisResult,
        relatedKnowledge=RelatedKnowledge
    )
    ark = AsyncArk(api_key=api_key)
    async def get_request(input_prompt):
        try:
            #print(f"Sending request for index {index} with prompt: {input_prompt}")
            response = await ark.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": system_prompt}

                        ]
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": input_prompt},
                        ]
                    },
                ],
                temperature=1,
                max_tokens=4096,
                top_p=0.7,
                
                stop=None


            )
            try:
                content = response.choices[0].message.content
                #print(content)

           
                content = eval(content)
                content["choices"] = choices
                content["question"] = question
                content["answer"] = answer
              
                content["lecture"] = lecture
                content["context"] = context
            except Exception as e:
                print(f"Error in eval: {e}")
                
                #print(f"Using raw content: {content}")
                return None
            return content
        except Exception as e:
            ##print(f"Error in request: {e}")
            if 'ModelAccountTpmRateLimitExceeded' in str(e):
                print("Rate limit exceeded, waiting for 60 seconds...")
                await asyncio.sleep(60)
                return None
            else:
                # interrupt the main thread
                print(f"input prompt: {input_prompt}")
                
                print(f"Error in request: {e}")
                return None
                
            
    content = await get_request(prompt)
    if content is None:
        print(f"Response is None for index {index}, skipping...")
        return None
    content["image_path"] = image_path
    print(content)
    return content
async def check_ori_response(responses,prompt_template,batch_size=1):
    async def check_response(response,prompt_template):
        
        vqa_data = {}
        vqa_data["input"] = response["input"]
        vqa_data["instruction"] = response["instruction"]
        
        vqa_data["reasoning"] = response["reasoning"]
    
        vqa_data["output"] = response["output"]
        check_prompt = prompt_template.format(vqa_data=vqa_data)
        ark = AsyncArk(api_key=api_key)
        content = await ark.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": system_prompt}
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": check_prompt},
                    ]
                },
            ],
            temperature=1,
            max_tokens=4096,
            top_p=0.7,
            stop=None
        )
        try:
            content = content.choices[0].message.content
            content = eval(content)
            # print(f"Response: {content}")
            # print(f"input prompt: {response}")
            return content, response
        except Exception as e:
            print(f"Error in eval: {e}")
            return None
    new_responses = []
    tasks = []
    for response in responses:
        task = asyncio.create_task(check_response(response, prompt_template))
        tasks.append(task)
    for i in range(0, len(tasks), batch_size):
        batches = tasks[i:i + batch_size]
        batches = await asyncio.gather(*batches)
        # Use asyncio.gather to run the tasks concurrently  
        for t, data in enumerate(batches):
            response, vqa_data = data
            if response is None:
                print(f"Response is None, skipping...")
                continue
            elif response["useful"]:
                new_responses.append(vqa_data)
                # print(vqa_data)
            else:
                print(response)
                print(vqa_data)

        
    return new_responses

async def rephrase_response(responses, index,poolSize, CheckQestion):
    response_index = 0
    tasks = []
    for i in range(len(responses)):
        task = asyncio.create_task(modify_sft_response(responses, i, user_prompts[index]))
        tasks.append(task)
    # Use asyncio.gather to run the tasks concurrently
    for i in range(0, len(tasks), poolSize):
        batch_tasks = tasks[i:i + poolSize]
        batch_responses = await asyncio.gather(*batch_tasks)
        for response in batch_responses:
            if response is not None:
                responses[response_index] = response
                response_index += 1
                print(f"Processed response {response_index}/{len(responses)}")
    return responses


if __name__ == "__main__":
    import json
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--input_file", type=str, \
                        default="/home/maxzhang/VLReasoningTCL/data/pdfs/version 9.1.json",\
                         help="path to the input json file")

    parser.add_argument("--output_file", type=str, default="/home/maxzhang/VLReasoningTCL/data/pdfs", help="path to the output folder")
    parser.add_argument("--indexes", type=int, default = 21, help="indexes for rephrase prompt 21 for sft, 22 for cpt")
    parser.add_argument("--poolSize", type=int, default=8, help="number of parallel tasks")
    parser.add_argument("--CheckQestion",type=int, default=22, help="whether to check the question or not, if not the index will be -1, otherwise will be positive index")
    
    args = parser.parse_args()
    
    output_dir = args.output_file
    indexes = args.indexes
    poolsize = args.poolSize
    input_file = args.input_file
    with open(input_file,"r") as f:
        input_file = json.load(f)

    responses = asyncio.run(rephrase_response(input_file, indexes, poolSize=poolsize, CheckQestion=args.CheckQestion))
    with open(os.path.join(output_dir, "rephrased_responses_qa.json"), "w", encoding='utf-8') as f:
        import json
        json.dump(responses, f, ensure_ascii=False, indent=4)
    print(responses)
    if args.CheckQestion != -1:
        file_path = os.path.join(output_dir, f"rephrased_responses_qa.json")
        responses = []
        indexes = args.CheckQestion
        with open(file_path, "r") as f:
            responses = json.load(f)
        responses = asyncio.run(check_ori_response(responses, user_prompts[indexes],batch_size=poolsize))
        with open(os.path.join(output_dir, "checked_responses_qa.json"), "w", encoding='utf-8') as f:
            import json
            json.dump(responses, f, ensure_ascii=False, indent=4)