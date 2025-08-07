import os
from openai import OpenAI, AsyncOpenAI
from Doubao.Datageneration import input_ImgDoubao
from Doubao.prompts_conf import system_prompt, user_prompts
from Utilis.utilis import img2base64
import asyncio
import json
import pandas as pd
import re
#api_key = "sk-85caf631866d4b0fb79eaeeb34f8f96e"
model = "qvq-max-2025-03-25"
qwen_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"


api_key = "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"
qwen_url = None
model = "o3"


qwen_url="https://ark.cn-beijing.volces.com/api/v3"
api_key = "ae37bba4-73be-4c22-b1a7-c6b1f5ec3a4b" # my own api
model = "ep-20250707132521-j2q26"


qwen_url = "http://0.0.0.0:8080/v1"
model = "/mnt/storage/models/Skywork/Skywork-R1V3-38B"
async def get_response(input_prompt,img_path=None,api_key=api_key, qwen_url=qwen_url, model=model,stream = False):
    
    client= AsyncOpenAI(
        api_key=api_key,
        base_url=qwen_url if qwen_url != None else None, 
       
    )
    img_path = img_path.replace("./","")
    img_url = img2base64(img_path) if img_path else None
    reasoning_content = ""
    answer_content = ""

    messages = [
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
                {"type": "image_url", "image_url": {"url": img_url}} 
            ]
        }
    ]
    
    print(stream)
    completion = await client.chat.completions.create(
        model=model,
        messages= messages,
        stream= stream,
        # temperature=0.7,
        # max_completion_tokens=4096,
        # top_p=0.8,
    )
    if stream:
        async for chunk in completion:
            if not chunk.choices:
                print("\nUsage:")
                print(chunk.usage)
            else:
                delta = chunk.choices[0].delta
                # 打印思考过程
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content != None:
                    #print(delta.reasoning_content, end='', flush=True)
                    reasoning_content += delta.reasoning_content
                else:
                    # 开始回复
                    # if delta.content != "" and is_answering is False:
                    #     print("\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n")
                    #     is_answering = True
                    # 打印回复过程
                    #print(delta.content, end='', flush=True)
                    answer_content += delta.content
    else:
        if not completion.choices:
            print("\nUsage:")
            print(completion.usage)
            return None
        if model == "o3":
            if completion.choices:
                for choice in completion.choices:
                    message = choice.message
                    if hasattr(message, 'reasoning_content') and message.reasoning_content is not None:
                        reasoning_content += message.reasoning_content
                    if hasattr(message, 'content') and message.content is not None:
                        answer_content += message.content
        else:
            answer_content = completion.choices[0].message.content
            reasoning_content = completion.choices[0].message.reasoning_content
    if reasoning_content is None and answer_content is None:
        print("Error: reasoning_content or answer_content is None")
        return None
    print("\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n")
    print(f"Reasoning content: {reasoning_content}")
    print("\n" + "*" * 20 + "最終答案" + "*" * 20 + "\n")
    print(answer_content)
    
    return answer_content, reasoning_content, img_path
    # # applying regex to extract the answer content between the first and last curly braces
    # start_idx = answer_content.find('{')
    # end_idx = answer_content.rfind('}')
    # if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
    #     print("Error: No valid answer content found")
    #     return answer_content
    # answer_content = answer_content[start_idx:end_idx + 1]
    # ##answer_content = answer_content.group(0)  # Get the matched content
    # answer_content = eval(answer_content)
    # final_output = {}
    # final_output["reasoning_content"] = reasoning_content
    # final_output["image_path"] = img_path
    # for key in answer_content:
    #     final_output[key] = answer_content[key]
    
    # return final_output
async def process_response(*args,**kwargs):
    try:
        response = await get_response(*args,**kwargs)
    except Exception as e:
        print(f"Error in get_response: {e}")
        return None
    answer_content, reasoning_content, img_path = response
    import re
    
    pattern = re.compile(r"```json\s*(\{[\s\S]*?\})\s*```", re.DOTALL)
    match = re.findall(pattern, answer_content)
    if match:
        answer_content = match[-1]
    else:

        start_idx = answer_content.find('{')
        end_idx = answer_content.rfind('}')
        if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
            print("Error: No valid answer content found")
            return None
        answer_content = answer_content[start_idx:end_idx + 1]
    ##answer_content = answer_content.group(0)  # Get the matched content
    try:
        answer_content = eval(answer_content)
    except Exception as e:
        print(f"Error evaluating answer content: {e}")
        return None
    final_output = {}
    final_output["reasoning_content"] = reasoning_content
    final_output["image_path"] = img_path
    for key in answer_content:
        final_output[key] = answer_content[key]
        if "无法设计出问题" in final_output[key]:
            return None
    
    return final_output
    
async def get_total_responses(prompt_index, input_material_file,image_folder,pool_size=10,stream = False):
    total_tasks = []
    input_prmpt = user_prompts[prompt_index]
    results = []
    with open(input_material_file, 'r') as f:
        input_materials = json.load(f)
    for material in input_materials:
        img_path = material.get("image_path")
        for key, value in material.items():
            if isinstance(value,list):
                value = "\n".join(value)
            if len(value.strip()) <=5:
                continue
        print(input_prmpt)
        final_input = input_prmpt.format(**material)
        task = asyncio.create_task(
            process_response(final_input, os.path.join(image_folder, img_path) if img_path else None,stream = stream)
        )
        total_tasks.append(task)
    for i in range(0, len(total_tasks), pool_size):
        batch_tasks = total_tasks[i:i + pool_size]
        batch_results = await asyncio.gather(*batch_tasks)
        results.extend(batch_results)
    # Filter out None results
    results = [result for result in results if result is not None]
    return results
async def check_data_quality(url,key,model,output_file,prompt_index,pool_size=10,check_times = 5, pass_score=1,stream = False ):
    with open(output_file, 'r') as f:
        data = json.load(f)
    final_output = []
    api_key = key
    qwen_url = url
    model = model
    # make sure the data's 
    for sample in data:
        image_path = sample.get("image_path")
        # 
        sample.pop("reasoning_content", None)
        sample.pop("img", None)
        #input_text = str(sample)
        
        
        tasks = []
        for _ in range(check_times):
            if type(prompt_index) == int:
                input_prompt = user_prompts[prompt_index]
            else:
                index = _ % len(prompt_index)
                input_prompt = user_prompts[prompt_index[index]]
            final_input = input_prompt.format(**sample)
            task = asyncio.create_task(get_response(final_input, img_path=image_path, api_key=api_key, qwen_url=qwen_url, model=model,stream=stream))
            tasks.append(task)
        results = await asyncio.gather(*tasks)
        # only extract the first integer from the result
        integers = []
        for result in results:
            result = result[0].strip()
            integers.append(int(result))
        mean_value = sum(integers) / len(integers)
        if mean_value >= pass_score:
            final_output.append(sample)
        else:
            print(f"Data quality check failed for prompt index {prompt_index} with mean value {mean_value}.")
       
    pd.DataFrame(final_output).to_csv(output_file.replace(
        ".json", "_filtered.csv"), index=False)


    print(f"file has been saved to csv {output_file.replace('.json', '_filtered.csv')}")
    return True
if __name__ == "__main__":
    index = 35
    file_path = "/home/maxzhang/VLReasoningTCL/data/output/total_response.json"
    image_folder = "/home/maxzhang/VLReasoningTCL/data/output"
    pool_size = 10
    results = asyncio.run(get_total_responses(index, file_path, image_folder, pool_size))
    output_file = f"/home/maxzhang/VLReasoningTCL/DataGenerationPrompt/Qwen/qwen_results_{index}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"Results saved to {output_file}")
    
    