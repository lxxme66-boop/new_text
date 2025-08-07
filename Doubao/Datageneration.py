import base64
import argparse
import os
# 通过 pip install volcengine-python-sdk[ark] 安装方舟SDK
from volcenginesdkarkruntime import Ark, AsyncArk
from openai import OpenAI,AsyncOpenAI
import os
import re
from .prompts_conf import system_prompt, user_prompts
import asyncio
ark_url = "https://ark.cn-beijing.volces.com/api/v3"
api_key = "ae37bba4-73be-4c22-b1a7-c6b1f5ec3a4b" # my own api
model = "ep-20250707132521-j2q26"
#model = "doubao-lite-128k-240828"
# define logger
ark_url = "http://0.0.0.0:8080/v1"
model = "/mnt/storage/models/Skywork/Skywork-R1V3-38B"




import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.makedirs("Doubao/logs", exist_ok=True)
current_time = os.path.getmtime(__file__)
file_handler = logging.FileHandler(f"Doubao/logs/doubao_{current_time}.log")


def extract_text_block(prev_idx, next_idx, content,markdown_content):
    """
    Extract text blocks between two image indices.
    """
    text_blocks = []
    
    for block in content:
        if block["page_idx"] <= next_idx and block["page_idx"] >= prev_idx:
            if block["type"] == "text":
                text_blocks.append(block["text"])
            elif block["type"] == "image":
                img_caption = block["img_caption"]
                text_blocks.append(f"![]({block['img_path']})")
                for caption in img_caption:
                    text_blocks.append(caption)
    # extract the content in markdown according to the first and last line in the text_blocks
    first_block = text_blocks[0] 
    last_block = text_blocks[-1] 

    start_idx = markdown_content.find(first_block)
    end_idx = 0
    for last_block in reversed(text_blocks):
        last_idx = markdown_content.find(last_block, start_idx)
        if last_idx >= end_idx:
            end_idx = last_idx
    if start_idx == -1:
        while start_idx == -1 and text_blocks:
            text_blocks.pop(0)
            first_block = text_blocks[0] if text_blocks else ""
            start_idx = markdown_content.find(first_block)
    if start_idx == -1:
        logger.warning("No start index found for the first block in markdown content.")
        start_idx = 0
        return ""
  
            
    end_idx = end_idx + len(last_block)+1
    # logger.info(f"Extracting content from index {start_idx} to {end_idx} in markdown content. {prev_idx} {next_idx}")
    extracted_content = markdown_content[start_idx:end_idx]
    
    return extracted_content



def extract_parsing_image_result(output_dir,page_ranges=None):
    # read the image and its corresponding catption from the whole pdf_file
    import json
    #content_json = os.path.join(output_dir,os.path.basename(output_dir).replace(".pdf","_content_list.json"))
    potential_files = os.listdir(output_dir)
    for file in potential_files:
        if "_content_list.json" in file:
            content_json = os.path.join(output_dir,file)
            
        if ".md" in file:
            md_file = os.path.join(output_dir,file)

    with open(content_json, "r") as f:
        content = json.load(f)
    with open(md_file, "r") as f:
        md_content = f.read()
        
    
    if page_ranges is not None:
        final_json_blocks = []
        while len(content):
            block = content.pop(0)
            for page_range in page_ranges:
                start_page, end_page = page_range
                if block["page_idx"] >= start_page-1 and block["page_idx"] <= end_page+1:
                    final_json_blocks.append(block)
                    break
        content = final_json_blocks
    else:
        start_page = 0
        end_page = None
    
    
   
    
    
    return_contents = []
    for i,block in enumerate(content):
        if block["type"] == "image":
            image_path = block["img_path"]
            
            img_caption = block["img_caption"]
            page_idx = block["page_idx"]
            if page_idx < start_page:
                continue
            else:
                if end_page is not None and page_idx > end_page:
                    continue
            prev_idx = max(page_idx-1, 0)
            next_idx = min(page_idx+1, content[-1]["page_idx"])
            #only extract the content that that is in and between the previous and next image

        
           
            # read the md file and find the line that contains the image substitute other iamge with '![this image is missing]()'
            # substitute this image with the ![this is the image given to you](image_path)
           
            temp_md_content = md_content
            temp_md_content = extract_text_block(prev_idx, next_idx, content, temp_md_content)
            res = re.finditer(r"!\[.*?\]\((.*?)\)", temp_md_content)
            for match in res:
                
                if match.group(1) == image_path:
                    temp_md_content = temp_md_content.replace(match.group(0), f"<image>")
                else:
                    temp_md_content = temp_md_content.replace(match.group(0), f"<irrelevant>")
    
            return_contents.append({"image_path":image_path,"img_caption":img_caption,"md_content":temp_md_content})
            # save the md_content to a file
            # image id is content between / and .
            img_id =re.search(r'\/(.*?)\.', image_path)
            img_id = img_id.group(1) if img_id else "unknown"
            with open(os.path.join(output_dir, f"parsing_md_content_{page_idx}_{img_id}.md"), "w") as f:
                f.write(temp_md_content)
        
    return return_contents

async def get_request(input_prompt, img_encoded_format,img_path,ark):
    if img_encoded_format is None:
        content =  [
                        {"type": "text", "text": input_prompt},
                    ]
    else:
        content = [
                        {"type": "text", "text": input_prompt},
                        {"type": "image_url", "image_url": {"url": img_encoded_format }}
                    ]
    try:
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
                    "content": content
                },
            ],
            temperature=1,
            max_tokens=16384,
            top_p=0.7,
            stop=None


        )
        # 
    except Exception as e:
        print(f"Error in request: {e}")
        print(input_prompt)
        raise e
        return None
        
        

    
    content = response.choices[0].message.content
    reason_response = response.choices[0].message.reasoning_content
    #print(content,reason_response,img_path)
    logger.info(f"seperator: {'-'*50}")
    logger.info(f"Response content: {content}")
    logger.info(f"seperator: {'-'*50}") 
    logger.info(f"Reasoning response: {reason_response}")
    logger.info(f"seperator: {'-'*50}")
    return {"content": content, "reason_response": reason_response,
                        "image_path": img_path}
def input_ImgDoubao(image_path):
    with open(image_path, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
        image_type = image_path.split(".")[-1]
        
    return f"data:image/{image_type};base64,{image_base64}"





async def parse_pdf(pdf_path,inddex = 5,page_ranges=None,img_lists = []):
    
    inputs = extract_parsing_image_result(pdf_path,page_ranges)
    ark = AsyncOpenAI(api_key=api_key,
                      base_url = ark_url
    
    )
    user_prompt = user_prompts[inddex]
    
    tasks = []
    for input_json in inputs:
        
        img_path = input_json["image_path"]
        
        input_md = input_json["md_content"]
        img_caption = input_json["img_caption"]
        input_prompt = user_prompt.format(markdown_content=input_md, image_path=img_path,
                                          image_caption=img_caption)  
        # logger.info(f"seperator: {'*'*5}")
        # logger.info(f"Processing image: {img_path} with prompt: {pdf_path}")
        # logger.info(f"Image caption: {img_caption}")
        # logger.info(f"seperator: {'*'*5}")


        try:
            img_encoded_format = input_ImgDoubao(os.path.join(pdf_path, img_path))
        except Exception as e:
            print(f"Error encoding image {img_path}: {e}")
            continue
        # print(img_path)
        # response = get_request(input_prompt, img_encoded_format)
        # # response = response.choices[0].message.content
        # print(response)
        # responses.append(response)
        
       
        
        if not os.path.exists(os.path.join(pdf_path, img_path)):
            continue
        if img_path in img_lists:
            print(f"Image {img_path} is in the img_lists, skipping the task.")
            continue
        else:
            # print(f"Image {img_path} is not in the img_lists, adding the task.")
            task = asyncio.create_task(get_request(input_prompt, img_encoded_format,os.path.join(pdf_path, img_path),ark))
            tasks.append(task)
        
        
    # Wait for all tasks to complete
    # print(f"Total tasks: {len(tasks)} initialized for {pdf_path}")
    return tasks
   
if __name__ == "__main__":
    pdf_path = "/Users/Shared/VLReasoningTCL/DataGenerationPrompt/parsingResults/0704.0056.pdf"
    parse_pdf(pdf_path,inddex = 5)