# -*- coding: utf-8 -*-
import os, re
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
import json
from transformers import AutoTokenizer
from tqdm import tqdm
from vllm import LLM, SamplingParams
import numpy as np

questioner = 'qwq_32'
model_len = 96 * 1024

save_steps = 2
batchsize = 32

if questioner == "qw2_72":
    model_path = "/data/lc/openmodels/qw2_72b_instruct"
    model_stopid = "<|im_end|>"
elif questioner == "qw2_72_awq":
    model_path = "/data/lc/openmodels/qw2_72b_instruct_awq"
    model_stopid = "<|im_end|>"
elif questioner == "qw2.5_32":
    model_path = "/data/lc/openmodels/qw2.5_32b_instruct"
    model_stopid = "<|im_end|>"
elif questioner == "qw2.5_72":
    model_path = "/data/lc/openmodels/qw2.5_72b_instruct"
    model_stopid = "<|im_end|>"
elif questioner == "llama3.1_70":
    model_path = "/data/lc/openmodels/llama3.1_70b_instruct"
    model_stopid = "<|eot_id|>"
elif questioner == "qwq_32":
    model_path = '/mnt/workspace/models/Qwen/QwQ-32B/'
    stop_token_ids = [151329, 151336, 151338]

tokenizer = AutoTokenizer.from_pretrained(model_path, rust_remote_code=True)

llm = LLM(model= model_path, trust_remote_code=True, gpu_memory_utilization=0.95, tensor_parallel_size=4, max_model_len = model_len) # decrease max_model_len for oom
sampling_params = SamplingParams(temperature=0.6, repetition_penalty=1.1, min_p=0, top_p= 0.95, top_k=40, max_tokens = 4096, stop_token_ids = stop_token_ids) 

def is_to_drop(text):
    
    text = text.strip()[:10]    
    patterns = ["", "#"]
    for pattern in patterns:
        if text == pattern:
            return True 
    patterns = ['http://www.cnki.net', 'https://www.cnki.net','^\[\d{1,4}\]', '^\*\s+\[\d{1,4}\]', '^\*\s+\(\d{1,4}\)', 
                '^致谢.*[0-9]$', '.*致\s*谢.*','.*目\s*录.*','\.\.\.\.\.\.\.\.', '\…\…\…',r"(http://www|doi:|DOI:|please contact)",
                r"(work was supported by|study was supported by|China|Republic of Korea|Authorized licensed use limited to)",
                r"\s[1-9]\d{5}(?!\d)",  # 邮编
                r"\w+([-+.]\w+)*@\w+([-.]\w+)*\.\w+([-.]\w+)*", r"(received in revised form|All rights reserved|©)", r"[a-zA-z]+://[^\s]*",
                r"(13[0-9]|14[5|7]|15[0|1|2|3|5|6|7|8|9]|18[0|1|2|3|5|6|7|8|9])\d{8}", r"\d{3}-\d{8}|\d{4}-\d{7}",
                '^分\s*类\s*号', '^学\s*科\s*专\s*业', '^签\s*字\s*日\s*期', '^申\s*请\s*人\s*员\s*姓\s*名',
                '^日\s*期', '^指\s*定\s*教\s*师', '学\s*位\s*论\s*文', '^工\s*作\s*单\s*位', '^电\s*话', '^通讯地址', '^邮\s*编', 
                '^中\s*图\s*分\s*类\s*号', '^评\s*阅\s*人', '^签\s*名', '^分\s*类\s*号', '^密\s*级', '^学\s*号', '^院\s*系', 
                '^委\s*员', '^国内图书分类号', '^国际图书分类号', '^导\s*师', '^申\s*请\s*学\s*位', '^工\s*程\s*领\s*域', '^所\s*在\s*单\s*位', 
                '^答\s*辩',  '^作\s*者', '^专\s*业', '^保\s*密', '^不\s*保\s*密', '^硕\s*土\s*姓\s*名', '^导\s*师', '^职\s*称', '^声\s*明', 
                '^申请学位', '^学科、专业', '^学\s*校\s*代\s*码', '^邢\s*坤\s*太\s*学', '^学\s*科\s*门\s*类', '^培\s*养\s*院\s*系',
                '^研\s*究\s*生', '^专\s*业', '^完\s*成\s*日\s*期', '^年\s*月\s*日', '^审\s*级', '^单\s*位\s*代\s*码', '^密\s*码', 
                '^学\s*位\s*授\s*予', '^校\s*址', '^授\s*予', '^论\s*文\s*分\s*类\s*号', '^研\s*突\s*生', '^研\s*究\s*方\s*向:', 
                '^研\s*究\s*生', '^学\s*校\s*代\s*号', '^主\s*席', '^U\s*D\s*C', '^U.D.C','^论\s*文\s*起\s*止', '^论\s*文\s*样\s*纸', 
                '^完\s*成\s*时\s*间', '^学\s*校\s*编\s*码', '^声\s*明\s*人', '^分\s*类\s*号', '^培\s*养\s*单\s*位', '^提\s*交\s*论\s*文', 
                '^资\s*助', '^学科(专业)', '^提\s*交\s*日\s*期', '^学\s*科\s*名\s*称', '^课\s*题\s*人', '^学\s*科\s*门\s*类', 
                '^一\s*级\s*学\s*科', '^学\s*位\s*申\s*请', '^学\s*院\s*名\s*称', '^主\s*任', '^院\s*系', '^专\s*业', '^姓\s*名', 
                '^完\s*成\s*日\s*期', '^作\s*者', '^申\s*请\s*学\s*位', '^工\s*程\s*领\s*域', '^学\s*科\s*名\s*称', '^领\s*域', '^学\s*院', 
                '^提\s*交\s*日\s*期', '^授\s*予\s*学\s*位', '^学\s*科', '^所\s*在\s*单\s*位',  '^电\s*子\s*邮\s*箱', '^联\s*系\s*地\s*址',
#                r'^!\[\](images/.*',  # 多余（可在检查有无中文字符时去掉）且导致报错
                r'^\[?\d+\]?',  r'^\s*\[?\d+\]?', r'^\［?\d+\］?', r'^\s*\［?\d+\］?' # mineru解析的参考文献格式
                ]
    for pattern in patterns:
        if re.search(pattern, text):
            return True
        
    patterns = ['申请号|专利号|已录用|学报|研究生|已收录|攻读|第一作者|第二作者|参考文献|专业名称|863项目|导师',
                '教授|感谢|致谢|谢谢|指导|朋友|家人|亲友|师弟|师妹|老师|同学|父母|充实|答辩|祝愿|独创性声明|作者签名',
                '发表文章|论文使用授权声明|本人|知网|论文使用权|发表的论文|申请的专利|申请专利|发表的文章|发表学术论文|发表论文',
                '参与科研项目|作者简介|三年的学习|大学硕士学位论文|大学博士学位论文|涉密论文|学校代码|论文提交日期|委员：|中图分类号',
                '原创性声明|顺利完成学业|All rights reserved|参 考 文 献|参考文献|所在学院|国家自然科学基金|教育部重点学科建设',
                '时间飞梭|时光飞梭|光阴似箭|白驹过隙|论文版权|本学位论文|使用授权书|References|Acknowledgements',
                '论文著作权|保密的学位论文|中国第一所现代大学|参加科研情况|独 创 性 声 明|论文使用授权|获得的专利|家庭的爱|文献标识码|文章编号'
                ]
    for pattern in patterns:
        if re.findall(pattern, text):
            return True   
        
    """
    判断是否不包含中文字符（暂时把公式也去掉）
    """
    num = 0
    for t in text:
        if  '\u4e00' <= t <= '\u9fa5':
            num += 1    
    if num / len(text) < 0.01:
        return True
                
    return False

def drop(texts, concatenation= "\n"):
    new_texts = []
    texts = texts.split("\n")
    for i, text in enumerate(texts):
        if not is_to_drop(text):
            new_texts.append(text)
    return concatenation.join(new_texts)

def extract(folder):
    files = os.listdir(folder)
    files.sort()
    return files

def load_json(file_path):
    with open(file_path, "r+", encoding="utf8") as load_f:
        dicts = json.load(load_f)
    dicts.sort(key=lambda s: s["id"])
    return dicts

def load_paper(file_path):
    with open(file_path, "r", encoding="utf8") as f:
        content = f.read()
        deal_content = drop(content) 
        
    return deal_content

def dcts2json(dcts, save_path):
    with open(save_path[0], 'w', encoding='utf8') as f:
        json.dump(dcts , f, indent=4, ensure_ascii=False) 

def to_batch(lst, groupsize):  # [a,b,c,d,e] -> [[a,b], [c,d], [e]], for batch inference
    return [lst[i:i+groupsize] for i in range(0,len(lst),groupsize)]


def judge_md_data(raw_folders, save_paths, jsonl_file_path):

    score_template = """你的任务是依据以下评分规则对文本质量进行打分，并输出最终得分。评分流程如下：
1.依照每个标准依次评估文本。对每个子问题如实作答。若对某子问题答案为明确 "是"，则按标准相应加分或减分;
2.记录每个标准的累计分数，得出总分;
3.依据以下说明，将最终评估结果整理为有效的 JSON 对象。

## 评分标准：
1.标准 1：问题完整性
(1) 内容无清晰主要问题，或缺乏足够线索得出正确答案，得 0 分。
(2) 内容包含一个主要问题，且有足够线索得出正确答案，得 + 1 分。
(3) 文本体现多位作者间互动与讨论，如提出答案、评估反思答案、回应批评、修订编辑答案，得 + 1 分。
2.标准 2：问题复杂性和技术深度
(1) 内容难度为大学水平或以下，得 0 分。
(2) 内容难度为研究生水平或以上，仅领域专家能理解，得 + 1 分。
(3) 所讨论问题极具挑战性，高技能非专家花费 30 分钟上网搜索或阅读文献后，仍无法完全理解问题或给出正确答案，得 + 1 分。
3.标准 3：技术正确性和准确性
(1) 文本含显著技术错误或不准确，得 -1 分。
(2) 文本有一定技术正确性，但存在明显缺陷或遗漏（如单位错误、推导不完整），得 0 分。
(3) 文本技术正确，但有小缺陷或遗漏（如小代数错误、解释不完整），得 + 0.5 分。
(4) 文本技术高度正确，解释清晰准确（如精确定义、完整推导），得 + 0.5 分。
(5) 文本技术卓越正确，解释严格精确（如形式化证明、精确计算），得 + 1 分。
4.标准 4：思维和推理
(1) 文本无任何思维或推理迹象，得 -1 分。
(2) 文本展现一些基本思维和推理能力（如直接应用已知技术、简单分析问题），得 + 0.5 分。
(3) 文本展现一定思维和推理能力（如考虑多种解决方法、讨论不同方案权衡），得 + 0.5 分。
(4) 文本展现显著思维和推理能力（如通过多步推理链解决复杂问题、运用专业科学领域高级推理模式），得 + 1 分。
(5) 文本展现卓越思维和推理能力（如以高度创新方式解决专业领域复杂问题、结合多种推理技术对问题进行新抽象），得 + 1 分。

最终评判标准：若各项标准得分均大于零，且标准 4 得分大于等于 1 分，则该文本内容适合生成逻辑推理问题。

[文本内容的开始]
{academic_paper}
[文本内容的结束]

格式要求：只输出文本内容是否适合生成复杂推理问题，不输出任何别的内容。并且是否适合严格按照以下格式进行输出：
【是】或者【否】。不要输出为空，不要输出其他内容，输出是或否时，要带上【】符号进行输出。


""" 
    for raw_folder, save_path in zip(raw_folders, save_paths):
        files = extract(raw_folder) # txt _files
        results = []
        already_ids = []
        to_do = []
        if os.path.exists(save_path):
            results = load_json(save_path)
            for already_sample in results:
                already_ids.append(already_sample["id"])
        print(len(already_ids),  len(results))
        
        for file in tqdm(files, desc = "check:" + raw_folder):
            if file.endswith("txt"):
                if not (file in already_ids):
                    to_do.append(file)             
        batches = to_batch(to_do, 32)
        comply_stand_datas = []
        for batch in tqdm(batches, desc = "judge:" + raw_folder):
            score_inputs = []
            for paper in batch:
                paper_name = paper.split("_part")[0]
                paper_content = load_paper(raw_folder + paper)
                # print(paper_name)
                # print(paper_content[-20:])
            
                score_prompt = score_template.replace("{academic_paper}", paper_content)
                score_messages = [{"role": "system", "content": "你是一个乐于助人的半导体显示技术领域的专家。"}, {"role": "user", "content": score_prompt}]

                data_li = tokenizer.apply_chat_template(score_messages,tokenize=False,add_generation_prompt=True, truncation=True)
                if (len(tokenizer.encode(data_li)) > 98304):
                    print(len(tokenizer.encode(data_li)))
                    score_prompt = score_prompt[:98304]
                    score_messages = [{"role": "system", "content": "你是一个乐于助人的半导体显示技术领域的专家。"}, {"role": "user", "content": score_prompt}]
                    data_li = tokenizer.apply_chat_template(score_messages,tokenize=False,add_generation_prompt=True, truncation=True)

                score_inputs.append(data_li)
                # if len(tokenizer.encode(score_prompt)) < model_len - 1024:
                #     score_inputs.append(tokenizer.apply_chat_template(score_messages,tokenize=False,add_generation_prompt=True, truncation=True))
                # else:
                #     encoded_prompt = tokenizer.encode(score_prompt)
                #     encoded_prompt = encoded_prompt[:model_len - 1024]
                #     truncated_prompt = tokenizer.decode(encoded_prompt)
                #     truncated_prompt = [{"role": "system", "content": "你是一个乐于助人的半导体显示技术领域的专家。"}, {"role": "user", "content": truncated_prompt}]
                #     score_inputs.append(tokenizer.apply_chat_template(truncated_prompt,tokenize=False,add_generation_prompt=True))

            score_outputs = llm.generate(score_inputs, sampling_params, use_tqdm = False)
            
            for index, paper in enumerate(batch):
                paper_name = paper.split("_part")[0]
                paper_content = load_paper(raw_folder + paper)
                
                score_text = score_outputs[index].outputs[0].text.split('\n')[-1]

                if '【是】' in score_text:
                    comply_stand_datas.append({"stats":1, "paper_name": paper_name, "paper_content": paper_content, "score_text":score_text})
                else:
                    comply_stand_datas.append({"stats":0, "paper_name": paper_name, "paper_content": paper_content, "score_text":score_text})


        with open(jsonl_file_path, "w", encoding="utf-8") as f:
            for item in comply_stand_datas:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        print(f"文本质量评测过程数据已成功存储到 {jsonl_file_path}")
    


def generate_question_data(jsonl_file_input, jsonl_file_output):
    prompt_template1 = """你是一位半导体显示技术领域的资深专家，擅长从技术文献中提炼核心知识点。你的职责是从论文中生成问题和相应的答案，问题和相应的答案对需要提供给资深的人员学习，问题和相应的答案的质量要高。请根据输入的学术论文内容，生成3个需要逻辑推理才能解答的高质量技术问题，请确保这些问题能够直接从论文中找到答案。这些问题将用于资深研究人员的专业能力评估，需满足以下要求：
##【核心要求】
### 问题设计准则：
(1) 仔细通读全文，找出涉及逻辑推理的文本部分，据此设计相关问题。
(2) 问题要基于论文里的技术原理，描述务必清晰、全面、明确。问题中主语或名词的表述要精准、全面且通用。
(3) 避免在问题中引用文献或文章自定义的专有名词。利用自身半导体显示领域知识，生成通用问题，确保不读论文也能理解问题含义。
(4) 问题中的名词描述必须和论文一致，不能缩写。比如论文是 "OLED 材料"，问题不能写成 "材料"；论文是 "LTPS 器件"，问题不能写成 "器件"。
(5) 提问不能针对论文里的某个特定示例。要让顶尖科学家不读论文也能理解并回答问题。问题里不能有 "书本""论文""本文""本实验" 等类似表述。； 
(6) 确保生成问题完整，与论文彻底解耦，不依赖论文内容。若问题有背景信息，必须阐述清楚。
(7) 问题简洁：生成的问题要凝练、简洁。
### 问题设计的科学严谨性：
(1) 因果链：问题需呈现完整技术逻辑链，比如 "机制 A 怎样影响参数 B，进而引发现象 C"。
(2) 周密性：思考过程要科学严谨，逐步推进。保证问题及对应答案源自论文内容，且答案在论文中有详细阐述。

## 【禁止事项】
× 禁止使用"本文/本研究/本实验"等论文自指表述
× 禁止提问孤立概念（如：XX技术的定义是什么）
× 禁止超出论文技术范围的假设性问题

##【格式要求】
用中文输出。当前阶段只设计问题，不输出答案。输出问题前必须用 </think> 结束思考后在输出问题。严格按照以下格式输出你设计的问题：
[[1]] 第1个问题
[[2]] 第2个问题
[[3]] 第3个问题 

[学术论文的开始]
{academic_paper}
[学术论文的结束]
"""

    prompt_template = """你是一位半导体显示技术领域的资深专家，擅长从技术文献中提炼核心知识点。你的职责是从论文中生成问题和相应的答案，问题和相应的答案对需要提供给资深的人员学习，问题和相应的答案的质量要高。请根据输入的学术论文内容，生成3个需要逻辑推理才能解答的高质量技术问题，请确保这些问题能够直接从论文中找到答案。这些问题将用于资深研究人员的专业能力评估，需满足以下要求：
【核心要求】
问题设计准则：
a) 首先你需要阅读全文，并判断哪些文本中涉及到逻辑推理的内容。然后你需要根据逻辑推理的内容设计相应的问题；
b) 问题必须基于论文中的技术原理进行设计，问题的描述必须明确清晰全面，问题中主语或名词的描述必须要精准、全面且具备通用性；
c) 问题中请不要引用文献或者文章定义的专有名词，请结合你自身半导体的显示领域的知识，将生成普适通用的问题，在不阅读论文的情况也能正常理解问题所表达的含义；
d) 问题中的名词描述不可以缩写，需要与论文中的描述一致。例如论文中提到的是"OLED材料"，问题中不能简化为"材料"。例如论文中提到的是"LTPS器件"，问题中不能简化为"器件"；
e) 不要针对于论文中的某个特定示例进行提问，问题尽量使顶尖科学家在不阅读论文的情况下也能理解和回答。且问题不能包含"书本"、"论文"、"本文"、"本实验"等相关信息； 
f) 保证问题的完整性，且完全不依赖论文内容，确保问题与论文完全解耦。若问题带有背景信息，一定要阐述清楚背景情况。

科学严谨性：
a) 因果链：问题需呈现完整技术逻辑链（如：机制A如何影响参数B，进而导致现象C）
b) 周密性：过程需要科学严谨，逐步思考，确保问题和对应的答案来源于论文的内容。且答案需要能在论文中完全找到详细的描述。
问题简洁：问题要凝练简洁。

【禁止事项】
× 禁止使用"本文/本研究/本实验"等论文自指表述
× 禁止提问孤立概念（如：XX技术的定义是什么）
× 禁止超出论文技术范围的假设性问题

【格式要求】：用中文输出。当前阶段只设计问题，不输出答案。严格按照以下格式输出你设计的问题：
[[1]] 第1个问题
[[2]] 第2个问题
[[3]] 第3个问题 

[学术论文的开始]
{academic_paper}
[学术论文的结束]
"""
    with open(jsonl_file_input, 'r', encoding='utf-8') as file: 
        inputs = []
        for line in file:
            line_data = json.loads(line)
            if line_data["stats"] == 1:
                paper_name = line_data["paper_name"]
                paper_content = line_data["paper_content"]

                generate_prompt = prompt_template.replace("{academic_papername}", paper_name).replace("{academic_paper}", paper_content)
                messages = [{"role": "system", "content": "你是一个乐于助人的半导体显示技术领域的专家。"}, {"role": "user", "content": generate_prompt}]

                #if len(tokenizer.encode(generate_prompt)) < model_len - 1024:
                inputs.append(tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True))
                    
        batches_gen = to_batch(inputs, 32)
           
        input_tag= 0
        for batch in tqdm(batches_gen):
            tag = 0
            questions_infos = []
            outputs = llm.generate(batch, sampling_params, use_tqdm = False)
            for i, score_output in enumerate(outputs):
                question_texts =  score_output.outputs[0].text
                if "</think>" in question_texts:
                    question_list = score_output.outputs[0].text.split("</think>")[1].strip().split('\n')
                else:
                    question_list = []
                print('question_list', question_list)
                questions_infos.append(question_list)

            with open(jsonl_file_input, 'r', encoding='utf-8') as f_in:
                res_data = []
                for index, line in enumerate(f_in):
                    if input_tag != index:
                        continue
                    line_data = json.loads(line)
                    if line_data["stats"] == 1:
                        if tag == batchsize:
                            break
                        line_data["question_list"] = questions_infos[tag]
                        tag += 1
                    else:
                        line_data["question_list"] = []
                    input_tag += 1
                    res_data.append(line_data)
            with open(jsonl_file_output, "a", encoding="utf-8") as f_out:
                for item in res_data:
                    f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
        

    print(f"文本生成过程数据已成功存储到 {jsonl_file_output}")


def convert_questionlist_li_data(jsonl_file_questionlist, jsonl_file_li):
    import json
    import logging
    from pathlib import Path
    """处理 JSONL 文件并生成新文件"""
    
    
    with open(jsonl_file_questionlist, 'r', encoding='utf-8') as infile, \
         open(jsonl_file_li, 'w', encoding='utf-8') as outfile:
        
        for idx, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                # 解析原始数据
                original_data = json.loads(line)
                
                # 示例：提取需要的字段并修改
                stats = original_data["stats"]
                paper_name = original_data["paper_name"]
                paper_content = original_data["paper_content"]
                score_text = original_data["score_text"]
                question_list = original_data["question_list"]
                for question_li in question_list:
                    processed_data = {
                        "stats": stats,
                        "paper_name": paper_name,
                        "paper_content": paper_content,
                        "score_text": score_text,
                        "question_li": question_li
                    }
                    
                    # 写入新文件
                    json.dump(processed_data, outfile, ensure_ascii=False)
                    outfile.write('\n')
                
                logging.debug(f"处理条目 {idx} 成功")
                
            except json.JSONDecodeError as e:
                logging.warning(f"第 {idx} 行解析失败: {e}")
            except Exception as e:
                logging.error(f"处理条目 {idx} 时发生未知错误: {e}")
        print("处理完成！")


def judge_question_data(jsonl_file_input, save_path):
    evaluator_template = """您是一位专家评估员，负责决定问题是否符合推理问题标准。您的评估必须结合给定文章内容和给定问题判断。
## 【评估标准】
### 因果性：
(1) 问题应展现出完整的技术逻辑链。比如，类似 "机制 A 怎样影响参数 B，最终致使现象 C 出现" 这种形式。
### 周密性：
(1) 思维过程要科学且严谨，需逐步思考。问题及对应的答案必须源于论文内容，且答案在论文中要有详细描述。
### 完整性：
(1) 问题是否全面涵盖文章相关内容的各个方面？
(2) 问题描述应简洁凝练，语义完整。
(3) 问题要与文章内容完全独立，不依赖文章也能被清晰理解，即问题需完整、自足。


[文章内容的开始]
{academic_paper}
[文章内容的结束]

[问题内容]
{academic_question}

格式要求：仅输出文本内容生成的问题是否符合标准，严格按以下格式，有且仅输出【是】或者【否】，不输出任何别的内容，不能输出为空，输出是或否时，要带上【】符号进行输出。用中文输出，严格按照以下格式进行输出：【是】或者【否】
"""
    with open(jsonl_file_input, 'r', encoding='utf-8') as file: 
        evaluator_inputs = []
        for line in file:
            line_data = json.loads(line)
            if line_data["stats"] == 0: continue
            paper_content = line_data["paper_content"]
            question_li = line_data["question_li"]


            evaluator_prompt = evaluator_template.replace("{academic_question}", question_li).replace("{academic_paper}", paper_content)
            evaluator_messages = [{"role": "system", "content": "你是一个乐于助人的半导体显示技术领域的专家。"}, {"role": "user", "content": evaluator_prompt}]
            # if len(tokenizer.encode(evaluator_prompt)) < model_len - 1024:
            evaluator_inputs.append(tokenizer.apply_chat_template(evaluator_messages,tokenize=False,add_generation_prompt=True))

    batches_gen = to_batch(evaluator_inputs, batchsize)

    input_tag= 0
    for batch in tqdm(batches_gen):
        tag = 0
        evaluator_outputs = []
        outputs = llm.generate(batch, sampling_params, use_tqdm = False)
        for i, evaluator_output in enumerate(outputs):
            evaluator_text = evaluator_output.outputs[0].text.split('\n')[-1]
            print('evaluator_text', i, evaluator_text) 
            evaluator_outputs.append(evaluator_text)

        with open(jsonl_file_input, 'r', encoding='utf-8') as f_in:
            results = []
            for index, line in enumerate(f_in):
                if input_tag != index:
                    continue
                line_data = json.loads(line)
                sample = dict()
                if line_data["stats"] == 1:
                    if tag == batchsize:
                        break
                    if '【是】' in evaluator_outputs[tag]:
                        sample["id"] = line_data["paper_name"]
                        sample["paper_content"] = line_data["paper_content"]
                        sample["question_li"] = line_data["question_li"]
                        results.append(sample)
                    tag += 1
                input_tag += 1
               
        with open(save_path[0], 'a', encoding='utf8') as f_res:
            json.dump(results, f_res, indent=4, ensure_ascii=False)      
                
                 
def ask(raw_folders, save_paths):
    judge_md_output_path = "/mnt/data/MLLM/lilinfeng/code/rl_data/process_data/data_txt/judge_txt_output_0401.jsonl"
    generate_question_output_path = "/mnt/data/MLLM/lilinfeng/code/rl_data/process_data/data_txt/generate_question_output_0401.jsonl"
    generate_question_li_output_path = "/mnt/data/MLLM/lilinfeng/code/rl_data/process_data/generate_question_output_0401_converted.jsonl"
    judge_md_data(raw_folders, save_paths, judge_md_output_path)
    generate_question_data(judge_md_output_path, generate_question_output_path)
    convert_questionlist_li_data(generate_question_output_path, generate_question_li_output_path)
    judge_question_data(generate_question_li_output_path, save_paths)


if __name__ == "__main__":             
    ask(["/mnt/data/MLLM/lilinfeng/code/rl_data/data_txt/"], ["/mnt/data/MLLM/lilinfeng/code/rl_data/0401-data_txt-new.json"])