# Academic Paper Question Generation Pipeline Documentation

## Overview
This code implements a pipeline for processing Chinese academic papers in the semiconductor display technology field. It evaluates paper quality, generates technical questions that require logical reasoning, and validates those questions.

## Key Components

### 1. Model Configuration
- Uses various Large Language Models (LLMs) including:
  - QwQ-32B (primary model)
  - Qwen2 models (72B, 32B variants)
  - LLaMA 3.1 70B
- Configured for multi-GPU usage (GPUs 4,5,6,7)
- Maximum context length: 96K tokens

### 2. Main Functions

#### `is_to_drop(text)` and `drop(texts)`
- **Purpose**: Filters out unwanted content from papers
- **Removes**:
  - References and citations
  - Author information and acknowledgments
  - Table of contents
  - Contact information (emails, phone numbers)
  - Non-Chinese content (less than 1% Chinese characters)
  - Thesis metadata (classification numbers, dates, etc.)

#### `judge_md_data(raw_folders, save_paths, jsonl_file_path)`
- **Purpose**: Evaluates if paper content is suitable for generating complex reasoning questions
- **Scoring Criteria**:
  1. **Problem Completeness** (0-2 points)
     - Clear main problem with sufficient clues for answers
     - Multi-author interaction and discussion
  2. **Complexity and Technical Depth** (0-2 points)
     - Graduate level or above content
     - Challenging problems requiring expertise
  3. **Technical Correctness** (-1 to 2 points)
     - Accuracy of technical content
     - Completeness of explanations
  4. **Thinking and Reasoning** (-1 to 3 points)
     - Evidence of logical reasoning
     - Advanced problem-solving approaches
- **Output**: Papers marked as 【是】(suitable) or 【否】(not suitable)

#### `generate_question_data(jsonl_file_input, jsonl_file_output)`
- **Purpose**: Generates 3 high-quality technical questions from suitable papers
- **Requirements for Questions**:
  - Based on logical reasoning content in the paper
  - Clear, complete, and universally understandable
  - No paper-specific references or abbreviations
  - Complete causal chains (e.g., "How does mechanism A affect parameter B, leading to phenomenon C?")
  - Can be understood without reading the paper
- **Forbidden Elements**:
  - Self-referential terms ("this paper", "this study")
  - Isolated concept definitions
  - Hypothetical questions beyond paper scope

#### `convert_questionlist_li_data(jsonl_file_questionlist, jsonl_file_li)`
- **Purpose**: Converts question list format to individual question entries
- **Process**: Splits papers with multiple questions into separate entries

#### `judge_question_data(jsonl_file_input, save_path)`
- **Purpose**: Validates generated questions against quality criteria
- **Evaluation Standards**:
  - **Causality**: Complete technical logic chains
  - **Thoroughness**: Scientific rigor and step-by-step thinking
  - **Completeness**: Self-contained questions independent of the paper
- **Output**: Questions marked as 【是】(valid) or 【否】(invalid)

### 3. Processing Pipeline

1. **Paper Quality Assessment**
   - Read papers from `/mnt/data/MLLM/lilinfeng/code/rl_data/data_txt/`
   - Filter out low-quality content
   - Evaluate suitability for question generation
   - Save results to `judge_txt_output_0401.jsonl`

2. **Question Generation**
   - Generate 3 questions per suitable paper
   - Apply strict formatting and content requirements
   - Save to `generate_question_output_0401.jsonl`

3. **Format Conversion**
   - Convert to individual question entries
   - Save to `generate_question_output_0401_converted.jsonl`

4. **Question Validation**
   - Evaluate each question against quality criteria
   - Save validated questions to final JSON file

### 4. Technical Details

- **Batch Processing**: Processes papers in batches of 32 for efficiency
- **Token Management**: Handles long documents by truncating to model limits
- **Parallel Processing**: Uses VLLM for efficient inference
- **Error Handling**: Gracefully handles JSON parsing errors and oversized inputs

## Usage

Run the pipeline with:
```python
ask(["/mnt/data/MLLM/lilinfeng/code/rl_data/data_txt/"], 
    ["/mnt/data/MLLM/lilinfeng/code/rl_data/0401-data_txt-new.json"])
```

## Output Format

Final output is a JSON file containing:
- Paper ID
- Paper content
- Validated questions suitable for testing logical reasoning in semiconductor display technology