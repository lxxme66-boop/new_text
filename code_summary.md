# Code Summary: Academic Paper Question Generator

## What This Code Does

This Python script creates an automated pipeline for generating high-quality technical questions from Chinese academic papers in the semiconductor display technology field. 

### Main Purpose
Transform academic papers → Evaluate quality → Generate reasoning questions → Validate questions

### Key Features

1. **Paper Filtering**: Removes irrelevant content like references, author info, and metadata
2. **Quality Assessment**: Scores papers on complexity, technical depth, and reasoning potential
3. **Question Generation**: Creates 3 logical reasoning questions per suitable paper
4. **Question Validation**: Ensures questions meet strict quality standards

### The Process

```
Input Papers (.txt files)
    ↓
Filter & Clean Content
    ↓
Evaluate Paper Quality (【是】/【否】)
    ↓
Generate 3 Questions per Suitable Paper
    ↓
Validate Each Question
    ↓
Output: JSON with Validated Questions
```

### Key Requirements for Generated Questions
- Must require logical reasoning to answer
- Independent of the paper (understandable without reading it)
- Show complete causal chains
- Graduate-level complexity
- No paper-specific references

### Technical Stack
- **Model**: QwQ-32B (or other large language models)
- **Framework**: VLLM for efficient inference
- **Processing**: Batch processing (32 papers at a time)
- **GPUs**: 4 GPUs in parallel

This tool is designed for creating high-quality training/evaluation datasets for AI models in the semiconductor display technology domain.