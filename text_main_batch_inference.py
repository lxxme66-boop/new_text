from TextGeneration.Datageneration import *
import asyncio
import time
import pickle as pkl
import argparse
import json
import random
import os

async def process_folders(folders, txt_path, temporary_folder, index=9, maximum_tasks=20, selected_task_number=500):
    total_tasks = []
    if args.read_hist and os.path.exists(os.path.join(storage_folder, "total_response.json")):
        print(f"Reading history data from {storage_folder}/total_response.json")
        history_data = json.load(open(os.path.join(storage_folder, "total_response.json"), "r", encoding="utf-8"))
        processed_files = [example["source_file"] for example in history_data if "source_file" in example]
        processed_files = list(set(processed_files))  # remove duplicates
    else:
        processed_files = []
    
    print(f"Already processed files: {len(processed_files) if processed_files else 'None'}")
    await asyncio.sleep(1)
    
    total_responses = []
    for folder in folders:
        file_path = os.path.join(txt_path, folder)
        
        # Skip if not a txt file
        if not folder.endswith('.txt'):
            continue
            
        # Skip if already processed
        if folder in processed_files:
            print(f"File {folder} already processed, skipping.")
            continue
            
        if not os.path.isfile(file_path):
            continue
            
        try:
            # Process txt file
            tasks = await parse_txt(file_path, index=index)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
            
        if len(tasks) == 0:
            print(f"No tasks found in {file_path}, skipping this file.")
            continue
            
        total_tasks.extend(tasks)
        
        if len(total_tasks) >= maximum_tasks:
            print(f"Total tasks {len(total_tasks)} exceeds the maximum_tasks {maximum_tasks}, waiting for tasks to complete.")
            batch_responses = await asyncio.gather(*total_tasks)
            # Filter out None responses
            batch_responses = [response for response in batch_responses if response is not None]
            total_responses.extend(batch_responses)
            total_tasks = []  # reset the tasks list
            
            os.makedirs(os.path.join(storage_folder, temporary_folder), exist_ok=True)
            with open(os.path.join(storage_folder, temporary_folder, f"batch_{len(total_responses)//maximum_tasks + 1}.json"), "w",
                encoding="utf-8") as f:
                json.dump(batch_responses, f, ensure_ascii=False, indent=4)
                print(f"Batch {len(total_responses)//maximum_tasks + 1} responses saved to {storage_folder}/{temporary_folder}/batch_{len(total_responses)//maximum_tasks + 1}.json")
                
        # Check if we have enough responses
        if len(total_responses) >= selected_task_number:
            print(f"Total responses {len(total_responses)} exceeds the selected_task_number {selected_task_number}, stopping the process.")
            break
            
    # Process remaining tasks
    if total_tasks:
        batch_responses = await asyncio.gather(*total_tasks)
        batch_responses = [response for response in batch_responses if response is not None]
        total_responses.extend(batch_responses)
        
    return total_responses


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process text files for QA generation.")
    parser.add_argument("--index", type=int, default=43, help="index for the task")
    parser.add_argument("--parallel_batch_size", type=int, default=100, help="number of parallel tasks")
    parser.add_argument("--txt_path", type=str, default="/workspace/text_qa_generation/data/input_texts", help="path to the txt folder")
    parser.add_argument("--storage_folder", type=str, default="/workspace/text_qa_generation/data/output", help="path to the storage folder")
    parser.add_argument("--temporary_folder", type=str, default="TEMP", help="path to the temporary folder")
    parser.add_argument("--selected_task_number", type=int, default=1000, help="number of tasks to select")
    parser.add_argument("--read_hist", type=bool, default=False, help="whether to read the history file")
    args = parser.parse_args()
    
    selected_task_number = args.selected_task_number
    index = args.index
    parallel_batch_size = args.parallel_batch_size
    txt_path = args.txt_path
    storage_folder = args.storage_folder
    os.makedirs(storage_folder, exist_ok=True)
    temporary_folder = args.temporary_folder
    
    # Get all txt files
    files = [f for f in os.listdir(txt_path) if f.endswith('.txt')]
    
    total_responses = []
    os.makedirs(storage_folder, exist_ok=True)
    
    final_results = asyncio.run(process_folders(files, txt_path, temporary_folder, index=index, 
                                               maximum_tasks=parallel_batch_size,
                                               selected_task_number=selected_task_number))
    
    # Save results
    f = open(os.path.join(storage_folder, "total_response.pkl"), "wb")
    if args.read_hist:
        history_data = json.load(open(os.path.join(storage_folder, "total_response.json"), "r", encoding="utf-8"))
        final_results.extend(history_data)
        pkl.dump(final_results, f)
    else:
        pkl.dump(final_results, f)
        
    print(f"Total {len(final_results)} responses saved to {storage_folder}/total_response.pkl")