import base64

def img2base64(image_path):
    with open(image_path, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
        image_type = image_path.split(".")[-1]
        
    return f"data:image/{image_type};base64,{image_base64}"
def generate_pages_range(path):
    import os
    import json
    files = os.listdir(path)
    files = [file for file in files if file.endswith('_content_list.json')]
    file = files[0] if files else None
    file_path = os.path.join(path, file) if file else None
    if file_path is None or not os.path.exists(file_path):
        print(f"No valid content list file found in {path}.")
        return 0
    
    pages = []
    start_page = None
    end_page = None
    
    if file_path and os.path.exists(file_path):
        data = json.load(open(file_path, 'r', encoding='utf-8'))
    
    for item in data:
        if item["type"] == "image":
            if start_page is None:
                if len(item["img_caption"]) > 0:
                    start_page = item["page_idx"]
            end_page = item["page_idx"] 
    if start_page is not None and end_page is not None:
        pages = {"pages":[[start_page, end_page]]}
        file_name = "pages.json"
        file_path = os.path.join(path, file_name)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(pages, f, ensure_ascii=False, indent=4)
        return 1
    else:
        return 0
        