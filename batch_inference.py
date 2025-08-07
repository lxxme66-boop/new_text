import asyncio
import json
import multiprocessing
from typing import Any, Dict, Optional
import httpx
from volcenginesdkarkruntime import AsyncArk
from volcenginesdkarkruntime._constants import CLIENT_REQUEST_HEADER
api_key = "ae37bba4-73be-4c22-b1a7-c6b1f5ec3a4b"
def main(
    num_worker_processes: int,
    max_concurrency_per_process: int,
    model: str,
    api_key: Optional[str] = None,
) -> None:
    """
    多进程处理火山方舟批量推理接入点的并发请求。启动 `num_worker_processes + 2` 个进程，其中一个进程用于读取输入，另一个进程用于处理输出，
    其余进程用于任务处理并将结果放置在输出队列中。
    如果您要在实际应用中使用，需要根据您的实际情况调整 `read_input` 和 `write_output` 的实现。
    这里为了通用性考虑，使用单独的进程分别处理输入和输出，实际应用中可以根据需要进行调整。尽量减少进程间的通信，使用进程内的通信机制。
    例如，如果您的输入是文件读取的，您可以先将文件分为 `num_worker_processes` 个规模接近的文件，然后在每个 worker 进程中读取一个文件。
    同样地，如果您的输出是写入文件的，您可以每个worker 进程中都将结果写入一个文件，最终再将所有输出文件合并。
    又例如，如果您的输入从消息队列中读取，您可以使用同一个 consumer group 分别在每个 worker 进程中读取消息。
    同样地，如果您的输出是写入到消息队列，您可以在每个 worker 进程将处理结果直接写入。
    Args:
        num_worker_processes (int): 启动的子进程数，建议初始设置为 cpu 核心数的 2~4 倍
        max_concurrency_per_process (int): 每个子进程的最大并发数，我自己测试单进程比较合理的值是 256/512, 根据实际情况调整
        model (str): 方舟的 Endpoint ID, 可以在控制台查看
        api_key (str): 方舟的 API Key, 可以在控制台查看，建议直接在环境变量中设置，避免泄漏
    """
    processes = []
    in_queue: "multiprocessing.Queue[Optional[Dict[str, Any]]]" = multiprocessing.Queue(
        maxsize=512
    )
    out_queue: "multiprocessing.Queue[Optional[Dict[str, Any]]]" = (
        multiprocessing.Queue(maxsize=512)
    )
    # 启动 `num_worker_processes` 个子进程，每个子进程都从输入队列中读取输入，并将结果放置在输出队列中
    for i in range(num_worker_processes):
        p = multiprocessing.Process(
            target=process,
            args=(
                i,
                max_concurrency_per_process,
                api_key,
                in_queue,
                out_queue,
            ),
        )
        p.start()
        processes.append(p)
    # 启动子进程后，使用另一个子进程读取输入，并放置在输入队列中
    # 这里的 `read_input` 函数应该是您自己的函数，用于从文件、消息队列或数据库中读取输入
    # 在输入读取完毕后，将 `num_worker_processes` 个 None 放置在输入队列中，以通知子进程停止
    print("start read input")
    p = multiprocessing.Process(
        target=read_input,
        args=(in_queue, model, num_worker_processes),
    )
    p.start() # 读取过程需要时间，所以不必等待数据读取完成 也可以使用 read_input 函数直接在主进程中读取输入
    # read_input(in_queue, model, num_worker_processes)
    
    processes.append(p)
    # 在主进程从输出队列中读取输出，并将其打印
    # 在实际应用中，您可能需要将输出写入文件、消息队列或数据库
    finished_processes = 0
    print("start write output")
    while True:
        result = out_queue.get()
        if result is None:
            finished_processes += 1
            if finished_processes == num_worker_processes:
                break
            continue
        # 这里应该是您自己的函数，用于将输出写入文件、消息队列或数据库
        # 我们这里只是简单地打印输出
        write_output(result)
    # 等待所有子进程结束
    for p in processes:
        p.join()
# 构造 10000 个输入对象
def read_input(
    in_queue: "multiprocessing.Queue[Optional[Dict[str, Any]]]",
    model: str,
    num_worker_processes: int,
) -> None:
    def encode_image(image_path: str) -> str:
        import base64
        img_type = image_path.split('.')[-1]  # 获取图片类型
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:image/{img_type};base64,{encoded_string}"
    for _ in range(8): # 总任务次数
        in_queue.put(
            {
                "model": model,
                "messages": [{"role": "user", "content": 
                             [  
                              
                                {"type":"image_url","image_url":{
                                    "url": encode_image("/Users/Shared/VLReasoningTCL/data/image/0705.0176/0705.0176_page_3_img_1.png")
                                } },
                                {"type": "text", "text": "用一句话描述这张图片的内容"},   
                             ]}],
                "extra_headers": {
                    # 可以在 extra_headers 中设置 CLIENT_REQUEST_HEADER，用于标识请求的唯一性
                    # 这样可以串联 client 和 server 的日志，方便排查问题
                    CLIENT_REQUEST_HEADER: "<your_unique_id_for_request>"
                },
            }
        )
    for _ in range(num_worker_processes):
        in_queue.put(None) # 向输入队列中放入 None，通知子进程停止
# 简单地将输出打印到控制台
def write_output(
    result: Dict[str, Any],
) -> None:
    print(json.dumps(result, ensure_ascii=False))
def process(
    worker_id: int,
    max_concurrency: int,
    api_key: Optional[str],
    in_queue: "multiprocessing.Queue[Optional[Dict[str, Any]]]",
    out_queue: "multiprocessing.Queue[Optional[Dict[str, Any]]]",
) -> None:
    loop = asyncio.new_event_loop() # 设置新的协程事件循环
    asyncio.set_event_loop(loop=loop) # 设置当前协程事件循环
    print(f"process {worker_id} start")
    http_client = make_http_client(max_concurrency)
    client = AsyncArk(api_key=api_key, http_client=http_client)
    async def inner() -> None:
        sem = asyncio.Semaphore(max_concurrency) # 设置并发锁, 每个进程的最大并发数
        tasks: "list[asyncio.Task[None]]" = []
        while True:
            record = await asyncio.to_thread(in_queue.get)
            if record is None:
                break
            await sem.acquire() # 获取信号量，控制并发数
            # 启动一个协程任务，处理输入队列中的数据
            tasks.append(
                loop.create_task(  # 创建协程任务
                    worker(
                        client,
                        record,
                        out_queue,
                        sem,
                    )
                )
            )
        await asyncio.gather(*tasks) # 等待所有任务完成
        await http_client.aclose() 
        out_queue.put(None) # 放入输出队列，所有进程都可获取的通信管路，通知主进程所有任务已完成
    loop.run_until_complete(inner())
async def worker(
    client: AsyncArk,
    record: Dict[str, Any],
    out: "multiprocessing.Queue[Optional[Dict[str, Any]]]",
    sem: asyncio.Semaphore,
) -> None:
    try:
        result = await client.batch_chat.completions.create(**record)
        out.put(result.to_dict())
    except BaseException as e:
        print(f"process failed: {e}")
    finally:
        sem.release()
def make_http_client(max_concurrency: int) -> httpx.AsyncClient:
    limits = httpx.Limits(
        max_connections=max_concurrency,
        max_keepalive_connections=max_concurrency,
    )
    timeout = httpx.Timeout(600)
    return httpx.AsyncClient(limits=limits, timeout=timeout)
if __name__ == "__main__":
    main(
        num_worker_processes=4,
        max_concurrency_per_process=4,
        model="ep-bi-20250731100643-zvnt9",
        api_key=api_key    # 建议直接在环境变量中设置，避免泄漏
    )