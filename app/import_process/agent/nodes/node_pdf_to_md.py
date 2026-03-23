import sys
from pathlib import Path

from app.core.logger import logger

from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task


def node_pdf_to_md(state: ImportGraphState) -> ImportGraphState:
    """
       节点: PDF转Markdown (node_pdf_to_md)
       为什么叫这个名字: 核心任务是将 PDF 非结构化数据转换为 Markdown 结构化数据。
       未来要实现:
       1. 调用 MinerU (magic-pdf) 工具。
       2. 将 PDF 转换成 Markdown 格式。
       3. 将结果保存到 state["md_content"]。
       """
    method_name = sys._getframe().f_code.co_name
    logger.debug(f">>> [method_name] 执行节点: {method_name}")
    logger.debug(f">>> [method_name] state: {state}")

    add_running_task(state["task_id"], method_name)
    return state


def step_1_validate_path(state: ImportGraphState) -> tuple[Path, Path]:
    pdf_path = state.get("pdf_path", "").strip()
    local_dir = state.get("local_dir", "").strip()
    if not pdf_path:
        logger.error(f">>> [method_name] pdf_path为空: {pdf_path}")
        raise ValueError("pdf_path为空")
    if not local_dir:
        logger.error(f">>> [method_name] local_dir为空: {local_dir}")
        raise ValueError("local_dir为空")

    pdf_path_path: Path = Path(pdf_path)
    local_dir_path: Path = Path(local_dir)

    if not pdf_path_path.exists():
        raise FileNotFoundError("pdf文件不存在")
    if not pdf_path_path.is_file():
        raise FileNotFoundError("pdf文件 非文件是目录")

    if not local_dir_path.is_dir():
        local_dir_path.mkdir(parents=True, exist_ok=True)
    return pdf_path_path, local_dir_path

def step_2_upload_and_pool():
    pass
