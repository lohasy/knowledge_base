import sys
from pathlib import Path
from typing import Tuple, List

from app.clients.minio_utils import get_minio_client
from app.core.logger import logger

from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task


def node_md_img(state: ImportGraphState) -> ImportGraphState:
    """
        节点: 图片处理 (node_md_img)
        为什么叫这个名字: 处理 Markdown 中的图片资源 (Image)。
        未来要实现:
        1. 扫描 Markdown 中的图片链接。
        2. 将图片上传到 MinIO 对象存储。
        3. (可选) 调用多模态模型生成图片描述。
        4. 替换 Markdown 中的图片链接为 MinIO URL。
        """
    method_name = sys._getframe().f_code.co_name
    logger.info(f">>> [{method_name}] 执行节点: {method_name}")
    add_running_task(state["task_id"], method_name)

    md_content, path_obj, images_dir = step_1_get_content(state)
    state["md_content"] = md_content

    minio_client = get_minio_client()
    if not minio_client:
        logger.warning(f">> [{method_name}] minio_client 为空")
        return state

    targets = step_2_scan_images(md_content, images_dir)
    if not targets:
        logger.info("未检测到MD中引用的支持格式图片，跳过后续处理")
        return state
    return state


def step_1_get_content(state: ImportGraphState) -> Tuple[str, Path, Path]:
    pass


def step_2_scan_images(md_content, images_dir) -> List[Tuple[str, str, Tuple[str, str]]]:
    pass
