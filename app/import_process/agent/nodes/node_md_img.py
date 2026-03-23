import os
import sys
from pathlib import Path
from typing import Tuple, List

from app.clients.minio_utils import get_minio_client
from app.core.logger import logger

from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

node_method_name = None


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

    global node_method_name
    node_method_name = method_name

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


def is_supported_image(filename: str) -> bool:
    """
    判断文件是否为MinIO支持的图片格式（后缀不区分大小写）
    :param filename: 文件名（含后缀）
    :return: 支持返回True，否则False
    """
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS


def step_1_get_content(state: ImportGraphState) -> Tuple[str, Path, Path]:
    md_path = state["md_path"]
    if not md_path:
        raise FileNotFoundError("md_path 为空")
    md_path_path = Path(md_path)
    if not state["md_content"]:
        state["md_content"] = md_path_path.read_text(encoding="utf-8")
        md_content = state["md_content"]
    else:
        md_content = state["md_content"]

    logger.debug(f">> [{node_method_name}] md_content: {md_content[:100]}")

    images_dir = md_path_path.parent / "images"
    return md_content, md_path_path, images_dir


def find_image_in_md(md_content: str, image_file: str, context_len: int = 100):
    import re
    pattern = re.compile(r"!\[.*?]\(.*?" + re.escape(image_file) + r".*?\)")
    results = []

    # 迭代查找所有MD图片标签匹配项
    for m in pattern.finditer(md_content):
        start, end = m.span()
        # 截取匹配位置的上文和下文（防止索引越界）
        pre_text = md_content[max(0, start - context_len):start]
        post_text = md_content[end:min(len(md_content), end + context_len)]
        # 打印图片上下文，便于调试
        logger.debug(f"图片[{image_file}]匹配到引用，上文：{pre_text.strip()}")
        logger.debug(f"图片[{image_file}]匹配到引用，下文：{post_text.strip()}")
        results.append((pre_text, post_text))

    if not results:
        logger.debug(f"MD内容中未找到图片[{image_file}]的引用")
    return results


def step_2_scan_images(md_content, images_dir) -> List[Tuple[str, str, Tuple[str, str]]]:
    image_files = os.listdir(images_dir)

    targets = []
    for image_file in image_files:
        if not is_supported_image(image_file):
            continue
        image_path = str(images_dir / image_file)
        context_list = find_image_in_md(md_content, image_file)
        if not context_list:
            continue
        targets.append((image_file, image_path, context_list[0]))
    return targets

if __name__ == "__main__":
    """本地测试入口：单独运行该文件时，执行MD图片处理全流程测试"""
    from app.utils.path_util import PROJECT_ROOT
    logger.info(f"本地测试 - 项目根目录：{PROJECT_ROOT}")

    # 测试MD文件路径（需手动将测试文件放入对应目录）
    test_md_name = os.path.join(r"output\hak180产品安全手册", "hak180产品安全手册.md")
    test_md_path = os.path.join(PROJECT_ROOT, test_md_name)

    # 校验测试文件是否存在
    if not os.path.exists(test_md_path):
        logger.error(f"本地测试 - 测试文件不存在：{test_md_path}")
        logger.info("请检查文件路径，或手动将测试MD文件放入项目根目录的output目录下")
    else:
        # 构造测试状态对象，模拟流程入参
        test_state = {
            "md_path": test_md_path,
            "task_id": "test_task_123456",
            "md_content": ""
        }
        logger.info("开始本地测试 - MD图片处理全流程")
        # 执行核心处理流程
        result_state = node_md_img(test_state)
        logger.info(f"本地测试完成 - 处理结果状态：{result_state}")
