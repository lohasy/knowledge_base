import base64
import os
import re
import sys
from pathlib import Path
from typing import Tuple, List, Dict

from langchain_core.messages import HumanMessage
from minio import Minio
from minio.deleteobjects import DeleteObject
from requests.packages import target

from app.clients.minio_utils import get_minio_client
from app.conf.lm_config import lm_config
from app.conf.minio_config import minio_config
from app.core.load_prompt import load_prompt
from app.core.logger import logger

from app.import_process.agent.state import ImportGraphState
from app.lm.lm_utils import get_llm_client
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

    summaries = step_3_generate_summaries(path_obj.stem, targets)

    new_md_content = step_4_upload_and_replace(minio_client, path_obj.stem, targets, summaries, md_content)
    state["md_content"] = new_md_content

    # 步骤5：备份并保存新MD文件，更新状态中的文件路径
    new_md_file_name = step_5_backup_new_md_file(state['md_path'], new_md_content)
    state["md_path"] = new_md_file_name
    logger.info(f"MD图片处理完成，新文件已保存：{new_md_file_name}")
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


def encode_image_to_base64(image_path: str) -> str:
    """
    将本地图片文件编码为Base64字符串（用于多模态大模型输入）
    :param image_path: 图片本地完整路径
    :return: 图片的Base64编码字符串（UTF-8解码）
    """
    with open(image_path, "rb") as img_file:
        base64_str = base64.b64encode(img_file.read()).decode("utf-8")
    logger.debug(f"图片Base64编码完成，文件：{image_path}，编码后长度：{len(base64_str)}")
    return base64_str


def summarize_image(image_path, root_folder, image_content):
    base64_image = encode_image_to_base64(image_path)
    llm_client = get_llm_client(model=lm_config.lv_model)
    prompt = load_prompt(name="image_summary", root_folder=root_folder, image_content=image_content)

    messages = [
        HumanMessage(
            content=[
                # 文本提示词：携带上下文，限定摘要规则
                {
                    "type": "text",
                    "text": prompt
                },
                # 多模态核心：Base64编码图片数据
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        )

    ]
    # 3. LangChain标准调用：invoke方法（工具类已封装超时/重试等参数）
    response = llm_client.invoke(messages)

    # 4. 解析响应（LangChain统一返回content字段，统一格式无需多层解析）
    summary = response.content.strip().replace("\n", "")
    logger.info(f"图片摘要生成成功：{image_path}，摘要：{summary}")
    return summary

    pass


def step_3_generate_summaries(doc_stem: str, targets: List[Tuple[str, str, Tuple[str, str]]],
                              requests_per_minute: int = 9) -> Dict[str, str]:
    summaries = {}

    for img_file, image_path, context in targets:
        summaries[img_file] = summarize_image(image_path, root_folder=doc_stem, image_content=context)

    return summaries


def clean_minio_directory(minio_client, minio_upload_dir):
    try:
        delete_file_list = minio_client.list_objects(bucket_name=minio_config.bucket_name, prefix=minio_upload_dir, recursive=True)
        delete_file_name_list = [DeleteObject(delete_file.object_name) for delete_file in delete_file_list]
        if delete_file_name_list:
            errors = minio_client.remove_objects(bucket_name=minio_config.bucket_name, objects=delete_file_name_list)
            if errors:
                logger.error(f"MinIO删除对象失败：{errors}")
    except Exception as e:
        logger.error(f"MinIO删除对象失败：{e}")


def upload_to_minio(minio_client, img_path, object_name):
    try:
        minio_client.fput_object(
            bucket_name=minio_config.bucket_name,
            object_name=object_name,
            file_path=img_path,
            content_type=f"{os.path.splitext(img_path)[1][1:]}",
        )

        object_name = object_name.replace("\\", "%5C")
        protocol = "https" if minio_config.minio_secure else "http"

        base_url = f"{protocol}://{minio_config.endpoint}/{minio_config.bucket_name}"
        # 拼接完整图片访问URL base_url 后面带 / 中间直接两个字符串拼接即可
        img_url = f"{base_url}{object_name}"
        logger.info(f"图片上传成功，访问URL：{img_url}")
    except Exception as e:
        logger.error(f"图片上传失败：{e}")
        return None
    return img_url


def upload_images_batch(minio_client: Minio,
                        upload_dir: str,
                        targets: List[Tuple[str, str, Tuple[str, str]]]
                        ) -> Dict[str, str]:
    urls = {}
    for img_file,img_path,_ in targets:
        object_name = f"{upload_dir}/{img_file}"

        if img_url := upload_to_minio(minio_client, img_path, object_name):
            urls[img_file] = img_url
    return urls


def merge_summary_and_url(summaries: Dict[str, str], urls: Dict[str, str]) -> Dict[str, Tuple[str, str]]:
    """
    合并图片摘要字典和URL字典，过滤掉上传失败无URL的图片
    :param summaries: 图片摘要字典，键：图片文件名，值：内容摘要
    :param urls: 图片URL字典，键：图片文件名，值：MinIO访问URL
    :return: 合并后的图片信息字典，键：图片文件名，值：(摘要, URL)元组
    """
    image_info = {}
    # 遍历摘要字典，仅保留有对应URL的图片
    for image_file, summary in summaries.items():
        if url := urls.get(image_file):
            image_info[image_file] = (summary, url)
    logger.info(f"图片摘要与URL合并完成，有效图片信息{len(image_info)}条")
    return image_info


def process_md_file(md_content:str, image_info:Dict[str, Tuple[str, str]]) -> str:
    for img_filename,(summary,new_url) in image_info.items():
        pattern = re.compile(
            r"!\[.*?\]\(.*?" + re.escape(img_filename) + r".*?\)",
            re.IGNORECASE
        )
        md_content = pattern.sub(f"![{summary}]({new_url})", md_content)
        logger.debug(f"完成MD图片引用替换：{img_filename} → {new_url}")
    logger.info(f"MD文件图片引用替换完成，共替换{len(image_info)}处图片引用")
    logger.debug(f"替换后MD内容：{md_content[:500]}..." if len(md_content) > 500 else f"替换后MD内容：{md_content}")
    return md_content



def step_4_upload_and_replace(minio_client: Minio, doc_stem: str,
                              targets: List[Tuple[str, str, Tuple[str, str]]], summaries: Dict[str, str],
                              md_content: str
                              ) -> str:
    minio_upload_dir = f"{minio_config.minio_img_dir}/{doc_stem}".replace(" ", "")

    clean_minio_directory(minio_client, minio_upload_dir)

    urls = upload_images_batch(minio_client, minio_upload_dir, targets)

    image_info = merge_summary_and_url(summaries, urls)

    if image_info:
        md_content = process_md_file(md_content, image_info)

    return md_content

def step_5_backup_new_md_file(origin_md_path: str, md_content: str):
    new_md_file_name = os.path.splitext(origin_md_path)[0] + "_new.md"
    new_md_file_path = os.path.join(os.path.dirname(origin_md_path), new_md_file_name)
    with open(new_md_file_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info(f"新MD文件保存成功：{new_md_file_path}")
    return new_md_file_name


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
