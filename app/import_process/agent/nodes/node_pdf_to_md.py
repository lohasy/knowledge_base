import os
import shutil
import sys
import zipfile
from pathlib import Path

from app.conf.mineru_config import mineru_config
from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState, create_default_state
from app.utils.task_utils import add_running_task

# MinerU配置（缓存配置信息）
MINERU_BASE_URL = mineru_config.base_url
MINERU_API_TOKEN = mineru_config.api_key
node_method_name = None

def node_pdf_to_md(state: ImportGraphState) -> ImportGraphState:
    """
       节点: PDF转Markdown (node_pdf_to_md)
        step_1_validate_path 校验路径，并转换为path对象

       """
    method_name = sys._getframe().f_code.co_name
    global node_method_name
    node_method_name = method_name
    add_running_task(state["task_id"], method_name)

    logger.debug(f">>> [{method_name}] 执行节点: {method_name}")
    logger.debug(f">>> [{method_name}] state: {state}")

    pdf_path_path, local_dir_path = step_1_validate_path(state)
    logger.debug(f">>> [{method_name}] pdf_path_path: {pdf_path_path}")
    logger.debug(f">>> [{method_name}] local_dir_path: {local_dir_path}")

    zip_url = step_2_upload_and_pool(pdf_path_path, local_dir_path)

    md_path = step_3_download(zip_url=zip_url, out_put_dir=local_dir_path, pdf_stem=pdf_path_path.stem)

    logger.info(f"【{method_name}】MD文件生成成功，路径：{md_path}")

    # 读取MD文件内容，捕获异常仅警告不终止
    try:
        with open(md_path, "r", encoding="utf-8") as f:
            state["md_content"] = f.read()
        logger.debug(f"【{method_name}】MD文件内容读取成功，内容长度：{len(state['md_content'])}字符")
    except Exception as e:
        logger.error(f"【{method_name}】读取MD文件内容失败：{str(e)}")

    logger.info(f"【{method_name}】节点执行完成，更新后工作流状态键：{list(state.keys())}")
    state["md_path"] = md_path
    return state


def step_3_download(zip_url: str, out_put_dir: Path, pdf_stem: str):
    zip_res = requests.get(zip_url, timeout=30)
    if zip_res.status_code != 200:
        raise RuntimeError(f"[{node_method_name}]下载失败: {zip_res.status_code}")

    zip_save_path = out_put_dir / f"{pdf_stem}_result.zip"
    with open(zip_save_path, "wb") as f:
        f.write(zip_res.content)

    unzip_dir = out_put_dir / pdf_stem

    # 清理旧目录，异常则警告不终止
    if unzip_dir.exists():
        try:
            # 递归删除整个目录树，包括目录本身及其所有子目录和文件。
            shutil.rmtree(unzip_dir)
            logger.info(f"[{node_method_name}] 已清理旧的解压目录：{unzip_dir}")
        except Exception as e:
            logger.warning(f"[{node_method_name}] 清理旧目录失败，可能不影响新文件解压：{str(e)}")


    # 重新创建解压目录
    unzip_dir.mkdir(parents=True, exist_ok=True)

    # 核心解压操作，保留原目录结构
    with zipfile.ZipFile(zip_save_path, 'r') as zip_file_obj:
        zip_file_obj.extractall(unzip_dir)
    logger.info(f"[{node_method_name}] ZIP包解压完成，解压目录：{unzip_dir}")

    md_generator = unzip_dir.rglob("*.md")
    md_files = list(md_generator)
    if not md_files:
        raise FileNotFoundError("ZIP包中未找到Markdown文件")

        # 4. 按优先级匹配目标MD文件（同名→full.md→第一个，兜底避免流程中断）
    target_md_file = None
    # 优先级1：与PDF纯名称完全同名的MD文件
    for md_file in md_files:
        if md_file.stem == pdf_stem:
            target_md_file = md_file
            logger.info(f"[{node_method_name}] 匹配到优先级1目标：与PDF同名的MD文件 {target_md_file.name}")
            break
    # 优先级2：MinerU默认生成的full.md（不区分大小写）
    if not target_md_file:
        for md_file in md_files:
            if md_file.name.lower() == "full.md":
                target_md_file = md_file
                logger.info(f"[{node_method_name}] 匹配到优先级2目标：MinerU默认文件 {target_md_file.name}")
                break
    # 优先级3：兜底取第一个MD文件
    if not target_md_file:
        target_md_file = md_files[0]
        logger.info(f"[{node_method_name}] 未匹配到前两级目标，兜底取第一个MD文件 {target_md_file.name}")

    # 重命名MD文件：统一为PDF纯名称，便于后续流程处理（仅不同名时执行）
    if target_md_file.stem != pdf_stem:
        logger.info(f"[{node_method_name}] 开始重命名MD文件，统一为PDF同名：{pdf_stem}.md")
        new_md_path = target_md_file.with_name(f"{pdf_stem}.md")
        try:
            # 将磁盘上的文件进行重命名
            target_md_file.rename(new_md_path)
            # 更新变量引用
            target_md_file = new_md_path
            logger.info(f"[{node_method_name}] MD文件重命名成功：{pdf_stem}.md")
        except OSError as e:
            logger.warning(f"[{node_method_name}] MD文件重命名失败，将使用原文件名继续流程：{str(e)}")

    # 转换为字符串绝对路径返回，适配后续仅支持字符串路径的函数
    final_md_path = str(target_md_file.absolute())
    logger.info(f"[{node_method_name}] 解析结果处理完成，最终MD文件路径：{final_md_path} =====")
    return final_md_path


def step_1_validate_path(state: ImportGraphState) -> tuple[Path, Path]:
    pdf_path = state.get("pdf_path", "").strip()
    local_dir = state.get("local_dir", "").strip()
    if not pdf_path:
        logger.error(f">>> [{node_method_name}] pdf_path为空: {pdf_path}")
        raise ValueError("pdf_path为空")
    if not local_dir:
        logger.error(f">>> [{node_method_name}] local_dir为空: {local_dir}")
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


def step_2_upload_and_pool(pdf_path_path: Path, local_dir_path: Path):
    if not MINERU_BASE_URL or not MINERU_API_TOKEN:
        raise ValueError("MINERU_BASE_URL or MINERU_API_TOKEN is not set")

    logger.debug("=========================mineru 获取上传链接============================================")
    request_headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MINERU_API_TOKEN}"
    }
    url_get_upload = f"{MINERU_BASE_URL}/file-urls/batch"
    req_data = {
        "files": [{"name": pdf_path_path.name}],
        "model_version": "vlm"  # 官方推荐解析模型
    }
    logger.debug(f"[获取上传链接] 调用接口：{url_get_upload}，请求参数：{req_data}")
    resp = requests.post(url=url_get_upload, headers=request_headers, json=req_data, timeout=30)
    logger.debug(f"[获取上传链接] 响应结果：{resp.status_code}，{resp.json()}")

    if resp.status_code != 200 or resp.json()["code"] != 0:
        raise RuntimeError(f"获取上传链接失败: {resp.status_code}，{resp.json()}")

    url = resp.json()["data"]["file_urls"][0]
    batch_id = resp.json()["data"]["batch_id"]

    logger.debug("=========================mineru 上传============================================")

    with(open(pdf_path_path, "rb")) as f:
        file_content = f.read()
    session = requests.Session()
    session.trust_env = False

    try:
        upload_resp = session.put(url, data=file_content, timeout=60)
        if upload_resp.status_code != 200:
            upload_resp = session.put(url, data=file_content, timeout=60)
            if upload_resp.status_code != 200:
                raise RuntimeError("上传文件失败")
    except:
        raise RuntimeError("上传文件失败")
    finally:
        session.close()

    logger.info(f"[上传文件] 响应结果：{upload_resp}")

    logger.debug("=========================mineru 下载============================================")

    poll_url = f"{MINERU_BASE_URL}/extract-results/batch/{batch_id}"
    zip_url = download_file_with_polling(poll_url, batch_id)
    logger.info(f"[下载文件] 响应结果：{zip_url}")

    return zip_url


def is_server_error(exception):
    """判断是否为服务端错误（500-599）"""
    if isinstance(exception, requests.exceptions.HTTPError):
        # 500-599 状态码才重试
        return 500 <= exception.response.status_code <= 599
    # 其他异常（如网络超时、连接错误）也重试
    return True


import polling
import requests


class BusinessError(Exception):
    """业务错误异常"""
    pass


def download_file_with_polling(url: str, batch_id: str):
    request_headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MINERU_API_TOKEN}"
    }

    def check_status():
        resp = requests.get(url=url, headers=request_headers, timeout=10)
        resp.raise_for_status()
        resp_json = resp.json()

        # 业务错误，抛出明确的 BusinessError
        if resp_json["code"] != 0:
            raise BusinessError(f"API业务错误: {resp_json}")

        extract_results = resp_json["data"]["extract_result"]
        result_item = extract_results[0]
        state_status = result_item["state"]

        if state_status == "done":
            full_zip_url = result_item.get("full_zip_url")
            if not full_zip_url:
                raise BusinessError("任务完成但未返回ZIP包下载链接")
            return full_zip_url

        elif state_status == "failed":
            err_msg = result_item.get("err_msg", "未知错误")
            raise BusinessError(f"解析任务失败：{err_msg}")

        else:  # pending
            return False  # 继续轮询

    try:
        result = polling.poll(
            target=check_status,
            step=2,
            timeout=600,
            poll_forever=False
        )
        return result
    except polling.TimeoutException:
        # 轮询超时
        raise TimeoutError(f"任务在600秒内未完成，batch_id：{batch_id}")
    except BusinessError as e:
        # 业务错误，直接抛出
        raise


if __name__ == "__main__":
    # 单元测试：验证PDF转MD全流程
    logger.info("===== 开始node_pdf_to_md节点单元测试 =====")

    from app.utils.path_util import PROJECT_ROOT

    logger.info(f"测试获取根地址：{PROJECT_ROOT}")

    test_pdf_name = os.path.join("doc", "hak180产品安全手册.pdf")
    test_pdf_path = os.path.join(PROJECT_ROOT, test_pdf_name)

    # 构造测试状态
    test_state = create_default_state(
        task_id="test_pdf2md_task_001",
        pdf_path=test_pdf_path,
        local_dir=os.path.join(PROJECT_ROOT, "output")
    )

    node_pdf_to_md(test_state)

    logger.info("===== 结束node_pdf_to_md节点单元测试 =====")
