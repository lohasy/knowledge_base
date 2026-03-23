import os.path
import sys

from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState, graph_default_state, create_default_state
from app.utils.format_utils import format_state
from app.utils.task_utils import add_running_task, add_done_task


def node_entry(state: ImportGraphState) -> ImportGraphState:
    """
     入口节点，判断文件是md还是pdf
     补充state中相关字段如下：

     is_pdf_read_enabled
     pdf_path
     或
     is_md_read_enabled
     md_path

     file_title

    :param state: state 传入参数需要有 路径 和 task_id
    :return: state 更新
    """
    method_name: str = sys._getframe().f_code.co_name
    logger.debug(f">>> [{method_name}] 开始执行节点: {method_name}")
    logger.debug(f">>> [{method_name}] state: {format_state(dict(state))}")

    add_running_task(state["task_id"], method_name)

    local_file_path: str = state.get("local_file_path","")
    if not local_file_path:
        logger.error(f">>> [{method_name}] local_file_path为空: {local_file_path}")
        return state

    if local_file_path.endswith(".pdf"):
        state["is_pdf_read_enabled"] = True
        state["pdf_path"] = local_file_path
    elif local_file_path.endswith(".md"):
        state["is_md_read_enabled"] = True
        state["md_path"] = local_file_path
    else:
        logger.warning(f"【{method_name}】文件类型校验失败：{local_file_path} → 不支持的格式，仅支持.pdf/.md")

    file_name: str= os.path.basename(local_file_path)
    state["file_title"] = os.path.splitext(file_name)[0]

    logger.info(f"【{method_name}】文件业务标识提取完成：file_title = {state['file_title']}")

    # 结束：记录节点运行状态
    add_done_task(state["task_id"], method_name)

    # 节点完成日志，打印当前工作流状态
    logger.debug(f"【{method_name}】节点执行完成")
    logger.debug(f"【{method_name}】state：{format_state(dict(state))}")


    return state
if __name__ == '__main__':

    # 单元测试：覆盖不支持类型、MD、PDF三种场景
    logger.info("===== 开始node_entry节点单元测试 =====")

    # 测试1: 不支持的TXT文件
    test_state1 = create_default_state(
        task_id="test_task_001",
        local_file_path="C://test//frank//联想海豚用户手册.txt"
    )
    node_entry(test_state1)

    # 测试2: MD文件
    test_state2 = create_default_state(
        task_id="test_task_002",
        local_file_path="C://test//frank//小米用户手册.md"
    )
    node_entry(test_state2)

    # 测试3: PDF文件
    test_state3 = create_default_state(
        task_id="test_task_003",
        local_file_path="C://test//frank//万用表的使用.pdf"
    )
    node_entry(test_state3)

    logger.info("===== 结束node_entry节点单元测试 =====")