from attr.validators import max_len

from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
import re
import json
import os
import sys
# 统一类型注解，避免混用any/Any
from typing import List, Dict, Any, Tuple
# LangChain文本分割器（标注核心用途，便于理解）
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 项目内部工具/状态/日志导入（保持原有路径）
from app.utils.task_utils import add_running_task
from app.import_process.agent.state import ImportGraphState
from app.core.logger import logger  # 项目统一日志工具，核心替换print

DEFAULT_MAX_CONTENT_LENGTH = 2000
DEFAULT_MIN_CONTENT_LENGTH = 500


def node_document_split(state: ImportGraphState) -> ImportGraphState:
    """
        节点: 文档切分 (node_document_split)
        提取标题，根据标题分割文档
        处理没有标题的特殊情况
        根据标题进行分割
        如果过长需要根据段落进行划分，如果太短需要合并
        备份分割好的chunk,方便后续检查
        为什么叫这个名字: 将长文档切分成小的 Chunks (切片) 以便检索。
        未来要实现:
        1. 基于 Markdown 标题层级进行递归切分。
        2. 对过长的段落进行二次切分。
        3. 生成包含 Metadata (标题路径) 的 Chunk 列表。
        """

    node_name = sys._getframe().f_code.co_name
    logger.info(f">>> 开始执行核心节点：【文档切分】{node_name}")
    # 将当前节点加入运行中任务，更新全局任务状态
    add_running_task(state["task_id"], node_name)
    try:
        content, file_title, max_len = step_1_inputs(state)
        logger.info(f"步骤1：输入数据加载完成，文件标题：{file_title}，最大Chunk长度：{max_len}")

        if content is None:
            logger.info(f">>> 节点执行终止：{node_name}（无有效MD内容）")
            return state
        sections, title_count, lines_count = step_2_split_by_titles(content, file_title)

        sections = step_3_handle_no_title(content, sections, title_count, file_title)

        sections = step_4_refine_chunks(sections, max_len)

        step_5_print_stas(lines_count, sections)

        step_6_backup(state, sections)

        state["chunks"] = sections
    except Exception as e:
        logger.error(f">>> 节点执行异常：{node_name}，异常信息：{e}")

    return state


def step_6_backup(state: ImportGraphState, sections: List[Dict[str, Any]]) -> None:
    """
       【步骤6】Chunk结果本地JSON备份（便于调试/问题排查，保留处理结果）
       :param state: 项目状态字典，需包含local_dir（备份目录）
       :param sections: 最终处理后的Chunk列表
       """
    # 提取备份目录：无则直接返回，不执行备份
    local_dir = state.get("local_dir")
    if not local_dir:
        logger.warning("步骤6：未配置备份目录（local_dir），跳过Chunk结果备份")
        return

    try:
        # 创建备份目录：已存在则不报错（exist_ok=True）
        os.makedirs(local_dir, exist_ok=True)
        # 拼接备份文件路径：local_dir + chunks.json（固定文件名，便于查找）
        backup_path = os.path.join(local_dir, "chunks.json")
        # 写入JSON文件：保留中文/格式化缩进，便于人工查看
        with open(backup_path, "w", encoding="utf-8") as f:
            """
            sections是Python 嵌套数据结构（List[Dict[str, Any]]，列表里装字典，字典里可能嵌套字符串 / 数字等），而普通文件写入
            （如f.write(sections)）仅支持写入字符串，直接写 Python 数据结构会报错。
            json.dump的核心作用就是：将 Python 原生数据结构（列表、字典、字符串、数字等）直接序列化并写入 JSON 文件，无需手动转换为字符串，
            同时保证数据格式规范、可跨语言 / 跨场景读取，完美适配「Chunk 列表备份」的需求。
            """
            json.dump(
                sections,
                f,
                # 开启 True："title": "\u4e00\u7ea7\u6807\u9898"（乱码，无法直接看）；
                # 开启 False："title": "一级标题"（正常中文，人工可直接阅读）。
                ensure_ascii=False,  # 保留中文，不转义为\u编码
                indent=2  # 格式化缩进，便于阅读
            )
        logger.info(f"步骤6：Chunk结果备份成功，备份文件路径：{backup_path}")
    except Exception as e:
        # 备份失败仅记录日志，不终止主流程
        logger.error(f"步骤6：Chunk结果备份失败，错误信息：{str(e)}", exc_info=False)


def step_1_inputs(state: ImportGraphState) -> tuple[Any, str, int]:
    content = state.get("md_content")
    if not content:
        return None, None, None
    content = content.replace("\r\n", "\n").replace("\r", "\n")

    file_title = state["file_title"]
    max_length = DEFAULT_MAX_CONTENT_LENGTH
    logger.info(f"步骤1：输入数据加载完成，文件标题：{file_title}，最大Chunk长度：{max_len}")

    return content, file_title, max_length


def step_2_split_by_titles(content: str, file_title: str) -> Tuple[List[Dict[str, Any]], int, int]:
    """
       【步骤2】按Markdown标题初次切分（核心：按#分级切分，跳过代码块内标题）
       LangChain前置预处理：将整份MD按标题拆分为独立章节，为后续精细化切分做基础
       :param content: 标准化后的MD完整内容（字符串）
       :param file_title: 所属文件标题，用于标记章节归属
       :return: 切分后的章节列表/有效标题数量/原始文本总行数
       """
    # 正则匹配Markdown 1-6级标题（核心规则，适配缩进/标准格式）
    # ^\s*：行首允许0/多个空格/Tab（兼容缩进的标题）
    # #{1,6}：匹配1-6个#（对应MD1-6级标题）
    # \s+：#后必须有至少1个空格（区分#是标题还是普通文本）
    # .+：标题文字至少1个字符（避免空标题）
    title_pattern = r'^\s*#{1,6}\s+.+'

    # 将MD内容按换行符拆分为行列表，逐行处理
    lines = content.split("\n")
    sections = []  # 最终切分的章节列表
    current_title = ""  # 当前章节标题
    current_lines = []  # 当前章节的行缓存
    title_count = 0  # 有效标题数量（非代码块内）
    in_code_block = False  # 代码块标记：避免误判代码块内的#为标题

    def _flush_section():
        """内部辅助函数：将当前缓存的章节写入sections，空缓存则跳过"""
        if not current_lines:
            return
        sections.append({
            "title": current_title,
            # 每段时间使用 \n换行区分
            "content": "\n".join(current_lines),
            "file_title": file_title,
        })

    # 逐行遍历，识别标题并切分章节
    for line in lines:
        stripped_line = line.strip()
        # 识别代码块边界（```/~~~）：进入/退出代码块时翻转状态
        if stripped_line.startswith("```") or stripped_line.startswith("~~~"):
            in_code_block = not in_code_block
            current_lines.append(line)
            continue

        # 判断是否为有效标题：非代码块内 + 匹配标题正则
        is_valid_title = (not in_code_block) and re.match(title_pattern, line)
        if is_valid_title:
            # 遇到新标题：先将上一个章节写入结果，再初始化新章节
            _flush_section()
            current_title = line.strip()  # 清理标题前后空格
            current_lines = [current_title]  # 新章节从标题开始
            title_count += 1
            logger.debug(f"识别到MD标题：{current_title}")
        else:
            # 普通行：追加到当前章节的行缓存
            current_lines.append(line)

    # 处理最后一个章节：循环结束后，将最后一个缓存的章节写入结果
    _flush_section()
    logger.info(f"步骤2：MD标题切分完成，识别到{title_count}个有效标题，原始文本共{len(lines)}行")
    return sections, title_count, len(lines)


def step_3_handle_no_title(content: str, sections: List[Dict[str, Any]], title_count: int, file_title: str) -> List[
    Dict[str, Any]]:
    if title_count == 0:
        return [{"title": "无标题", "content": content, "file_title": file_title}]
    return sections


def _split_long_section(section, max_length):
    content = section.get("content", "") or ""
    if len(content) <= max_length:
        return [section]
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    title = section.get("title", "") or ""

    title_prefix = f"{title}\n\n" if title else ""

    available_len = max_length - len(title_prefix)

    if available_len <= 0:
        logger.warning(f"章节标题过长，无法切分：{title[:20]}...")
        return [section]

    body = content
    if title and body.lstrip().startswith(title):
        start_index = body.find(title) + len(title)
        body = body[start_index:].lstrip()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=available_len,
        chunk_overlap=0,
        separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " "],

    )

    # 切分正文并组装子章节（带完整元信息，便于溯源）
    sub_sections = []
    for idx, chunk in enumerate(splitter.split_text(body), start=1):
        # 清理空内容：跳过切分后的空字符串
        text = chunk.strip()
        if not text:
            continue
        # 组装子Chunk完整内容：标题前缀 + 切分后的正文
        full_text = (title_prefix + text).strip()
        # 子章节元信息：保留父级关联，添加序号，便于后续检索/溯源
        sub_sections.append({
            "title": f"{title}-{idx}" if title else f"chunk-{idx}",  # 子Chunk标题（带序号）
            "content": full_text,  # 切分后的完整内容
            "parent_title": title,  # 父章节标题（用于后续合并）
            "part": idx,  # 子Chunk序号
            "file_title": section.get("file_title"),  # 所属文件标题
        })

    logger.debug(f"超长章节切分完成：{title} → 生成{len(sub_sections)}个子Chunk")
    return sub_sections




def _merge_short_sections(sections : List[Dict[str, Any]], min_length: int = DEFAULT_MIN_CONTENT_LENGTH):
    if not sections:
        logger.debug("待合并Chunk列表为空，直接返回")
        return []
    merged_sections = []  # 最终合并结果
    current_chunk = None  # 迭代累加器：保存当前待合并的Chunk
    for sec in sections:
        if current_chunk is None:
            current_chunk = sec
            continue
        is_current_short = len(current_chunk.get("content","")) < min_length
        is_same_parent = current_chunk.get("parent_title") == sec.get("parent_title")
        if is_current_short and is_same_parent:
            parent_title = sec.get("parent_title", "")
            next_content = sec["content"]
            if parent_title and next_content.startswith(parent_title):
                next_content = next_content[len(parent_title):].lstrip()
            current_chunk["content"] += "\n\n" + next_content
            if "part" in sec:
                current_chunk["part"] = sec["part"]
            logger.debug(f"合并短Chunk：{current_chunk.get('parent_title')} → 累计长度{len(current_chunk.get('content',''))}")
        else:
            # 不满足合并条件：将当前块加入结果，切换为新的待合并块
            merged_sections.append(current_chunk)
            current_chunk = sec
        # 循环结束后，将最后一个待合并块加入结果
    if current_chunk is not None:
        merged_sections.append(current_chunk)
    logger.debug(f"短Chunk合并完成：原{len(sections)}个 → 合并后{len(merged_sections)}个")
    return merged_sections

def step_4_refine_chunks(sections: List[Dict[str, Any]], max_len: int) -> List[Dict[str, Any]]:
    """
    调用辅助函数，对过长或过短的 Chunk 进行二次处理。
    :param sections:
    :param max_len:
    :return:
    """
    if not max_len or max_len <= 0:
        logger.warning(f"步骤4：Chunk最大长度配置无效（{max_len}），跳过精细化处理")
        return sections
    refined_split = []
    for sec in sections:
        refined_split.extend(_split_long_section(sec, max_len))

    final_sections = _merge_short_sections(refined_split)
    
    for sec in final_sections:
        if not isinstance(sec, dict):
            continue
        if "part" not in sec:
            sec["part"] = 0

        if not sec.get("parent_title"):
            sec["parent_title"] = sec.get("title") or ""

    return final_sections



def step_5_print_stas(lines_count: int, sections: List[Dict[str, Any]]) -> None:
    """
       【步骤5】输出文档切分统计信息（日志记录，便于监控/调试）
       :param lines_count: MD原始文本总行数
       :param sections: 最终处理后的Chunk列表
       """
    chunk_num = len(sections)
    # 输出核心统计信息：原始行数/最终Chunk数/首个Chunk预览
    logger.info("-" * 50 + " 文档切分统计信息 " + "-" * 50)
    logger.info(f"MD原始文本总行数：{lines_count}")
    logger.info(f"最终生成Chunk数量：{chunk_num}")
    if sections:
        first_title = sections[0].get("title", "无标题")
        logger.info(f"首个Chunk标题预览：{first_title}")
    logger.info("-" * 110)


if __name__ == '__main__':
    """
    单元测试：联合node_md_img（图片处理节点）进行集成测试
    测试条件：1.已配置.env（MinIO/大模型环境） 2.存在测试MD文件 3.能导入node_md_img
    测试流程：先运行图片处理→再运行文档切分，验证端到端流程
    """

    """本地测试入口：单独运行该文件时，执行MD图片处理全流程测试"""
    from app.utils.path_util import PROJECT_ROOT
    from app.import_process.agent.nodes.node_md_img import node_md_img

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
            "md_content": "",
            "file_title": "hak180产品安全手册",
            "local_dir":os.path.join(PROJECT_ROOT, "output"),
        }
        logger.info("开始本地测试 - MD图片处理全流程")
        # 执行核心处理流程
        result_state = node_md_img(test_state)
        logger.info(f"本地测试完成 - 处理结果状态：{result_state}")
        logger.info("\n=== 开始执行文档切分节点集成测试 ===")

        logger.info(">> 开始运行当前节点：node_document_split（文档切分）")
        final_state = node_document_split(result_state)
        final_chunks = final_state.get("chunks", [])
        logger.info(f"✅ 测试成功：最终生成{len(final_chunks)}个有效Chunk{json.dumps(final_chunks,ensure_ascii=False)}")
