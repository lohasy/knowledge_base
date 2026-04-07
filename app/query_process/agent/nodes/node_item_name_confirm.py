import json
import os
import sys
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage

from app.clients.milvus_utils import get_milvus_client, create_hybrid_search_requests, hybrid_search
from app.clients.mongo_history_utils import save_chat_message, get_recent_messages, update_message_item_names
from app.core.load_prompt import load_prompt
from app.lm.embedding_utils import generate_embeddings
from app.lm.lm_utils import get_llm_client
from app.utils.task_utils import add_running_task, add_done_task
from app.core.logger import logger


def step_3_extract_info(original_query, history):
    logger.info("Step 3: 正在初始化 LLM 客户端...")
    client = get_llm_client(json_mode=True)
    history_text = ""
    for msg in history:
        history_text += f"{msg['role']} : {msg['message']} \n"
    logger.info(f"Step 3: 历史上下文准备完成 (长度: {len(history_text)})")

    prompt = load_prompt("rewritten_query_and_itemnames", history_text=history_text, query=query)
    logger.info(f"Step 3: 提示词加载成功")

    messages = [
        SystemMessage(content="你是一个专业的客服助手，擅长理解用户意图和提取关键信息。"),
        HumanMessage(content=prompt)
    ]
    try:
        logger.info("Step 3: 正在调用 LLM...")
        response = client.invoke(messages)
        logger.info("Step 3: 收到 LLM 响应")
        content = response.content
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "")
        result = json.loads(content)
        logger.info(f"Step 3: 解析 LLM 结果: {result}")


        if "item_names" not in result:
            result["item_names"] = []

        if "rewritten_query" not in result:
            result["rewritten_query"] = original_query
        return result
    except Exception as e:
        # 捕获所有异常（如LLM调用失败、JSON解析失败等），记录错误日志
        logger.error(f"Step 3 LLM 提取失败: {e}")
        # 异常时返回默认结果：空商品名列表+原始查询
        return {"item_names": [], "rewritten_query": original_query}


def step_4_vectorize_and_query(item_names):
    logger.info(f"Step 4: Starting vectorization and query for items: {item_names}")
    results = []
    client = get_milvus_client()

    if not client:
        logger.error("Failed to connect to Milvus")
        return results

    collection_name = os.environ.get("ITEM_NAME_COLLECTION")
    if not collection_name:
        logger.error("No collection name found in env")
        return results
    logger.info("Step 4: 正在生成向量...")
    embeddings = generate_embeddings(results)
    logger.info(f"Step 4: 已生成 {len(item_names)} 个商品名的向量。开始 Milvus 搜索...")

    for i in range(len(item_names)):

        try:
            logger.info(f"Step 4: 正在处理商品 {i + 1}/{len(item_names)}: {item_names[i]}")
            dense_vector = embeddings.get("dense")[i]
            sparse_vector = embeddings.get("sparse")[i]
            reqs = create_hybrid_search_requests(
                dense_vector=dense_vector,
                sparse_vector=sparse_vector,
                limit=5
            )
            logger.info(f"Step 4: 正在 Milvus 集合 '{collection_name}' 中执行混合搜索: '{item_names[i]}'")

            search_res = hybrid_search(
                client=client,  # Milvus客户端连接实例
                collection_name=collection_name,  # 目标向量集合名（存储商品向量的表）
                reqs=reqs,  # 混合搜索请求对象列表
                ranker_weights=(0.8, 0.2),  # 稠/稀疏向量评分权重配比（和为1最佳）
                limit=5,  # 最终返回Top5匹配结果
                norm_score=True,  # 开启评分归一化，统一评分量级为0-1
                output_fields=["item_name"]  # 指定返回Milvus中存储的商品名字段（业务字段）
            )
            logger.info(f"Step 4: '{item_names[i]}' 搜索完成。找到 {len(search_res[0]) if search_res else 0} 个匹配项。")

            matches = []
            if search_res and len(search_res) > 0:
                # 遍历当前商品名的Top5匹配结果（search_res[0]为该商品的独立搜索结果集）
                for hit in search_res[0]:
                    # 提取匹配结果中的商品名和评分，做防KeyError处理（设置默认空字典）
                    # hit格式：{"id": 数据库ID, "distance": 相似度评分, "entity": {"item_name": "标准化商品名"}}
                    matches.append(
                        {
                            "item_name": hit.get("entity", {}).get("item_name"),  # 数据库标准化商品名
                            "score": hit.get("distance"),  # 0-1相似度评分
                        }
                    )

                # 将当前商品名的原始名称+匹配结果，封装后加入最终结果列表
            results.append({
                "extracted_name": item_names[i],  # step3提取的原始商品名称
                "matches": matches  # 该商品名的Top5匹配结果（含评分）
            })
        except Exception as e:
            logger.error(f"Step 4: 查询商品名 '{item_names[i]}' 时出错: {e}")
    return results


def step_5_align_item_names(query_results):
    confirmed_item_names: List[str] = []
    options: List[str] = []

    logger.info(f"获得待处理的数据源：{query_results}")
    for res in query_results:
        extracted_name = (res.get("extracted_name", "")).strip()
        matches = res.get("matches", []) or []
        if not matches:
            continue
        matches.sort(key=lambda x: x.get("score",0), reverse=True)
        high = [m for m in matches if m.get("score", 0) > 0.85]
        mid = [m for m in matches if m.get("score", 0) >= 0.6]
        if len(high) == 1:
            confirmed_item_names.append(high[0].get("item_name"))
            continue
        if len(high) > 1:
            picked = None
            if extracted_name:
                for m in high:
                    if m.get("item_name") == extracted_name:
                        picked = m
                        break
            if not picked:
                picked = high[0]
            confirmed_item_names.append(picked.get("item_name"))
            continue  # 匹配到规则b，跳过后续规则判断

        if len(mid) > 0:
            # 取中置信度结果的前5个，加入候选列表
            for m in mid[:5]:
                options.append(m.get("item_name"))

        return {
            "confirmed_item_names": list(set(confirmed_item_names)),  # 去重，避免重复确认
            "options": list(set(options))  # 去重，避免重复候选

        }


def step_6_check_confirmation(state, align_result, session_id, history, rewritten_query):
    confirmed = align_result.get("confirmed_item_names", [])
    options = align_result.get("options", [])

    if confirmed:
        # 收集历史消息中未关联商品名的消息ID（需批量更新关联）
        ids_to_update = []
        for msg in history:
            if not msg.get("item_names"):  # 仅更新item_names为空的历史消息
                mid = msg.get("_id")  # 提取消息唯一ID
                if mid:
                    ids_to_update.append(str(mid))  # 转为字符串，避免ID格式问题

        # 若存在需更新的消息ID，批量更新历史消息的商品名关联
        if ids_to_update:
            update_message_item_names(ids_to_update, confirmed)
        # 更新会话状态：设置确认商品名、改写后的查询
        state["item_names"] = confirmed
        state["rewritten_query"] = rewritten_query
        # 若状态中存在旧答案，删除（避免干扰后续流程）
        if "answer" in state:
            del state["answer"]
        # 返回更新后的状态
        return state
        # 分支B：无确认商品名，但有候选商品名（中置信度，需用户明确）

    if options:
        # 候选商品名拼接为字符串（取前3个，避免过长），格式："商品1、商品2、商品3"
        options_str = "、".join(options[:3])
        # 构造向用户确认的提示语
        answer = f"您是想问以下哪个产品：{options_str}？请明确一下型号。"
        # 更新会话状态：设置确认提示语、清空商品名列表
        state["answer"] = answer
        state["item_names"] = []
        return state

        # 分支C：无确认商品名，且无候选商品名（无匹配结果，需用户重新提供）
    state["answer"] = "抱歉，未找到相关产品，请提供准确型号以便我为您查询。"
    state["item_names"] = []
    return state

def step_7_write_history(state, session_id, history, rewritten_query, message_id):
    """
     7 把本次处理的核心信息（用户问题、助手答案、商品名、改写查询）写入MongoDB的会话历史
     包含2个核心操作：1. 写入助手答案（若有）；2. 更新用户原始问题的关联信息
     :param state: 字典 - step6更新后的会话状态，包含answer/item_names等字段
     :param session_id: 字符串 - 会话唯一标识
     :param history: 列表[字典] - 近期会话历史（无实际业务逻辑，预留扩展）
     :param rewritten_query: 字符串 - step3改写后的完整问题
     :param message_id: 字符串 - 本次用户问题的消息唯一ID（step2生成）
     :return: 字典 - 最终的会话状态（无额外修改，直接返回入参state）
     """
    # 若会话状态中有助手答案（分支B/C），写入助手消息到历史
    if state.get("answer"):
        save_chat_message(
            session_id=session_id,  # 会话ID，关联所属会话
            role="assistant",  # 消息角色：助手
            text=state["answer"],  # 消息内容：向用户确认的提示语/无结果提示语
            rewritten_query="",  # 助手消息无需改写查询，设为空
            item_names=state.get("item_names", [])  # 关联的商品名列表（分支B/C均为空）
        )

    # 强制更新本次用户原始问题的关联信息（核心：补充改写查询、商品名）
    save_chat_message(
        session_id=session_id,  # 会话ID，关联所属会话
        role="user",  # 消息角色：用户
        text=state["original_query"],  # 消息内容：用户原始查询
        rewritten_query=rewritten_query,  # 补充step3改写后的完整问题
        item_names=state.get("item_names", []),  # 补充关联的商品名列表
        message_id=message_id  # 消息ID，指定更新已存在的用户消息（而非新增）
    )

    # 返回最终会话状态，供下游节点使用
    return state



def node_item_name_confirm(state):
    """
    节点功能：确认用户问题中的核心商品名称。
    输入：state['original_query']
    输出：更新 state['item_names']
    """
    print(f"---node_item_name_confirm---开始处理")

    session_id = state["session_id"]
    original_query = state.get("original_query", "")
    is_stream = state.get("is_stream", False)
    # 记录任务开始
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state["is_stream"])
    history = get_recent_messages(session_id, limit=10)
    logger.info(f"Node: 获取到 {len(history)} 条历史消息")

    message_id = save_chat_message(state['session_id'], "user", original_query, "", state.get("item_names", []))
    logger.debug(f"Node: 用户消息已初始保存, ID: {message_id}")

    extract_res = step_3_extract_info(original_query, history)
    item_names = extract_res.get("item_names", [])
    rewritten_query = extract_res.get("rewritten_query", original_query)

    # 更新 State 中的 rewrite_query
    state["rewritten_query"] = rewritten_query

    align_result = {}

    if len(item_names) > 0:
        query_results = step_4_vectorize_and_query(item_names)
        align_result = step_5_align_item_names(query_results)
    else:
        logger.info("Node: 未提取到商品名，跳过向量检索")

    state = step_6_check_confirmation(state, align_result, session_id, history, rewritten_query)

    # 7. 写入最终历史
    final_state = step_7_write_history(state, session_id, history, rewritten_query, message_id)

    # 将 history 存入 state，供后续节点（如 node_answer_output）使用
    final_state["history"] = history

    # 标记任务完成
    add_done_task(session_id, "node_item_name_confirm", is_stream)

    logger.info(f"Node: 处理结束, Final State Item Names: {final_state.get('item_names')}")
    return final_state


if __name__ == "__main__":
    # 模拟输入状态
    mock_state = {
        "session_id": "test_session_001",
        "original_query": "HAK 180 烫金机怎么用？",
        "is_stream": False
    }

    print(">>> 开始测试 node_item_name_confirm...")
    try:
        # 运行节点
        result_state = node_item_name_confirm(mock_state)

        print("\n>>> 测试完成！最终状态:")
        print(json.dumps(result_state, indent=2, ensure_ascii=False))

        # 简单验证
        if result_state.get("item_names"):
            print(f"\n[PASS] 成功提取并确认商品名: {result_state['item_names']}")
        else:
            print(f"\n[WARN] 未确认到商品名 (可能是向量库无匹配或LLM未提取)")

    except Exception as e:
        print(f"\n[FAIL] 测试运行出错: {e}")