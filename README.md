# RAG Knowledge Base

基于 LangGraph 的智能文档处理与检索增强生成（RAG）知识库系统。

## ✨ 功能特性

- **多格式文档解析**: 支持 PDF/Markdown 文档自动处理
- **智能文档分块**: 基于语义的文档分割与向量化
- **多模态向量检索**: 集成 BGE-M3 嵌入模型，支持稠密与稀疏向量
- **图片资源处理**: 自动提取 MD 文档图片并上传至 MinIO 对象存储
- **知识图谱集成**: 可选 Neo4j 实体关系抽取与存储
- **对话历史管理**: MongoDB 持久化存储会话记录
- **可观测性**: 完善的日志系统与错误处理

## 🏗️ 技术架构

```
文档上传 → PDF解析 → 文本分块 → 向量化 → Milvus存储 → 检索增强 → API服务
      ↓          ↓           ↓          ↓          ↓           ↓
   MinIO     MinerU     LangChain    BGE-M3    重排序      FastAPI
```

### 核心组件

| 组件 | 用途 | 技术选型 |
|------|------|----------|
| **文档处理** | PDF解析、格式转换 | MinerU + magic-pdf |
| **向量模型** | 文本嵌入生成 | BGE-M3 (稠密+稀疏) |
| **向量存储** | 向量相似度检索 | Milvus |
| **对象存储** | 图片/文件存储 | MinIO |
| **图数据库** | 实体关系存储 | Neo4j (可选) |
| **文档数据库** | 对话历史存储 | MongoDB |
| **工作流引擎** | 流程编排 | LangGraph |
| **LLM服务** | 大语言模型接口 | 通义千问/DashScope |
| **API服务** | REST接口服务 | FastAPI |

## 🚀 快速开始

### 环境要求

- Python >= 3.11
- CUDA >= 11.7 (GPU 加速推荐)
- Docker (用于 Milvus、MinIO、MongoDB 等基础设施)

### 安装步骤

1. **克隆项目**
   ```bash
   git clone <repository-url>
   cd knowledge-base
   ```

2. **创建虚拟环境**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Linux/Mac
   source .venv/bin/activate
   ```

3. **安装依赖**
   ```bash
   pip install -e .
   ```

4. **环境配置**
   复制项目根目录下的 `.env` 文件，根据实际情况修改配置项：
   ```bash
   # 参考现有 .env 文件格式，配置各服务连接信息
   # 重要：请替换其中的 API Key、密码等敏感信息
   ```

### 关键配置说明

```env
# LLM 服务 (通义千问)
OPENAI_API_KEY=sk-your-api-key
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_DEFAULT_MODEL=qwen-flash

# 向量模型
BGE_M3_PATH=./models/bge-m3
BGE_DEVICE=cuda:0  # 或 cpu

# 向量数据库
MILVUS_URL=http://localhost:19530
CHUNKS_COLLECTION=kb_chunks

# 对象存储
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET_NAME=knowledge-base-files

# 文档数据库
MONGO_URL=mongodb://localhost:27017
MONGO_DB_NAME=knowledge_base
```

5. **启动依赖服务**
   需要启动以下依赖服务（可通过 Docker 运行）：
   - **Milvus**: 向量数据库 (`docker run -d -p 19530:19530 milvusdb/milvus`)
   - **MinIO**: 对象存储 (`docker run -d -p 9000:9000 -p 9001:9001 minio/minio`)
   - **MongoDB**: 文档数据库 (`docker run -d -p 27017:27017 mongo`)
   - **Neo4j** (可选): 图数据库 (`docker run -d -p 7474:7474 -p 7687:7687 neo4j`)

6. **运行测试**
   ```bash
   # 测试文档导入流水线
   python -m app.import_process.agent.nodes.node_entry
   python -m app.import_process.agent.nodes.node_pdf_to_md
   ```

## 📁 项目结构

```
knowledge-base/
├── app/
│   ├── core/           # 核心模块（日志、配置、工具类）
│   ├── import_process/ # 文档导入流水线
│   │   └── agent/
│   │       ├── nodes/  # LangGraph 处理节点
│   │       │   ├── node_entry.py              # 入口校验
│   │       │   ├── node_pdf_to_md.py          # PDF转MD
│   │       │   ├── node_md_img.py             # 图片处理
│   │       │   ├── node_document_split.py     # 文档分块
│   │       │   ├── node_item_name_recognition.py # 实体识别
│   │       │   ├── node_bge_embedding.py      # 向量化
│   │       │   └── node_import_milvus.py      # Milvus入库
│   │       ├── state.py                       # 工作流状态定义
│   │       └── main_graph.py                  # LangGraph 主图定义
│   ├── clients/        # 数据库客户端
│   │   ├── milvus_utils.py    # Milvus 操作
│   │   ├── minio_utils.py     # MinIO 操作
│   │   ├── mongo_history_utils.py # MongoDB 操作
│   │   └── neo4j_utils.py     # Neo4j 操作
│   ├── lm/            # 大模型相关
│   │   ├── lm_utils.py        # LLM 客户端
│   │   ├── embedding_utils.py # 嵌入工具
│   │   └── reranker_utils.py  # 重排序模型
│   ├── utils/         # 工具函数
│   └── conf/          # 配置文件
├── logs/              # 日志目录
├── output/            # 处理输出目录
├── tests/             # 单元测试
├── pyproject.toml     # 项目依赖
├── .env              # 环境配置
└── README.md         # 本文档
```

## 🔧 使用示例

### 1. 文档导入流水线

```python
from app.import_process.agent.main_graph import kb_import_app
from app.import_process.agent.state import create_default_state

# 初始化状态
initial_state = create_default_state(
    file_path="path/to/document.pdf",
    task_id="task_001",
    is_pdf_read_enabled=True
)

# 执行导入流程
result = kb_import_app.invoke(initial_state)
print(f"导入完成: {result}")
```

### 2. 向量检索示例

```python
from app.clients.milvus_utils import search_similar_chunks

# 相似度搜索
results = search_similar_chunks(
    query="产品安全注意事项",
    collection_name="kb_chunks",
    top_k=5
)
```

### 3. API 服务启动

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## ⚙️ 配置说明

详细配置参数请参考项目根目录下的 `.env` 文件，其中包含所有可配置项说明。

## 📊 监控与日志

- **日志系统**: 基于 loguru，支持控制台和文件输出
- **日志级别**: DEBUG/INFO/WARNING/ERROR/CRITICAL 可配置
- **日志轮转**: 自动清理过期日志文件（默认保留7天）

## 🤝 贡献指南

1. Fork 本仓库
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [LangChain](https://github.com/langchain-ai/langchain) - LLM 应用开发框架
- [Milvus](https://milvus.io/) - 向量数据库
- [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) - 多语言嵌入模型
- [DashScope](https://dashscope.aliyun.com/) - 通义千问模型服务

## 📞 支持与反馈

如有问题或建议，请通过以下方式联系：
- 提交 [Issue](https://github.com/your-repo/issues)
- 邮件联系：your-email@example.com