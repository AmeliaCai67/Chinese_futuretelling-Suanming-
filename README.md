# 中国传统算命系统

![八卦背景](./AI算命/static/baguazhen.jpg)

## 项目简介

本系统是一个融合传统命理学与现代人工智能技术的智能算命平台。通过整合多部经典命理典籍，结合先进的自然语言处理技术，为用户提供精准的命理分析和运势预测服务。

## 核心功能

- 🧠 多模型智能问答（支持本地模型、GLM、MiniMax等）
- 📚 命理典籍知识库检索
- 📅 运势预测与命盘分析
- 📊 对话历史记录与查询
- ⚙️ 参数实时调整与优化
- 🔒 数据加密存储与隐私保护

## 技术架构

### 前端

- HTML5/CSS3 构建响应式界面
- JavaScript 实现动态交互
- WebSocket 实时通信

### 后端

- Python Flask 构建 RESTful API
- SQLite 数据存储
- Sentence-Transformers 文本检索
- Scikit-learn 相似度计算

### AI 模型

- 本地模型：支持自定义部署
- 云端模型：GLM-4、MiniMax 等

## 命理知识库

系统整合了多部重要的命理典籍作为知识库（所有参考资料全部开源可查）:

- 《渊海子平》
- 《三命通会》
- 《千里命稿》
- 《命理约言》

## 快速开始

### 环境要求

- Python 3.10+
- GPU 支持（推荐）

### 安装步骤

#### 1. 克隆项目

```bash
git clone https://github.com/AmeliaCai67/Chinese_futuretelling-Suanming-.git
cd chinese-fortune-telling-system
```

#### 2. 创建Conda虚拟环境

```bash
conda create -n fortune-telling python=3.10
conda activate fortune-telling
```

#### 3. 安装依赖

```bash
pip install -r requirements.txt
```

#### 4. 启动服务

```bash
python AI算命/suanming.py
```

然后访问[http://127.0.0.1:8000](http://127.0.0.1:8000)即可开始。

### 配置说明

1. 在 `config/model_config.json` 中配置模型参数
2. 将文本编码模型放置于 `embedding/` 目录
3. 命理典籍数据存放于 `csv/` 目录

## 项目结构

```plaintext
AI算命/
├── static/                    # 前端静态资源
│   ├── baguazhen.jpg          # 八卦背景图
│   ├── index.html             # 首页
│   └── updated-fortune-telling-app.html  # 主界面
├── config/                    # 配置文件
│   └── model_config.json      # 模型配置
├── csv/                       # 命理典籍数据库
├── embedding/                 # 文本编码模型
├── suanming.py                # 后端主程序
└── res_database.db            # 查询记录数据库
```

## 使用指南

1. 启动服务后访问首页
2. 选择预测模型
3. 输入咨询问题
4. 查看系统分析结果
5. 可通过侧边栏查看历史记录、调整参数

## 注意事项

- 本系统仅供娱乐参考，不作为决策依据
- 请妥善保管个人隐私信息
- 建议在受信任的网络环境下使用
- 定期备份重要数据

## 开发路线图

- [x] 基础问答功能
- [x] 多模型支持
- [x] 历史记录存储
- [ ] 命盘可视化功能
- [ ] 运势趋势图表
- [ ] 移动端适配

## 贡献指南

欢迎通过 Issues 提交问题或 Pull Request 贡献代码。请遵循以下规范：

1. 代码风格符合 PEP8 标准
2. 提交前通过所有单元测试
3. 更新相关文档说明

## 许可证

本项目采用 MIT 开源许可证，详情请参阅 LICENSE 文件。
