# Contributing

1. 从`main`创建功能分支。
2. 不得在代码、测试或Issue中写入API密钥、参与者身份信息或原始音视频。
3. 所有文件路径必须通过`coregulation_poc.paths`解析，不得硬编码开发者机器路径。
4. 新增状态判断字段时，应同时更新`config/state_codebook.yaml`、Pydantic模型和测试。
5. 提交前运行`ruff check .`和`pytest`。

