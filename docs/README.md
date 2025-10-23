# 项目文档

这个目录包含项目的详细文档。

## 文档结构

- `README.md` - 本文档
- `api/` - API文档
- `user_guide/` - 用户指南
- `developer_guide/` - 开发者指南

## 生成文档

项目使用Sphinx生成文档：

```bash
# 安装文档依赖
pip install -r requirements-dev.txt

# 生成HTML文档
cd docs
make html

# 查看生成的文档
# 打开 docs/_build/html/index.html
```

## 文档规范

- 使用Markdown格式编写
- 遵循Google风格的Python文档字符串
- 包含代码示例和说明
- 定期更新文档以反映代码变化
