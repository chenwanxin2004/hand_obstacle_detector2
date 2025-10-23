# Makefile for Python项目 (支持 uv 和 pip)

.PHONY: help install install-dev test lint format clean run build uv-install uv-sync

help:  ## 显示帮助信息
	@echo "可用的命令:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# 传统 pip 命令
install:  ## 使用 pip 安装生产依赖
	pip install -r requirements.txt

install-dev:  ## 使用 pip 安装开发依赖
	pip install -r requirements-dev.txt

# uv 命令 (推荐)
uv-install:  ## 使用 uv 安装生产依赖
	uv pip install -r requirements.txt

uv-sync:  ## 使用 uv 同步所有依赖 (推荐)
	uv sync

uv-sync-dev:  ## 使用 uv 同步开发依赖
	uv sync --extra dev

# 通用命令
test:  ## 运行测试
	uv run pytest tests/ -v

test-cov:  ## 运行测试并生成覆盖率报告
	uv run pytest tests/ --cov=src --cov-report=html --cov-report=term

lint:  ## 运行代码检查
	uv run flake8 src/ tests/
	uv run mypy src/

format:  ## 格式化代码
	uv run black src/ tests/
	uv run isort src/ tests/

clean:  ## 清理生成的文件
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .venv/

run:  ## 运行项目
	uv run python src/project_name/main.py

build:  ## 构建项目
	uv run python setup.py sdist bdist_wheel

# 开发环境设置
dev-setup: uv-sync-dev  ## 使用 uv 设置开发环境

# 检查代码质量
check: lint test  ## 检查代码质量

# 准备发布
prepare-release: clean format lint test build  ## 准备发布

# 快速启动 (使用 uv)
quick-start: uv-sync  ## 快速启动项目 (推荐)
	@echo "🎉 项目依赖已安装完成!"
	@echo "运行项目: make run"
	@echo "运行测试: make test"
