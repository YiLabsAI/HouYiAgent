# PDF 文件读取方法指南

## 推荐工具

### 1. pdftotext (命令行)

最快速的方式，适合提取纯文本：

```bash
# 提取到文本文件（推荐，避免大量输出）
pdftotext input.pdf output.txt

# 提取特定页面
pdftotext -f 1 -l 10 input.pdf output.txt  # 第1-10页
```

### 2. pdfplumber (Python)

适合需要提取表格的场景：

```python
import pdfplumber

with pdfplumber.open("input.pdf") as pdf:
    # 提取所有文本
    for page in pdf.pages:
        text = page.extract_text()
        print(text)

    # 提取表格
    for page in pdf.pages:
        tables = page.extract_tables()
        for table in tables:
            print(table)
```

### 3. pypdf (Python)

轻量级选择：

```python
from pypdf import PdfReader

reader = PdfReader("input.pdf")
for page in reader.pages:
    text = page.extract_text()
    print(text)
```

## 快速决策表

| 场景 | 推荐工具 | 说明 |
|------|---------|------|
| 快速提取文本 | pdftotext | 最快，无 Python 依赖 |
| 需要提取表格 | pdfplumber | 表格识别最准确 |
| 简单文本提取 | pypdf | 轻量，已集成 |
| 扫描件/图片PDF | OCR工具 | 需要额外处理 |

## 最佳实践

1. **先提取到文件，再搜索**
   ```bash
   pdftotext input.pdf output.txt
   grep "关键词" output.txt
   ```

2. **避免直接输出到 stdout**
   - 大文件会消耗大量 token
   - 使用临时文件存储提取结果

3. **分页处理大文件**
   - 使用 `-f` 和 `-l` 参数限制页面范围
   - 根据目录/索引定位相关页面

4. **表格数据**
   - 优先使用 pdfplumber
   - 将表格转换为 DataFrame 处理
