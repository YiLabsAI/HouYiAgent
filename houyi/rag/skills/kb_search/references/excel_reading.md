# Excel 文件读取方法指南

## 推荐工具

### pandas (Python)

最常用的 Excel 读取方式：

```python
import pandas as pd

# 读取整个工作表
df = pd.read_excel("input.xlsx")

# 读取特定工作表
df = pd.read_excel("input.xlsx", sheet_name="Sheet1")

# 限制读取行数（推荐用于大文件）
df = pd.read_excel("input.xlsx", nrows=100)

# 读取特定列
df = pd.read_excel("input.xlsx", usecols=["A", "B", "C"])
```

## 常用操作

### 1. 查看数据结构

```python
# 查看前几行
df.head(10)

# 查看列名
df.columns.tolist()

# 查看数据类型
df.dtypes

# 查看行列数
df.shape
```

### 2. 数据过滤

```python
# 按条件过滤
filtered = df[df["column_name"] == "value"]

# 多条件过滤
filtered = df[(df["col1"] > 100) & (df["col2"] == "A")]

# 包含特定文本
filtered = df[df["column"].str.contains("关键词", na=False)]
```

### 3. 数据聚合

```python
# 分组统计
summary = df.groupby("category").agg({
    "value": ["sum", "mean", "count"]
})

# 透视表
pivot = df.pivot_table(
    values="value",
    index="category",
    columns="month",
    aggfunc="sum"
)
```

## 最佳实践

1. **先探索结构**
   ```python
   # 只读取前 10 行了解结构
   df = pd.read_excel("input.xlsx", nrows=10)
   print(df.columns.tolist())
   ```

2. **按需读取列**
   ```python
   # 只读取需要的列
   df = pd.read_excel("input.xlsx", usecols=["日期", "销售额"])
   ```

3. **处理大文件**
   ```python
   # 分块读取
   for chunk in pd.read_excel("input.xlsx", chunksize=1000):
       process(chunk)
   ```

4. **日期处理**
   ```python
   df = pd.read_excel("input.xlsx", parse_dates=["日期"])
   ```

## 快速决策表

| 场景 | 方法 | 说明 |
|------|------|------|
| 查看结构 | `nrows=10` | 只读前10行 |
| 过滤数据 | `df[condition]` | 条件筛选 |
| 统计分析 | `groupby().agg()` | 分组聚合 |
| 大文件 | `chunksize` | 分块处理 |
