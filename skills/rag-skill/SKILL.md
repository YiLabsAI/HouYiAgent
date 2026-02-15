---
name: kb-retriever
description: "A retrieval and Q&A assistant for local knowledge base directories. Core workflow: (1) hierarchical index navigation, (2) mandatory reference reading before handling PDF/Excel, (3) process files before retrieval. Combines grep, Read, pdfplumber, and pandas for progressive retrieval by file type, avoiding full-file loading. Use when the user asks to answer questions from the knowledge base, retrieve information, or look up materials."
---

# Local Knowledge Base Retrieval Skill (kb-retriever)

## Knowledge Base Directory Overview

- The knowledge base resides under a root directory and contains various file types (e.g., `.md`/`.txt`, `.pdf`, `.xlsx`), typically organized into multi-level subdirectories by type or business domain.
- A **hierarchical directory index file** system is used:
  - The root directory contains a `data_structure.md` that describes the main "domain directories" and their purposes.
  - Each domain directory may have its own `data_structure.md` describing the subdirectories/files within it and their respective purposes.
  - Deeper subdirectories may also contain `data_structure.md` files, forming a multi-level index tree.
- Knowledge base root directory conventions:
  - By default, the knowledge base is assumed to be at `knowledge/` under the current project root.
  - If the user explicitly specifies a different path in the conversation (e.g., "my knowledge base is at /data/kb" or "use ./docs as the knowledge base"), use the user-specified path as the root.
  - When the default `knowledge/` path does not exist or is inaccessible, ask the user to confirm the actual knowledge base root rather than guessing.
- Individual business files may be large:
  - Do not read entire files directly with Read
  - For PDF and Excel files, use the corresponding skills for structured processing first, then combine with grep/partial reads for precise retrieval

### Locating the `knowledge` Root Directory

- Prioritize user input: if the user provides a path (e.g., `./docs`, `./knowledge-personal`), use it directly.
- Default root: otherwise, use `knowledge/` under the current project.
  - Explicitly check directory existence via shell: prefer `test -d knowledge`, or fall back to `ls -d knowledge`.
  - Note: Do NOT use `Glob "knowledge" in .` patterns to determine directory existence. `Glob` only returns file paths, not directories. An empty result cannot distinguish between "directory does not exist" and "directory exists but is empty."
- Only use Glob to search within a directory after confirming its existence via `test -d`, specifying the directory as `path`. For example:
  - Index files: `pattern="**/data_structure.md"`, `path="knowledge"`
  - All Markdown: `pattern="**/*.md"`, `path="knowledge"`
- If the default `knowledge/` does not exist (`test -d` fails): do not guess other directories. Clearly inform the user that the default root was not found and ask them to specify the actual knowledge base path.

## Key Principle: Learn Before Processing

**Mandatory checklist when encountering PDF or Excel files**:

- [ ] Read the corresponding reference document to learn the processing method
- [ ] Understood the recommended tools and commands
- [ ] Completed file processing (extraction/conversion)
- [ ] Now ready to proceed with retrieval

**Prohibited actions**:
- Do not attempt to process PDFs without reading pdf_reading.md first
- Do not attempt to process Excel files without reading excel_reading.md first
- Do not skip the file processing step and search raw PDF/Excel files directly

## Overall Workflow

1. Understand the user's request
   - Parse the user's question to extract:
     - Topic/domain keywords (e.g., "sales report," "system architecture," "API documentation")
     - Time or scope constraints (e.g., "2023 Q1," "latest version")
     - Required output type (explanation, summary, specific field values, etc.)
   - Determine the knowledge base root directory:
     - First check whether the user specified a knowledge base path in the question.
     - Otherwise, use the default `knowledge/` root.
     - If the default root does not exist or the directory structure is abnormal, ask the user for confirmation rather than making assumptions.

2. Navigate the hierarchical directory index via `data_structure.md`
   - Maintain a "current working directory" concept:
     - Start from the user-specified knowledge base root; if none specified, use the current directory.
   - In the current working directory, if `data_structure.md` exists:
     - Read the first several hundred lines (e.g., limit=300) using Read, continuing in segments if necessary.
     - Goals:
       - Understand what subdirectories and files exist in the current directory
       - Understand the purpose of each subdirectory/file
     - Based on the user's question, select the **most relevant subdirectories or files** to form a candidate set.
   - For candidate subdirectories:
     - Recursively enter the subdirectory, making it the new "current working directory," and continue searching for its `data_structure.md`, repeating the process.
     - During recursion, avoid diving into all branches at once; prioritize the path most relevant to the question.
   - For candidate business files (md/text, PDF, Excel, etc.):
     - After completing the necessary directory-level exploration, collect these files as the final **retrieval target list**.
   - When prioritizing:
     - Prefer domain directories and files whose purpose descriptions closely match the question topic
     - Then consider time/version constraints (if reflected in the index)
     - General documentation (e.g., README.md, high-level design docs) should have lower priority

3. Learn file processing methods (mandatory for PDF/Excel)
   - **Before processing PDF files**:
     - **Must first read** [references/pdf_reading.md](references/pdf_reading.md) (note: this directory is under the Skills directory, not the Knowledge directory) to learn extraction methods
     - Focus on: pdftotext commands, pdfplumber usage, table extraction methods
   - **Before processing Excel files**:
     - **Must first read** [references/excel_reading.md](references/excel_reading.md) to learn reading methods
     - **Must first read** [references/excel_analysis.md](references/excel_analysis.md) to learn analysis methods
     - Focus on: pandas reading, column selection, data filtering
   - **Purpose**: Ensure correct tools and methods are used; avoid blind retrieval

4. Execute processing and retrieval by file type
   - Use the methods just learned to process files (extract, convert, structure)
   - For each candidate file, follow the strategies below for "Markdown/Text," "PDF," and "Excel"
   - General principles:
     - Start with the most relevant and precise files
     - Within each file, use progressive partial retrieval; avoid loading entire contents at once
     - If the current file does not yield satisfactory information, move to the next candidate

5. Iterative retrieval
   - All file types use the unified "multi-round iterative retrieval mechanism" (see Common Retrieval Principles below)

6. Answer composition and source attribution
   - Consolidate context gathered from multiple retrieval rounds to answer the user's question.
   - Best practices:
     - Provide clear, direct answers
     - Cite the file names used (include approximate locations such as sections or line/page numbers when necessary)
   - If the answer is based on inference or incomplete information:
     - Clearly note assumptions and uncertainties
     - Suggest that the user can refine the scope with more specific files or keywords

## Common Retrieval Principles

### Keyword Selection Strategy
- Extract 3-8 keywords from the user's question (including possible English abbreviations, synonyms, hypernyms/hyponyms)
- Combine word phrases (e.g., "sales report," "API timeout")
- Include business terms, technical jargon, and common abbreviations as needed (e.g., "UV," "PV," "GMV")

### grep Retrieval Fundamentals
- Always specify include patterns and paths as precisely as possible; avoid searching entire directories
- Try core nouns and terms from the question first, then try synonyms
- For each hit, read only the local region around the match (several lines above and below)
- Record "filename + location + text snippet"

### Multi-Round Iterative Retrieval Mechanism (up to 5 rounds)
All file types follow the same iterative strategy:
1. **Iteration control**
   - Maintain a "retrieval attempt count," maximum 5
   - Increment the count after each retrieval round
2. **Per-round workflow**
   1. Generate/update retrieval keywords based on the question (may include synonyms, expanded terms)
   2. Select files or file sections not yet fully searched
   3. Execute retrieval (grep / partial read / specialized skill call)
   4. Analyze the retrieved context snippets
   5. Determine whether the information is sufficient to answer the question
3. **Termination conditions**
   - Found sufficient context to support the answer; or
   - Reached 5 attempts without finding adequate information
4. **Handling insufficient information**
   - Clearly inform the user that the information is missing or may not exist in the current knowledge base
   - Provide the closest information found, noting any uncertainties
   - Suggest how the user can narrow the scope (more specific filenames, keywords, time ranges, etc.)

### Important Notes

- Do NOT call `Glob "knowledge" in .` or any Glob pattern to determine directory existence on the first attempt. Directory existence must be checked via shell commands (e.g., `test -d`).
- When using this skill to query the knowledge base, do not use web search or other external tools to obtain information.

## File-Type-Specific Strategies

### 1. Markdown / Text Files (.md, .txt, .log, etc.)

1. **Candidate file selection**
   - Judge relevance based on `data_structure.md`, filenames, and paths
   - Prioritize title and directory-type files (e.g., summary documents, design overviews)

2. **grep location and partial reading**
   - Use the Grep tool on specific candidate files; restrict with include patterns for specific extensions (e.g., "*.md")
   - For files with matches, use Read to read only the local region around the match:
     - Control reading with line offset and limit (e.g., read several dozen lines before and after the match)
     - Avoid full-file reads

3. **Special handling**
   - If content is only a table of contents or headings, follow links or section names to locate deeper content
   - Apply the "multi-round iterative retrieval mechanism" (see Common Retrieval Principles above)

### 2. PDF File Retrieval Strategy

**Workflow**:

1. **First: Read the processing guide**
   - Before processing any PDF, **must first read** [references/pdf_reading.md](references/pdf_reading.md) (note: this directory is under the Skills directory, not the Knowledge directory)
   - Focus on: pdftotext commands, pdfplumber usage, table extraction methods, quick decision table

2. **Select candidate PDFs**
   - Based on descriptions in `data_structure.md`, select the 1-3 most relevant files
   - If the user specifies a particular PDF, prioritize that file

3. **Apply learned methods to extract text**
   - Use the tools recommended in pdf_reading.md (prefer pdftotext or pdfplumber)
   - **Important**: Use `pdftotext input.pdf output.txt` to extract text to a file; do not output directly to stdout (to avoid consuming excessive tokens)
   - For table extraction, use pdfplumber's table extraction features

4. **Search the extracted results**
   - Use grep to search the extracted text by keyword
   - For each hit, extract context from the surrounding area (dozens of lines or adjacent pages)
   - Record "filename + page number/approximate location + text snippet"
   - Apply the "multi-round iterative retrieval mechanism" (see Common Retrieval Principles above)

### 3. Excel File Retrieval Strategy

**Workflow**:

1. **First: Read the processing guides**
   - Before processing any Excel file, **must first read**:
     - [references/excel_reading.md](references/excel_reading.md) - learn reading methods (note: this directory is under the Skills directory, not the Knowledge directory)
     - [references/excel_analysis.md](references/excel_analysis.md) - learn analysis methods (note: this directory is under the Skills directory, not the Knowledge directory)
   - Focus on: pandas reading methods, column selection, data filtering, aggregation operations

2. **Select candidate Excel files**
   - Based on `data_structure.md` and file/worksheet naming, select the most relevant sheets
   - Prefer workbooks/worksheets containing keywords like "report," "statistics," "log," "config," "mapping"
   - If the user specifies a particular Excel file, prioritize that file

3. **Apply learned methods to explore structure**
   - Use pandas to read the first 10-50 rows (use the `nrows` parameter to limit)
   - Focus on: column names/field names, data types (numeric, date, text), key fields
   - Compare column names with the user's question to identify potentially relevant fields (e.g., "revenue," "sales amount," "error_code")

4. **Execute data retrieval and analysis**
   - Use the learned pandas methods for filtering and aggregation (e.g., `df[df['column'] == value]`)
   - Read only the rows near matches each time; avoid loading the entire sheet at once
   - If the question includes a time range, add time-based filtering to the retrieval
   - Apply the "multi-round iterative retrieval mechanism" (see Common Retrieval Principles above)

## Collaboration with Other Tools

### PDF Processing
- **Must read** [references/pdf_reading.md](references/pdf_reading.md) before processing any PDF to learn the methods
- Use pdfplumber/pypdf for text extraction, table extraction, and metadata reading
- Prefer the pdftotext command-line tool for quick text extraction

### Excel Processing
- **Must read** before processing any Excel file:
  - [references/excel_reading.md](references/excel_reading.md) - learn reading methods
  - [references/excel_analysis.md](references/excel_analysis.md) - learn analysis methods
- Use pandas for data exploration, preview, filtering, and analysis

### Tool Usage Principles
- **Grep**: search for keywords in specified files to find line numbers and matching snippets; always specify precise include patterns and paths
- **Read**: use only for partial file reading; always set a reasonable limit (e.g., 200-500 lines) and an appropriate offset
- **For any potentially large file**:
  - Do not read from start to finish
  - Always narrow the scope first via index files, tables of contents, or keyword searches before reading

## Response Style and Error Handling

- Response style
  - Answer in the same language the user used (Chinese/English).
  - Lead with the conclusion, then provide brief supporting evidence.
  - When needed, list referenced files and approximate locations, for example:
    - Source: design/api_gateway.md around line 100
    - Source: reports/2023_Q1_sales.xlsx Summary worksheet
- When information is missing or uncertain
  - Clearly state that no exact match was found in the current knowledge base, or that only a partial answer is possible.
  - Do not fabricate facts.
  - Suggest how the user can help narrow the scope:
    - Specify a more specific directory/file
    - Provide more precise keywords or field names
    - Specify a time/version range
