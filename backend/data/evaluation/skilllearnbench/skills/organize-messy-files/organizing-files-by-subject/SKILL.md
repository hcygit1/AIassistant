---
name: "organizing-files-by-subject"
description: "Sorts a large directory of mixed document files (PDF, PPTX, DOCX) into predefined subject folders based on content analysis. Use when you need to organize 100+ files into 5 distinct categories, ensuring every file is moved exactly once with no leftovers."
metadata: { "openclaw": { "emoji": "🗂️" } }
---

# Organizing Files by Subject

## What this skill does
This workflow organizes a large, unstructured directory of document files (e.g., PDFs, PPTX, DOCX) into specific subject folders. It reads or extracts metadata from the files to determine their subject matter, creates the required destination folders, and moves all files so that no files remain in the original directory. The process ensures file names and contents remain completely unmodified.

## When to use this skill
- You have a large batch (>100) of mixed document files dumped into a single directory.
- The destination categories are predefined and mutually exclusive.
- You need to ensure every single file is moved exactly once, with zero files left behind.
- File contents and names must not be altered during the sorting process.

## Prerequisites
- Command-line access with `mkdir` and `mv` or `rsync` capabilities.
- Ability to read or extract text/metadata from PDF, DOCX, and PPTX files.
- Python environment or shell tools for text extraction (e.g., `pdftotext`, `python-docx`).

## Steps
1. **Inventory the source directory:** List all files in the target directory to establish a complete baseline. This is critical to verify later that zero files are left behind.
2. **Create destination folders:** Use `mkdir -p` to create the exact predefined subject folders. Ensure folder names match the required schema exactly (e.g., `LLM`, `trapped_ion_and_qc`, `black_hole`, `DNA`, `music_history`).
3. **Analyze file contents:** Read the text or metadata of each file. For ambiguous files, inspect deeper content (e.g., abstracts, titles) to ensure accurate categorization. Do not rely solely on filenames.
4. **Map files to categories:** Assign each file to exactly one subject folder. If a file does not clearly fit into four of the categories, assign it to the designated fallback category to ensure it is not left behind.
5. **Move the files:** Use `mv` to transfer files from the source directory into their respective subject folders. Moving (rather than copying) ensures that no "left out" files remain in the original directory.
6. **Clean up the source directory:** Verify the original source directory is now completely empty.

## Verification
1. **Check for missing files:** Compare the total count of files across all 5 subject folders against the initial baseline inventory. The counts must match exactly.
2. **Check for duplicates:** Ensure no file exists in more than one folder. Running a checksum or filename uniqueness check across the subject folders should return exactly the number of files in the baseline.
3. **Verify empty source:** Confirm the original holding directory contains 0 files.

## Pitfalls and solutions
- **Pitfall:** Files are copied instead of moved, leaving duplicates in the original directory and violating the "no other files left out" requirement.
  - **Fix:** Use `mv` instead of `cp`. After the operation, explicitly check the original directory and remove any leftover files if they were accidentally copied.
- **Pitfall:** Ambiguous files are misclassified into the wrong subject.
  - **Fix:** For files that do not immediately match a subject based on title, extract the first 1-2 pages of text or document metadata to determine the core topic before assigning the folder.