#!/usr/bin/env python3
"""
Script to convert all Jupyter notebooks (.ipynb) to Markdown format.
Markdown files are named with '_markdown_for_llms.md' suffix for LLM readability.
"""

import os
import sys
import subprocess
from pathlib import Path
import nbformat
from nbconvert import MarkdownExporter
import re

def sanitize_filename(filename):
    """Sanitize filename by removing/replacing problematic characters."""
    # Remove or replace problematic characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove multiple consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    return sanitized



def convert_notebook_to_markdown(notebook_path):
    """Convert notebook to Markdown format."""
    try:
        # Load the notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        # Create Markdown exporter
        md_exporter = MarkdownExporter()
        md_exporter.exclude_input_prompt = True
        md_exporter.exclude_output_prompt = True
        
        # Convert to Markdown
        (body, resources) = md_exporter.from_notebook_node(nb)
        
        # Create output filename in same directory
        notebook_dir = os.path.dirname(notebook_path)
        notebook_name = Path(notebook_path).stem
        sanitized_name = sanitize_filename(notebook_name)
        md_filename = f"{sanitized_name}_markdown_for_llms.md"
        md_path = os.path.join(notebook_dir, md_filename)
        
        # Add header comment to Markdown content
        header_comment = f"# Converted from {os.path.basename(notebook_path)} - Markdown format optimized for LLM readability\n\n"
        body_with_header = header_comment + body
        
        # Write Markdown file
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(body_with_header)
        
        print(f"✓ Converted to Markdown: {md_path}")
        return md_path
        
    except Exception as e:
        print(f"✗ Error converting {notebook_path} to Markdown: {e}")
        return None

def find_all_notebooks(root_dir):
    """Find all .ipynb files in the directory tree."""
    notebooks = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.ipynb'):
                notebooks.append(os.path.join(root, file))
    return notebooks

def main():
    # Get the current directory
    current_dir = os.getcwd()
    
    print("🔍 Finding all Jupyter notebooks...")
    notebooks = find_all_notebooks(current_dir)
    
    if not notebooks:
        print("No .ipynb files found in the current directory tree.")
        return
    
    print(f"Found {len(notebooks)} notebook(s) to convert:")
    for nb in notebooks:
        print(f"  - {nb}")
    
    print(f"\n📁 Converted files will be saved in the same directories as the original notebooks")
    print(f"  Markdown files: *_markdown_for_llms.md")
    
    print(f"\n🔄 Converting notebooks...")
    
    successful_conversions = 0
    total_notebooks = len(notebooks)
    
    for notebook_path in notebooks:
        print(f"\n📝 Processing: {notebook_path}")
        
        # Convert to Markdown
        md_result = convert_notebook_to_markdown(notebook_path)
        
        if md_result:
            successful_conversions += 1
    
    print(f"\n✅ Conversion complete!")
    print(f"Successfully converted {successful_conversions}/{total_notebooks} notebooks")
    print(f"Files saved in the same directories as the original notebooks")

if __name__ == "__main__":
    main() 