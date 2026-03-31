import os
from pathlib import Path
from docling.document_converter import DocumentConverter
import time

# --- Configuration ---
INPUT_ROOT = "./data/raw/01"       # Where your PDFs are
OUTPUT_ROOT = "./data/markdown/01_md" # Where MD files will go

def batch_convert_preserve_structure(input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # 1. Initialize the Heavy Lifter (Docling) once
    print("Initializing Converter...")
    converter = DocumentConverter()
    
    # 2. Walk through every folder and subfolder
    # rglob('*') finds ALL files recursively
    files = [f for f in input_path.rglob('*') if f.is_file() and f.suffix.lower() == ".pdf"]
    
    if not files:
        print("No PDF files found!")
        return

    print(f"Found {len(files)} PDF files. Starting Batch Conversion...\n")

    success_count = 0
    
    for file_p in files:
        # Create the corresponding output path
        # logic: remove input_root prefix -> append to output_root -> change ext to .md
        relative_path = file_p.relative_to(input_path)
        dest_path = output_path / relative_path.with_suffix(".md")
        
        # Ensure the directory exists
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Skip if already exists (Resume capability!)
        if dest_path.exists():
            print(f"  [Skip] {relative_path} (Already converted)")
            continue

        print(f"  Processing: {relative_path}...")
        start_t = time.time()
        
        try:
            # --- The Conversion ---
            result = converter.convert(file_p)
            markdown_content = result.document.export_to_markdown()
            
            # --- Save to File ---
            with open(dest_path, "w", encoding="utf-8") as f:
                f.write(markdown_content)
            
            elapsed = time.time() - start_t
            print(f"     -> Done in {elapsed:.2f}s")
            success_count += 1
            
        except Exception as e:
            print(f"     [!] FAILED: {file_p.name} | Error: {e}")

    print(f"\nBatch Complete. Converted {success_count} new files.")

def get_folder_context(folder) -> str:
    folder = Path(folder)
    md_files = sorted(folder.rglob("*.md"))

    if not md_files:
        return ""

    parts = []
    for md_file in md_files:
        try:
            content = md_file.read_text(encoding="utf-8").strip()
            if content:
                parts.append(f"### {md_file.name}\n\n{content}")
        except Exception as e:
            print(f"  [!] Could not read {md_file.name}: {e}")

    return "\n\n---\n\n".join(parts)


# --- Run It ---
if __name__ == "__main__":
    batch_convert_preserve_structure(INPUT_ROOT, OUTPUT_ROOT)
