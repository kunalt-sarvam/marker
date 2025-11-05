#!/usr/bin/env python3
"""
Convert Marker JSON output to clean HTML.
Resolves content-ref tags and creates a readable HTML document.
Extracts and saves images as separate files.
"""
import json
import re
import sys
import base64
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm


class MarkerJSONToHTML:
    """Convert Marker JSON to HTML"""
    
    def __init__(self, base_url=None):
        self.blocks_map = {}  # Map of block ID to block content
        self.images = {}  # Map of image ID to image data
        self.output_dir = None  # Directory where HTML will be saved
        self.base_url = base_url  # Base URL for absolute image paths (e.g., "/results/filename_base")
    
    def build_blocks_map(self, data: Dict):
        """Build a map of all blocks by their ID and extract images"""
        self.blocks_map = {}
        self.images = {}
        
        def traverse(node):
            if 'id' in node:
                self.blocks_map[node['id']] = node
                
                # Extract images if present
                if 'images' in node and node['images']:
                    for img_id, img_data in node['images'].items():
                        self.images[img_id] = img_data
            
            if 'children' in node and node['children']:
                for child in node['children']:
                    traverse(child)
        
        traverse(data)
    
    def add_bbox_to_html(self, html: str, bbox: List[float], page_num: int, block_type: str = '') -> str:
        """Add bbox, page, and block_type attributes to the first HTML tag"""
        if not html.strip():
            return html
        
        # Format bbox as string
        bbox_str = f'[{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]' if bbox else '[]'
        
        # Find the first opening tag and add attributes
        # Match tags like <p>, <h1>, <li>, <table>, etc.
        pattern = r'(<[a-zA-Z][^>]*?)(>)'
        
        def add_attr(match):
            tag_start = match.group(1)
            tag_end = match.group(2)
            # Add bbox, page, and type attributes before closing >
            attrs = f' bbox="{bbox_str}" page="{page_num + 1}"'
            if block_type:
                attrs += f' type="{block_type}"'
            return f'{tag_start}{attrs}{tag_end}'
        
        # Replace only the first occurrence
        result = re.sub(pattern, add_attr, html, count=1)
        return result
    
    def get_page_number(self, block_id: str) -> int:
        """Extract page number from block ID like '/page/0/Text/5'"""
        try:
            # ID format: /page/{page_num}/{block_type}/{block_id}
            parts = block_id.split('/')
            if len(parts) >= 3 and parts[1] == 'page':
                return int(parts[2])
        except:
            pass
        return 0
    
    def create_image_tag(self, block: Dict, block_id: str, bbox: List[float], page_num: int, block_type: str) -> str:
        """Create an img tag for image blocks"""
        if block_id not in self.images:
            return ''
        
        # Get image filename from block ID
        # e.g., /page/0/Picture/1 -> page_0_Picture_1.jpeg
        parts = block_id.replace('/', '_').strip('_')
        img_filename = f"{parts}.jpeg"
        
        # Format bbox
        bbox_str = f'[{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]' if bbox else '[]'
        
        # Get description
        description = block.get('description', '')
        
        # Alt text: use description or generic text
        alt_text = description if description else f"{block_type} on page {page_num + 1}"
        
        # Get title from metadata if available
        metadata = block.get('metadata', {})
        title = metadata.get('title', '') if metadata else ''
        
        # Create img tag with all metadata
        # Use absolute URL if base_url is provided, otherwise use relative
        if self.base_url:
            img_src = f"{self.base_url}/{img_filename}"
        else:
            img_src = f"./{img_filename}"
        img_tag = f'<img src="{img_src}" alt="{alt_text}"'

        if title:
            img_tag += f' title="{title}"'
        
        if description:
            img_tag += f' desc="{description}"'
        
        img_tag += f' type="{block_type}" bbox="{bbox_str}" page="{page_num + 1}"/>'
        
        return img_tag
    
    def process_block(self, block: Dict) -> str:
        """Process any block type and return HTML"""
        block_id = block.get('id', '')
        block_html = block.get('html', '')
        block_type = block.get('block_type', 'Unknown')
        bbox = block.get('bbox', [])
        page_num = self.get_page_number(block_id)
        
        # Handle image blocks (Picture, Figure)
        if block_type in ['Picture', 'Figure']:
            # Try to find image data
            if block_id in self.images:
                return self.create_image_tag(block, block_id, bbox, page_num, block_type)
            elif block.get('images'):
                for img_id in block.get('images', {}).keys():
                    if img_id in self.images:
                        return self.create_image_tag(block, img_id, bbox, page_num, block_type)
        
        # Handle group blocks (PictureGroup, FigureGroup, TableGroup, ListGroup)
        if block_type in ['PictureGroup', 'FigureGroup', 'TableGroup', 'ListGroup']:
            # Process children and combine
            children_html = []
            
            for child in block.get('children', []):
                child_html = self.process_block(child)
                if child_html:
                    children_html.append(child_html)
            
            # Wrap in div with metadata
            if children_html:
                bbox_str = f'[{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]' if bbox else '[]'
                content = '\n'.join(children_html)
                return f'<div type="{block_type}" bbox="{bbox_str}" page="{page_num + 1}">\n{content}\n</div>\n'
        
        # Handle regular blocks with HTML
        if block_html:
            # Resolve any content-refs in the HTML
            if '<content-ref' in block_html:
                block_html = self.resolve_content_refs(block_html)
            
            # Add metadata attributes
            block_html = self.add_bbox_to_html(block_html, bbox, page_num, block_type)
            return block_html
        
        # If block has children but no HTML, process children
        if block.get('children'):
            children_html = []
            for child in block.get('children', []):
                child_html = self.process_block(child)
                if child_html:
                    children_html.append(child_html)
            
            if children_html:
                bbox_str = f'[{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]' if bbox else '[]'
                content = '\n'.join(children_html)
                return f'<div type="{block_type}" bbox="{bbox_str}" page="{page_num + 1}">\n{content}\n</div>\n'
        
        return ''
    
    def resolve_content_refs(self, html: str) -> str:
        """Resolve content-ref tags to actual content"""
        # Find all content-ref tags
        pattern = r"<content-ref src='([^']+)'></content-ref>"
        
        def replace_ref(match):
            ref_id = match.group(1)
            if ref_id in self.blocks_map:
                block = self.blocks_map[ref_id]
                return self.process_block(block)
            return ''
        
        # Replace all content-refs
        resolved = re.sub(pattern, replace_ref, html)
        
        # Remove any remaining content-ref tags
        resolved = re.sub(r"<content-ref[^>]*></content-ref>", '', resolved)
        
        return resolved
    
    def clean_html(self, html: str) -> str:
        """Clean up HTML (remove block-type attributes, etc.)"""
        # Remove block-type attributes
        html = re.sub(r' block-type="[^"]*"', '', html)
        
        # Remove block-type='...' attributes (single quotes)
        html = re.sub(r" block-type='[^']*'", '', html)
        
        # Add newline after closing tags for better readability
        html = re.sub(r'(</[^>]+>)', r'\1\n', html)
        
        # Clean up extra whitespace
        html = re.sub(r'\n\s*\n\s*\n', '\n\n', html)
        
        return html.strip()
    
    def extract_body_content(self, data: Dict) -> str:
        """Extract and resolve the main content"""
        # Start with document children (pages)
        pages_content = []
        
        for page in data.get('children', []):
            if page.get('block_type') == 'Page':
                page_html = page.get('html', '')
                # Resolve all content-refs in this page
                resolved_html = self.resolve_content_refs(page_html)
                # Clean up
                resolved_html = self.clean_html(resolved_html)
                
                if resolved_html:
                    pages_content.append(resolved_html)
        
        return '\n'.join(pages_content)
    
    def save_images(self, output_dir: str):
        """Save all images as separate files"""
        if not self.images:
            return
        
        saved_count = 0
        with tqdm(total=len(self.images), desc="Saving images", leave=False) as pbar:
            for img_id, img_base64 in self.images.items():
                try:
                    # Decode base64 image
                    img_data = base64.b64decode(img_base64)
                    
                    # Create filename from block ID
                    parts = img_id.replace('/', '_').strip('_')
                    img_filename = f"{parts}.jpeg"
                    img_path = Path(output_dir) / img_filename
                    
                    # Save image
                    with open(img_path, 'wb') as f:
                        f.write(img_data)
                    
                    saved_count += 1
                except Exception as e:
                    tqdm.write(f"Warning: Could not save image {img_id}: {e}")
                finally:
                    pbar.update(1)
        
        if saved_count > 0:
            tqdm.write(f"  Saved {saved_count} images")
    
    def convert(self, json_path: str, output_html_path: str = None) -> str:
        """
        Convert JSON to HTML and save images.
        
        Args:
            json_path: Path to JSON file
            output_html_path: Optional path to save HTML (if None, returns HTML string)
        
        Returns:
            HTML string
        """
        # Load JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Build blocks map and extract images
        self.build_blocks_map(data)
        
        # Extract and resolve content
        body_content = self.extract_body_content(data)
        
        # Create full HTML document (without style tag)
        html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
</head>
<body>
{body_content}
</body>
</html>"""
        
        # Save HTML and images if output path provided
        if output_html_path:
            output_path = Path(output_html_path)
            self.output_dir = str(output_path.parent)
            
            # Save HTML
            with open(output_html_path, 'w', encoding='utf-8') as f:
                f.write(html)
            print(f"✓ Saved HTML to: {output_html_path}")
            
            # Save images to same directory
            self.save_images(self.output_dir)
        
        return html


def convert_single_file(json_file: Path, input_base: Path, output_base: Path, copy_images: bool = True, skip_existing: bool = True) -> bool:
    """Convert a single JSON file preserving directory structure"""
    try:
        # Calculate relative path from input base
        rel_path = json_file.relative_to(input_base)
        
        # Create output path preserving structure
        output_file = output_base / rel_path.with_suffix('.html')
        
        # Skip if output already exists
        if skip_existing and output_file.exists():
            return True  # Return True (success) but don't convert
        
        output_dir = output_file.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert
        converter = MarkerJSONToHTML()
        converter.convert(str(json_file), str(output_file))
        
        # Copy existing image files if they exist and output_base != input_base
        if copy_images and output_base != input_base:
            input_dir = json_file.parent
            for img_file in input_dir.glob('*.jpeg'):
                if img_file.stem.startswith('page_'):
                    output_img = output_dir / img_file.name
                    import shutil
                    shutil.copy2(img_file, output_img)
        
        return True
    except Exception as e:
        print(f"✗ Error converting {json_file.name}: {e}")
        return False


def main():
    """Main entry point"""
    import argparse
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    parser = argparse.ArgumentParser(
        description="Convert Marker JSON to HTML with parallel processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single file
  python json_to_html.py document.json

  # Convert directory (in-place)
  python json_to_html.py --directory ./marker_output
  
  # Convert with separate output directory
  python json_to_html.py --directory ./marker_output --output-dir ./html_output
  
  # Parallel processing
  python json_to_html.py --directory ./marker_output --workers 8
        """
    )
    
    parser.add_argument(
        'json_file',
        nargs='?',
        help='JSON file to convert'
    )
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        help='Output HTML file path (for single file)'
    )
    parser.add_argument(
        '--directory',
        '-d',
        type=str,
        help='Convert all JSON files in directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory (preserves structure from input directory)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Number of parallel workers (default: 1)'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing HTML files (default: skip existing)'
    )
    
    args = parser.parse_args()
    
    skip_existing = not args.overwrite
    
    # Convert directory
    if args.directory:
        directory = Path(args.directory)
        if not directory.exists():
            print(f"Error: Directory not found: {args.directory}")
            sys.exit(1)
        
        # Determine output base
        output_base = Path(args.output_dir) if args.output_dir else directory
        input_base = directory
        
        # Find all JSON files recursively
        print("Scanning for JSON files...")
        json_files = []
        for json_file in tqdm(list(directory.rglob('*.json')), desc="Finding files"):
            if not json_file.name.endswith('_meta.json'):
                json_files.append(json_file)
        
        if not json_files:
            print("No JSON files found")
            sys.exit(1)
        
        print(f"\nFound {len(json_files)} JSON files")
        print(f"Input directory: {input_base}")
        print(f"Output directory: {output_base}")
        print(f"Workers: {args.workers}")
        print()
        
        # Check which files need conversion
        files_to_convert = []
        skipped = 0
        
        with tqdm(total=len(json_files), desc="Checking existing files") as pbar:
            for json_file in json_files:
                rel_path = json_file.relative_to(input_base)
                output_file = output_base / rel_path.with_suffix('.html')
                
                if skip_existing and output_file.exists():
                    skipped += 1
                else:
                    files_to_convert.append(json_file)
                
                pbar.update(1)
        
        if skipped > 0:
            print(f"Skipping {skipped} existing files")
        
        if not files_to_convert:
            print("All files already converted!")
            return
        
        print(f"Converting {len(files_to_convert)} files\n")
        
        # Convert files
        converted = 0
        failed = 0
        
        if args.workers > 1:
            # Parallel processing
            print(f"Converting with {args.workers} parallel workers...\n")
            
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(convert_single_file, json_file, input_base, output_base, True, False): json_file
                    for json_file in files_to_convert
                }
                
                with tqdm(total=len(files_to_convert), desc="Converting files") as pbar:
                    for future in as_completed(futures):
                        json_file = futures[future]
                        try:
                            success = future.result()
                            if success:
                                converted += 1
                            else:
                                failed += 1
                        except Exception as e:
                            tqdm.write(f"✗ Error: {json_file.name}: {e}")
                            failed += 1
                        finally:
                            pbar.update(1)
                            pbar.set_postfix({"Success": converted, "Failed": failed})
        else:
            # Sequential processing
            print("Converting sequentially...\n")
            
            with tqdm(total=len(files_to_convert), desc="Converting files") as pbar:
                for json_file in files_to_convert:
                    success = convert_single_file(json_file, input_base, output_base, True, False)
                    if success:
                        converted += 1
                    else:
                        failed += 1
                    
                    pbar.update(1)
                    pbar.set_postfix({"Success": converted, "Failed": failed})
        
        print(f"\n✓ Converted {converted} files")
        if failed > 0:
            print(f"✗ Failed {failed} files")
    
    # Convert single file
    elif args.json_file:
        json_path = Path(args.json_file)
        if not json_path.exists():
            print(f"Error: File not found: {args.json_file}")
            sys.exit(1)
        
        # Determine output path
        if args.output:
            output_path = args.output
        else:
            output_path = json_path.with_suffix('.html')
        
        converter = MarkerJSONToHTML()
        converter.convert(str(json_path), output_path)
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()

