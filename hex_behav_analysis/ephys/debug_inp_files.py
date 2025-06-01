#!/usr/bin/env python3
# ephys_file_tester.py - Tests file encoding and marker detection issues
import sys
import os
import base64
import re
import binascii

def inspect_file(file_path):
    """
    Thoroughly inspect a file for encoding issues and header markers.
    
    Args:
        file_path (str): Path to the file to inspect
    """
    print(f"\n=== Inspecting file: {file_path} ===")
    
    # Basic file info
    file_size = os.path.getsize(file_path)
    print(f"File size: {file_size} bytes")
    
    # Read the file content
    with open(file_path, 'rb') as f:
        content = f.read()
    
    # Check if file might be base64 encoded
    is_base64 = False
    try:
        # Try to decode a small chunk to test if it's base64
        test_chunk = content[:100]
        decoded_test = base64.b64decode(test_chunk)
        # Try decoding the whole file
        decoded_content = base64.b64decode(content)
        is_base64 = True
        print("✓ File appears to be base64 encoded")
        print(f"  Base64 decoded size: {len(decoded_content)} bytes")
        # Use the decoded content for further analysis
        content = decoded_content
    except Exception as e:
        print(f"✗ File is not base64 encoded: {str(e)}")
    
    # Always decode with Latin-1 first for detailed inspection
    latin1_text = content.decode('latin-1', errors='replace')
    
    print("\n=== Latin-1 Decoding Results ===")
    print(f"Successfully decoded file using latin-1 encoding")
    print(f"First 200 characters with latin-1 encoding:")
    print(f"'{latin1_text[:200]}'")
    
    # Check for data_start in Latin-1
    data_start_pos = latin1_text.find("data_start")
    if data_start_pos != -1:
        print(f"\n✓ Found 'data_start' at position {data_start_pos} using latin-1 encoding")
        # Show context around the marker
        start = max(0, data_start_pos - 20)
        end = min(len(latin1_text), data_start_pos + len("data_start") + 20)
        context = latin1_text[start:end]
        print(f"  Context: '{context}'")
        
        # Check for hidden characters
        hex_context = ' '.join(hex(ord(c))[2:].zfill(2) for c in latin1_text[data_start_pos:data_start_pos+len("data_start")])
        print(f"  Hex: {hex_context}")
        
        # Show the header using Latin-1
        header_str = latin1_text[:data_start_pos]
        header_lines = [line.strip() for line in header_str.splitlines() if line.strip()]
        print("\nHeader content with latin-1 encoding:")
        for line in header_lines:
            print(f"  {line}")
    else:
        print(f"✗ Could not find 'data_start' marker using latin-1 encoding")
    
    # Try different text decodings for comparison
    encodings = ['utf-8', 'latin-1', 'ascii', 'utf-16', 'windows-1252']
    successful_encodings = []
    
    print("\n=== Testing Other Text Encodings ===")
    for encoding in encodings:
        try:
            text = content.decode(encoding, errors='strict')
            successful_encodings.append(encoding)
            print(f"✓ Successfully decoded as {encoding}")
            
            # Check if data_start marker is found with this encoding
            if encoding != 'latin-1':  # We already checked latin-1
                if text.find("data_start") != -1:
                    print(f"  ✓ 'data_start' marker found at position {text.find('data_start')} with {encoding}")
                else:
                    print(f"  ✗ 'data_start' marker NOT found with {encoding}")
        except UnicodeDecodeError:
            print(f"✗ Failed to decode as {encoding}")
    
    # Continue with the original analysis but use latin1_text
    text = latin1_text
    
    # Search for "data_start" marker with variations
    print("\n=== Searching for data_start marker variations (using latin-1) ===")
    markers = [
        "data_start", "DATA_START", "Data_Start", 
        "data start", "datastart", "data-start"
    ]
    
    marker_positions = {}
    for marker in markers:
        pos = text.find(marker)
        if pos != -1:
            marker_positions[marker] = pos
            print(f"✓ Found '{marker}' at position {pos}")
            
            # Show context around the marker
            start = max(0, pos - 20)
            end = min(len(text), pos + len(marker) + 20)
            context = text[start:end]
            print(f"  Context: '{context}'")
            
            # Check for hidden characters
            hex_context = ' '.join(hex(ord(c))[2:].zfill(2) for c in text[pos:pos+len(marker)])
            print(f"  Hex: {hex_context}")
        else:
            print(f"✗ Did not find '{marker}'")
    
    if not marker_positions:
        print("! WARNING: No data_start marker found with any variation")
        
        # Find potential header-like content
        print("\n=== Looking for header-like content ===")
        header_patterns = ["trial_date", "trial_time", "timebase", "bytes_per", "data_format"]
        for pattern in header_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                pos = match.start()
                line_start = text.rfind('\n', 0, pos) + 1
                line_end = text.find('\n', pos)
                if line_end == -1:
                    line_end = len(text)
                line = text[line_start:line_end]
                print(f"  Found header field: '{line}' at position {pos}")
    
    # Look for binary data indicators
    binary_indicators = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F]')
    binary_matches = binary_indicators.search(text[:1000])
    if binary_matches:
        print("\n! File contains binary data in the header portion")
        print(f"  First binary character at position {binary_matches.start()}")
    
    # Check for common line endings
    line_endings = {
        "CRLF (Windows)": text.count('\r\n'),
        "LF (Unix)": text.count('\n') - text.count('\r\n'),
        "CR (Mac)": text.count('\r') - text.count('\r\n')
    }
    
    print("\n=== Line Ending Analysis ===")
    for ending, count in line_endings.items():
        print(f"{ending}: {count} occurrences")
    
    # Look for file structure
    print("\n=== File Structure Analysis ===")
    # Split into lines and analyze
    lines = re.split(r'\r?\n', text)
    print(f"Total lines: {len(lines)}")
    
    if len(lines) > 0:
        print("First 5 lines:")
        for i, line in enumerate(lines[:5]):
            print(f"  {i+1}: '{line}'")
        
        print("Last 5 lines before potential data start:")
        data_start_idx = -1
        for marker, pos in marker_positions.items():
            line_num = text[:pos].count('\n')
            data_start_idx = line_num
            break
        
        if data_start_idx != -1:
            start_idx = max(0, data_start_idx - 5)
            for i in range(start_idx, data_start_idx + 1):
                if i < len(lines):
                    print(f"  {i+1}: '{lines[i]}'")
        else:
            # If no data_start marker, show last 5 lines
            for i, line in enumerate(lines[-5:]):
                print(f"  {len(lines)-5+i+1}: '{line}'")
    
    # Summarize findings
    print("\n=== Summary ===")
    print("- Latin-1 Decoding:")
    if data_start_pos != -1:
        print(f"  ✓ Found 'data_start' marker at position {data_start_pos} using latin-1")
    else:
        print(f"  ✗ Could NOT find 'data_start' marker using latin-1")
    
    if is_base64:
        print("- File is base64 encoded")
    
    if successful_encodings:
        print(f"- File can be decoded as: {', '.join(successful_encodings)}")
    else:
        print("- File could not be properly decoded with common text encodings")
    
    if marker_positions:
        first_marker = next(iter(marker_positions.items()))
        print(f"- Found '{first_marker[0]}' marker at position {first_marker[1]}")
    else:
        print("- No data_start marker found with any variation")
        
    # Suggestions
    print("\n=== Suggestions ===")
    if data_start_pos != -1:
        print("1. Use latin-1 encoding for reliable parsing of this file")
        print("2. Update your parser to consistently use latin-1 encoding")
    elif not marker_positions:
        print("1. The file may be corrupted or have a different format than expected")
        print("2. Try opening the file in a hex editor to inspect the binary content")
        print("3. Check if the file is truncated or incomplete")
    elif is_base64:
        print("1. Modify the parser to handle base64 encoded files correctly")
        print("2. Ensure the base64 decoding is applied before searching for markers")
    else:
        print("1. Update the parser to handle the specific encoding: " + 
              (successful_encodings[0] if successful_encodings else "unknown"))
        print("2. Search for the marker variation found: " + 
              (next(iter(marker_positions.keys())) if marker_positions else "none found"))
    
    print("\nDone inspecting file.")

def main():
    """
    Command line interface for the script.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Test and debug ephys file format issues')
    parser.add_argument('file_path', help='Path to the file to inspect')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file_path):
        print(f"Error: File '{args.file_path}' does not exist.")
        return 1
        
    try:
        inspect_file(args.file_path)
        return 0
    except Exception as e:
        import traceback
        print(f"Error inspecting file: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())