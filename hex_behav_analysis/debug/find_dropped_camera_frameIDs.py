import json

def analyze_sequence(filename):
    """
    Analyzes a sequence of numbers from either a text file or JSON file and identifies jumps larger than 1.
    For JSON files, it looks for numbers in the 'frame_IDs' key.
    
    Args:
        filename (str): Path to the file containing numbers (either .txt or .json)
        
    Returns:
        list: List of tuples containing (position, number1, number2, jump_size)
              for each jump larger than 1
    """
    try:
        # Determine file type and read numbers
        if filename.lower().endswith('.json'):
            with open(filename, 'r') as file:
                data = json.load(file)
                if 'frame_IDs' not in data:
                    raise KeyError("No 'frame_IDs' key found in JSON file")
                numbers = [float(num) for num in data['frame_IDs']]
        else:  # Assume text file
            with open(filename, 'r') as file:
                numbers = [float(num.strip()) for num in file.readlines() if num.strip()]
        
        if not numbers:
            print("No numbers found in file.")
            return []
            
        # List to store jumps
        large_jumps = []
        
        # Analyze sequence for jumps
        for i in range(len(numbers) - 1):
            current_num = numbers[i]
            next_num = numbers[i + 1]
            jump_size = abs(next_num - current_num)
            
            if jump_size > 1:
                large_jumps.append((i, current_num, next_num, jump_size))
        
        # Print results
        if large_jumps:
            print(f"\nFound {len(large_jumps)} jumps larger than 1:")
            for pos, num1, num2, jump in large_jumps:
                print(f"Position {pos}: {num1} -> {num2} (jump of {jump:.2f})")
        else:
            print("No jumps larger than 1 found in the sequence.")
            
        return large_jumps
            
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in file '{filename}'")
        return []
    except KeyError as e:
        print(f"Error: {str(e)}")
        return []
    except ValueError as e:
        print(f"Error: Invalid number found in file. Make sure all values are valid numbers.")
        return []

# Example usage
if __name__ == "__main__":
    # Example with text file
    txt_filename = r"Z:\debug_vids\250131_112303_wtjp273-3f\250131_112303_wtjp273-3f_Tracker_data.json"
    print("Analyzing text file:")
    analyze_sequence(txt_filename)
    