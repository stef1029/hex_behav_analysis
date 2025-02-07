import json
from pathlib import Path
# Function to read numbers from a txt file
def read_numbers_from_txt(txt_file):
    with open(txt_file, 'r') as file:
        numbers = [int(line.strip()) for line in file]
    return numbers

# Function to update JSON file with new frame_IDs
def update_json_with_numbers(json_file, numbers):
    # Load the existing JSON data
    with open(json_file, 'r') as file:
        data = json.load(file)
    
    # Update the "frame_IDs" key with the new list of numbers
    data['frame_IDs'] = numbers
    
    # Save the updated JSON data back to the file
    with open(json_file, 'w') as file:
        json.dump(data, file, indent=4)

# Main function to execute the above
def main(txt_file, json_file):
    # Read numbers from txt file
    numbers = read_numbers_from_txt(txt_file)
    
    # Update JSON file with the numbers list
    update_json_with_numbers(json_file, numbers)

# Example usage
if __name__ == "__main__":
    json_file = Path(r"/cephfs2/srogers/Behaviour code/2409_September_cohort/Data/241001_134703/241001_134722_wtjx249-4b/241001_134722_wtjx249-4b-ArduinoDAQ.json")  # Replace with your .txt file
    txt_file = Path(r"/cephfs2/srogers/Behaviour code/2409_September_cohort/Data/241001_134703/241001_134722_wtjx249-4b/frame_ids.txt")   # Replace with your .json file
    
    main(txt_file, json_file)
