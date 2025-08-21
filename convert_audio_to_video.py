#!/usr/bin/env python3
"""
Script to convert audio references to video references in JSON files
Converts:
- audio_name -> video_name
- audio_path -> video_path  
- .wav -> .mp4
- train_audio -> train_video
- audio_v5_0 -> video_v5_0
"""

import json
import os
import re

def convert_audio_to_video_in_data(data):
    """Convert audio references to video references in the data structure"""
    if isinstance(data, dict):
        new_dict = {}
        for key, value in data.items():
            # Convert keys
            if key == 'audio_name':
                new_key = 'video_name'
                # Convert .wav to .mp4
                if isinstance(value, str) and value.endswith('.wav'):
                    value = value.replace('.wav', '.mp4')
            elif key == 'audio_path':
                new_key = 'video_path'
                # Convert path components
                if isinstance(value, str):
                    value = value.replace('train_audio', 'train_video')
                    value = value.replace('audio_v5_0', 'video_v5_0')
                    value = value.replace('.wav', '.mp4')
            else:
                new_key = key
            
            # Recursively process the value
            new_dict[new_key] = convert_audio_to_video_in_data(value)
        return new_dict
    elif isinstance(data, list):
        return [convert_audio_to_video_in_data(item) for item in data]
    else:
        return data

def convert_file(input_file, output_file):
    """Convert a JSON file from audio to video references"""
    print(f"Converting {input_file} -> {output_file}")
    
    # Read the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert audio references to video references
    converted_data = convert_audio_to_video_in_data(data)
    
    # Write the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Successfully converted {input_file}")

def main():
    """Main function to convert all mapped files"""
    base_dir = "json"
    
    # Files to convert
    files_to_convert = [
        ("mapped_train_data.json", "mapped_train_data_video.json"),
        ("mapped_val_data.json", "mapped_val_data_video.json"), 
        ("mapped_test_data.json", "mapped_test_data_video.json")
    ]
    
    print("🎬 Converting audio references to video references...")
    print("=" * 50)
    
    for input_file, output_file in files_to_convert:
        input_path = os.path.join(base_dir, input_file)
        output_path = os.path.join(base_dir, output_file)
        
        if os.path.exists(input_path):
            convert_file(input_path, output_path)
        else:
            print(f"⚠️ Warning: {input_path} not found, skipping...")
    
    print("=" * 50)
    print("🎉 Conversion completed!")
    print("\nNew video-based files created:")
    for _, output_file in files_to_convert:
        output_path = os.path.join(base_dir, output_file)
        if os.path.exists(output_path):
            print(f"   📁 {output_path}")

if __name__ == "__main__":
    main()
