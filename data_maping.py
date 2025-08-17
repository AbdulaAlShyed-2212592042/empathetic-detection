import copy
import os
import re
import torch
import numpy as np
import json
import phonemizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd

def save_mapped_data_for_multimodal(conversations, output_path):
    """
    Save mapped data in the format expected by multimodal_empathetic_dialogue class.
    Each item contains: conversation_id, last turn, speaker_profile, listener_profile, topic
    """
    mapped_data = []
    for conv in conversations:
        # Use the last turn for each conversation
        turns = conv.get('turns', [])
        if not turns:
            continue
        last_turn = turns[-1]
        mapped_item = {
            'conversation_id': conv.get('conversation_id'),
            'turn': last_turn,
            'speaker_profile': conv.get('speaker_profile', {}),
            'listener_profile': conv.get('listener_profile', {}),
            'topic': conv.get('topic', '')
        }
        mapped_data.append(mapped_item)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(mapped_data, f, ensure_ascii=False, indent=2)
    print(f"Saved mapped data for multimodal_empathetic_dialogue to {output_path}")
import os
import json
import random
from collections import Counter

def map_and_reconstruct_all(json_path, audio_dir):
    with open(json_path, "r", encoding="utf-8") as f:
        conversations = json.load(f)

    all_audio_files = [f for f in os.listdir(audio_dir) if f.endswith(".wav") and not f.startswith("._")]
    all_audio_set = set(all_audio_files)

    total_audio_files = len(all_audio_files)
    mapped_audio_files = set()
    labeled_audio_files = set()

    reconstructed_conversations = []

    for conv in conversations:
        conv_id = str(conv.get("conversation_id", "")).zfill(5)
        spk_id = conv.get("speaker_profile", {}).get("ID")
        lst_id = conv.get("listener_profile", {}).get("ID")

        reconstructed_turns = []
        conv_labeled_audio = set()  

        for turn in conv.get("turns", []):
            dialogue_history = turn.get("dialogue_history", [])
            turn_reconstructed = []

            for utt in dialogue_history:
                idx = utt.get("index", 0)
                role = utt.get("role", "").lower()
                text = utt.get("utterance", "")

                if role == "speaker":
                    audio_name = f"dia{conv_id}utt{idx}_{spk_id}.wav"
                elif role == "listener":
                    audio_name = f"dia{conv_id}utt{idx}_{lst_id}.wav"
                else:
                    audio_name = None

                audio_path = os.path.join(audio_dir, audio_name) if audio_name in all_audio_set else None
                if audio_name in all_audio_set:
                    mapped_audio_files.add(audio_name)
                    if turn.get("chain_of_empathy", {}).get("speaker_emotion") is not None:
                        labeled_audio_files.add(audio_name)
                        conv_labeled_audio.add(audio_name)

                turn_reconstructed.append({
                    "index": idx,
                    "role": role,
                    "text": text,
                    "audio_name": audio_name,
                    "audio_path": audio_path
                })

            # Add listener response
            last_idx = max([utt.get("index", -1) for utt in dialogue_history], default=-1)
            response_idx = last_idx + 1
            response_text = turn.get("response", "")
            response_audio_name = f"dia{conv_id}utt{response_idx}_{lst_id}.wav"
            response_audio_path = os.path.join(audio_dir, response_audio_name) if response_audio_name in all_audio_set else None
            if response_audio_name in all_audio_set:
                mapped_audio_files.add(response_audio_name)
                if turn.get("chain_of_empathy", {}).get("speaker_emotion") is not None:
                    labeled_audio_files.add(response_audio_name)
                    conv_labeled_audio.add(response_audio_name)

            turn_reconstructed.append({
                "index": response_idx,
                "role": "listener_response",
                "text": response_text,
                "audio_name": response_audio_name,
                "audio_path": response_audio_path
            })

            reconstructed_turns.append({
                "turn_id": turn.get("turn_id"),
                "context": turn.get("context", ""),
                "dialogue": sorted(turn_reconstructed, key=lambda x: x["index"]),
                "chain_of_empathy": turn.get("chain_of_empathy", {})
            })

        speaker_emotion = None
        if conv.get("turns"):
            first_turn = conv["turns"][0]
            if "chain_of_empathy" in first_turn:
                speaker_emotion = first_turn["chain_of_empathy"].get("speaker_emotion", None)

        reconstructed_conversations.append({
            "conversation_id": conv_id,
            "speaker_profile": conv.get("speaker_profile", {}),
            "listener_profile": conv.get("listener_profile", {}),
            "topic": conv.get("topic", ""),
            "turns": reconstructed_turns,
            "speaker_emotion": speaker_emotion,
            "labeled_audio_names": list(conv_labeled_audio)  # unique labeled audio per conversation
        })

    print(f"Total audio files found: {total_audio_files}")
    print(f"Total mapped audio files: {len(mapped_audio_files)}")
    print(f"Total labeled audio files: {len(labeled_audio_files)}")
    print(f"Total unmapped audio files: {total_audio_files - len(mapped_audio_files)}\n")

    return reconstructed_conversations

def split_dataset(conversations, train_ratio=0.6, test_ratio=0.2, val_ratio=0.2, seed=42):
    conv_with_emotion = [c for c in conversations if c["speaker_emotion"] is not None]

    conv_id_to_convs = {}
    for c in conv_with_emotion:
        conv_id = c["conversation_id"]
        conv_id_to_convs.setdefault(conv_id, []).append(c)

    unique_conv_ids = list(conv_id_to_convs.keys())
    random.seed(seed)
    random.shuffle(unique_conv_ids)

    n = len(unique_conv_ids)
    n_train = int(n * train_ratio)
    n_test = int(n * test_ratio)
    n_val = n - n_train - n_test

    train_ids = set(unique_conv_ids[:n_train])
    test_ids = set(unique_conv_ids[n_train:n_train + n_test])
    val_ids = set(unique_conv_ids[n_train + n_test:])


    # Remove any overlapping conversation IDs from all splits except the first one they appear in
    assigned_ids = set()
    train_set, test_set, val_set = [], [], []
    for conv_id in unique_conv_ids:
        if conv_id in train_ids and conv_id not in assigned_ids:
            train_set.extend(conv_id_to_convs[conv_id])
            assigned_ids.add(conv_id)
        elif conv_id in test_ids and conv_id not in assigned_ids:
            test_set.extend(conv_id_to_convs[conv_id])
            assigned_ids.add(conv_id)
        elif conv_id in val_ids and conv_id not in assigned_ids:
            val_set.extend(conv_id_to_convs[conv_id])
            assigned_ids.add(conv_id)

    # ✅ Count unique labeled audio per split
    train_labeled_audio = len(set(audio for c in train_set for audio in c["labeled_audio_names"]))
    test_labeled_audio = len(set(audio for c in test_set for audio in c["labeled_audio_names"]))
    val_labeled_audio = len(set(audio for c in val_set for audio in c["labeled_audio_names"]))

    print(f"Total conversations (with emotion): {len(unique_conv_ids)}")
    print(f"Train set: {len(train_set)} conversations | Labeled audio: {train_labeled_audio}")
    print(f"Test set: {len(test_set)} conversations | Labeled audio: {test_labeled_audio}")
    print(f"Validation set: {len(val_set)} conversations | Labeled audio: {val_labeled_audio}\n")

    # Leakage check
    def get_ids(dataset):
        return set(c["conversation_id"] for c in dataset)

    train_ids_check = get_ids(train_set)
    test_ids_check = get_ids(test_set)
    val_ids_check = get_ids(val_set)

    print("Data Leakage Report:")
    print(f"Train-Test overlap: {len(train_ids_check & test_ids_check)} conversations")
    print(f"Train-Val overlap: {len(train_ids_check & val_ids_check)} conversations")
    print(f"Test-Val overlap: {len(test_ids_check & val_ids_check)} conversations")
    if not (train_ids_check & test_ids_check or train_ids_check & val_ids_check or test_ids_check & val_ids_check):
        print("No Leakage detected. Clean splits.\n")
    else:
        print("Leakage Detected!\n")

    # Emotion distribution
    def emotion_dist(dataset):
        return Counter(c["speaker_emotion"] for c in dataset)

    print("Train emotion distribution:", dict(emotion_dist(train_set)))
    print("Test emotion distribution:", dict(emotion_dist(test_set)))
    print("Validation emotion distribution:", dict(emotion_dist(val_set)))

    return train_set, test_set, val_set

def print_conversation(conv):
    print(f"--- Conversation ID: {conv['conversation_id']} ---")
    print(f"Topic: {conv['topic']}")
    print(f"Speaker profile: {conv['speaker_profile']}")
    print(f"Listener profile: {conv['listener_profile']}\n")

    for turn in conv["turns"]:
        print(f"Turn ID: {turn['turn_id']}")
        print(f"Context: {turn['context']}")
        print("Dialogue and Response:")
        for utt in turn["dialogue"]:
            print(f"  [{utt['index']}] {utt['role'].capitalize()}: {utt['text']}")
            print(f"      Audio file: {utt['audio_name']}")
            print(f"      Audio path: {utt['audio_path']}")
            if utt["role"] == "goal_of_response":
                print(f"      >>> This is the GOAL OF RESPONSE for this turn <<<")
        print(f"Chain of empathy: {turn['chain_of_empathy']}\n")

def save_dataset_splits(train_set, test_set, val_set, output_dir="json"):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save each split to a separate JSON file
    splits = {
        "train": train_set,
        "test": test_set,
        "val": val_set
    }

    for split_name, data in splits.items():
        output_path = os.path.join(output_dir, f"{split_name}_data.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved {split_name} set to {output_path}")

    # Save a summary file with dataset statistics
    summary = {
        "dataset_stats": {
            "train_conversations": len(train_set),
            "test_conversations": len(test_set),
            "val_conversations": len(val_set),
            "train_labeled_audio": len(set(audio for c in train_set for audio in c["labeled_audio_names"])),
            "test_labeled_audio": len(set(audio for c in test_set for audio in c["labeled_audio_names"])),
            "val_labeled_audio": len(set(audio for c in val_set for audio in c["labeled_audio_names"])),
            "emotion_distribution": {
                "train": dict(Counter(c["speaker_emotion"] for c in train_set)),
                "test": dict(Counter(c["speaker_emotion"] for c in test_set)),
                "val": dict(Counter(c["speaker_emotion"] for c in val_set))
            }
        }
    }

    summary_path = os.path.join(output_dir, "dataset_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved dataset summary to {summary_path}")

def save_class_distribution(self, output_path):
        """
        Count and save the number of entries per emotion class to a JSON file, and print each class entry.
        """
        class_counts = {emotion: 0 for emotion in self.emotion_projection.keys()}
        for item in self.data:
            emotion = item['turn']['chain_of_empathy'].get('speaker_emotion', None)
            if emotion in class_counts:
                class_counts[emotion] += 1
            else:
                # If emotion not in projection, count as 'other'
                class_counts.setdefault('other', 0)
                class_counts['other'] += 1
        # Print each class and its entry count
        print("Class distribution:")
        for emotion, count in class_counts.items():
            print(f"{emotion}: {count}")
        # Save to JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(class_counts, f, ensure_ascii=False, indent=2)
        print(f"Saved class distribution to {output_path}")

class multimodal_empathetic_dialogue(Dataset):
    def __init__(self, args):
        super(multimodal_empathetic_dialogue, self).__init__()
        self.args = args
        self.age_projection = {
            "child": 0,
            "young": 1,
            "middle-aged": 2,
            "elderly": 3
        }
        self.gender_projection = {
            "male": 0,
            "female": 1
        }
        self.timbre_projection = {
            "high": 0,
            "mid": 1,
            "low": 2
        }

        combinations = []
        self.profile_projection = {}
        label = 0
        for age in self.age_projection.keys():
            for gender in self.gender_projection.keys():
                for  timbre in self.timbre_projection.keys():
                    combination = f"{age}_{gender}_{timbre}"
                    combinations.append(combination)
                    self.profile_projection[combination] = label
                    label += 1

        self.ed_emotion_projection = {
            'conflicted': 'anxious',
            'vulnerability': 'afraid',
            'helplessness': 'afraid',
            'sadness': 'sad',
            'pensive': 'sentimental',
            'frustration': 'annoyed',
            'weary': 'tired',
            'anxiety': 'anxious',
            'reflective': 'sentimental',
            'upset': 'disappointed',
            'worried': 'anxious',
            'fear': 'afraid',
            'frustrated': 'sad',
            'fatigue': 'tired',
            'lost': 'jealous',
            'disappointment': 'disappointed',
            'nostalgia': 'nostalgic',
            'exhaustion': 'tired',
            'uneasy': 'anxious',
            'loneliness': 'lonely',
            'fragile': 'afraid',
            'confused': 'jealous',
            'vulnerable': 'afraid',
            'thoughtful': 'sentimental',
            'stressed': 'anxious',
            'concerned': 'anxious',
            'tiredness': 'tired',
            'burdened': 'anxious',
            'melancholy': 'sad',
            'overwhelmed': 'anxious',
            'worry': 'anxious',
            'heavy-hearted': 'sad',
            'melancholic': 'sad',
            'nervous': 'anxious',
            'fearful': 'afraid',
            'stress': 'anxious',
            'confusion': 'anxious',
            'inadequacy': 'ashamed',
            'regret': 'guilty',
            'helpless': 'afraid',
            'concern': 'anxious',
            'exhausted': 'tired',
            'overwhelm': 'anxious',
            'tired': 'tired',
            'disappointed': 'sad',
            'surprised': 'surprised',
            'excited': 'happy',
            'angry': 'angry',
            'proud': 'happy',
            'annoyed': 'angry',
            'grateful': 'happy',
            'lonely': 'sad',
            'afraid': 'fear',
            'terrified': 'fear',
            'guilty': 'sad',
            'impressed': 'surprised',
            'disgusted': 'disgusted',
            'hopeful': 'happy',
            'confident': 'happy',
            'furious': 'angry',
            'anxious': 'sad',
            'anticipating': 'happy',
            'joyful': 'happy',
            'nostalgic': 'sad',
            'prepared': 'happy',
            'jealous': 'contempt',
            'content': 'happy',
            'devastated': 'surprised',
            'embarrassed': 'sad',
            'caring': 'happy',
            'sentimental': 'sad',
            'trusting': 'happy',
            'ashamed': 'sad',
            'apprehensive': 'fear',
            'faithful': 'happy'       
        }

        self.emotion_projection = {
            "happy":0,
            "surprised":1,
            "angry":2,
            "fear":3,
            "sad":4,
            "disgusted":5,
            "contempt":6
        }

        with open(os.path.join(args['data_path'], args['mode']+'.json'), 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)
        print(f"Loaded {len(self.raw_data)} items from {os.path.join(args['data_path'], args['mode']+'.json')}")

    def save_full_class_summary(self, output_path):
        # Use self.data if available, else self.raw_data
        data_source = getattr(self, 'data', None)
        if data_source is None:
            data_source = getattr(self, 'raw_data', [])

        # Emotion class counts
        emotion_counts = {emotion: 0 for emotion in self.emotion_projection.keys()}
        emotion_counts['other'] = 0
        # Age, gender, timbre counts
        age_counts = {age: 0 for age in self.age_projection.keys()}
        gender_counts = {gender: 0 for gender in self.gender_projection.keys()}
        timbre_counts = {timbre: 0 for timbre in self.timbre_projection.keys()}

        for item in data_source:
            listener_profile = item.get('listener_profile', {})
            age = listener_profile.get('age', None)
            if age in age_counts:
                age_counts[age] += 1
            gender = listener_profile.get('gender', None)
            if gender in gender_counts:
                gender_counts[gender] += 1
            timbre = listener_profile.get('timbre', None)
            if timbre in timbre_counts:
                timbre_counts[timbre] += 1
            # Loop through all turns
            turns = item.get('turns', [])
            if not turns:
                # fallback for single turn format
                turns = [item.get('turn', {})]
            for turn in turns:
                chain_of_empathy = turn.get('chain_of_empathy', {})
                raw_emotion = chain_of_empathy.get('speaker_emotion', None)
                # Map to main class using ed_emotion_projection
                mapped_emotion = self.ed_emotion_projection.get(raw_emotion, raw_emotion)
                if mapped_emotion in self.emotion_projection:
                    emotion_counts[mapped_emotion] += 1
                else:
                    emotion_counts['other'] += 1

        print("Emotion class distribution:")
        for emotion in self.emotion_projection.keys():
            print(f"{emotion}: {emotion_counts[emotion]}")
        print(f"other: {emotion_counts['other']}")

        print("\nAge group distribution:")
        for age in self.age_projection.keys():
            print(f"{age}: {age_counts[age]}")

        print("\nGender distribution:")
        for gender in self.gender_projection.keys():
            print(f"{gender}: {gender_counts[gender]}")

        print("\nTimbre distribution:")
        for timbre in self.timbre_projection.keys():
            print(f"{timbre}: {timbre_counts[timbre]}")

        # Save all to JSON
        summary = {
            "emotion_counts": emotion_counts,
            "age_counts": age_counts,
            "gender_counts": gender_counts,
            "timbre_counts": timbre_counts
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"Saved full class summary to {output_path}")

ed_emotion_projection = {
    'conflicted': 'anxious',
    'vulnerability': 'afraid',
    'helplessness': 'afraid',
    'sadness': 'sad',
    'pensive': 'sentimental',
    'frustration': 'annoyed',
    'weary': 'tired',
    'anxiety': 'anxious',
    'reflective': 'sentimental',
    'upset': 'disappointed',
    'worried': 'anxious',
    'fear': 'afraid',
    'frustrated': 'sad',
    'fatigue': 'tired',
    'lost': 'jealous',
    'disappointment': 'disappointed',
    'nostalgia': 'nostalgic',
    'exhaustion': 'tired',
    'uneasy': 'anxious',
    'loneliness': 'lonely',
    'fragile': 'afraid',
    'confused': 'jealous',
    'vulnerable': 'afraid',
    'thoughtful': 'sentimental',
    'stressed': 'anxious',
    'concerned': 'anxious',
    'tiredness': 'tired',
    'burdened': 'anxious',
    'melancholy': 'sad',
    'overwhelmed': 'anxious',
    'worry': 'anxious',
    'heavy-hearted': 'sad',
    'melancholic': 'sad',
    'nervous': 'anxious',
    'fearful': 'afraid',
    'stress': 'anxious',
    'confusion': 'anxious',
    'inadequacy': 'ashamed',
    'regret': 'guilty',
    'helpless': 'afraid',
    'concern': 'anxious',
    'exhausted': 'tired',
    'overwhelm': 'anxious',
    'tired': 'tired',
    'disappointed': 'sad',
    'surprised': 'surprised',
    'excited': 'happy',
    'angry': 'angry',
    'proud': 'happy',
    'annoyed': 'angry',
    'grateful': 'happy',
    'lonely': 'sad',
    'afraid': 'fear',
    'terrified': 'fear',
    'guilty': 'sad',
    'impressed': 'surprised',
    'disgusted': 'disgusted',
    'hopeful': 'happy',
    'confident': 'happy',
    'furious': 'angry',
    'anxious': 'sad',
    'anticipating': 'happy',
    'joyful': 'happy',
    'nostalgic': 'sad',
    'prepared': 'happy',
    'jealous': 'contempt',
    'content': 'happy',
    'devastated': 'surprised',
    'embarrassed': 'sad',
    'caring': 'happy',
    'sentimental': 'sad',
    'trusting': 'happy',
    'ashamed': 'sad',
    'apprehensive': 'fear',
    'faithful': 'happy'       
}

def compute_class_distribution(data, age_projection, gender_projection, timbre_projection, ed_emotion_projection, emotion_projection):
    emotion_counts = {emotion: 0 for emotion in emotion_projection.keys()}
    emotion_counts['other'] = 0
    age_counts = {age: 0 for age in age_projection.keys()}
    gender_counts = {gender: 0 for gender in gender_projection.keys()}
    timbre_counts = {timbre: 0 for timbre in timbre_projection.keys()}
    for item in data:
        listener_profile = item.get('listener_profile', {})
        age = listener_profile.get('age', None)
        if age in age_counts:
            age_counts[age] += 1
        gender = listener_profile.get('gender', None)
        if gender in gender_counts:
            gender_counts[gender] += 1
        timbre = listener_profile.get('timbre', None)
        if timbre in timbre_counts:
            timbre_counts[timbre] += 1
        turns = item.get('turns', [])
        if not turns:
            turns = [item.get('turn', {})]
        for turn in turns:
            chain_of_empathy = turn.get('chain_of_empathy', {})
            raw_emotion = chain_of_empathy.get('speaker_emotion', None)
            mapped_emotion = ed_emotion_projection.get(raw_emotion, raw_emotion)
            if mapped_emotion in emotion_projection:
                emotion_counts[mapped_emotion] += 1
            else:
                emotion_counts['other'] += 1
    return {
        "emotion_counts": emotion_counts,
        "age_counts": age_counts,
        "gender_counts": gender_counts,
        "timbre_counts": timbre_counts
    }

def save_full_dataset_summary():
    import json
    # Load projections from class definition
    age_projection = {"child": 0, "young": 1, "middle-aged": 2, "elderly": 3}
    gender_projection = {"male": 0, "female": 1}
    timbre_projection = {"high": 0, "mid": 1, "low": 2}
    emotion_projection = {"happy":0, "surprised":1, "angry":2, "fear":3, "sad":4, "disgusted":5, "contempt":6}
    # use ed_emotion_projection directly
    # Load splits
    with open('json/mapped_train_data.json', 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open('json/mapped_test_data.json', 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    with open('json/mapped_val_data.json', 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    # Compute distributions
    train_stats = compute_class_distribution(train_data, age_projection, gender_projection, timbre_projection, ed_emotion_projection, emotion_projection)
    test_stats = compute_class_distribution(test_data, age_projection, gender_projection, timbre_projection, ed_emotion_projection, emotion_projection)
    val_stats = compute_class_distribution(val_data, age_projection, gender_projection, timbre_projection, ed_emotion_projection, emotion_projection)
    # Get conversation and labeled audio counts
    with open('json/dataset_summary.json', 'r', encoding='utf-8') as f:
        summary = json.load(f)["dataset_stats"]
    # Save all to one file
    full_summary = {
        "train": train_stats,
        "test": test_stats,
        "val": val_stats,
        "train_conversations": summary["train_conversations"],
        "test_conversations": summary["test_conversations"],
        "val_conversations": summary["val_conversations"],
        "train_labeled_audio": summary["train_labeled_audio"],
        "test_labeled_audio": summary["test_labeled_audio"],
        "val_labeled_audio": summary["val_labeled_audio"]
    }
    with open('json/full_class_summary.json', 'w', encoding='utf-8') as f:
        json.dump(full_summary, f, ensure_ascii=False, indent=2)
    print("Saved full class summary for train, test, val splits to json/full_class_summary.json")
# Call this function in your main block after all splits are saved

if __name__ == "__main__":
    json_path = "data/train_audio/audio_v5_0/train.json"
    audio_dir = "data/train_audio/audio_v5_0"

    all_data = map_and_reconstruct_all(json_path, audio_dir)
    train_set, test_set, val_set = split_dataset(all_data)

    # Save the dataset splits and summary
    save_dataset_splits(train_set, test_set, val_set)

    # Save mapped data for multimodal_empathetic_dialogue (example: train set)
    save_mapped_data_for_multimodal(train_set, os.path.join('json', 'mapped_train_data.json'))
    save_mapped_data_for_multimodal(test_set, os.path.join('json', 'mapped_test_data.json'))
    save_mapped_data_for_multimodal(val_set, os.path.join('json', 'mapped_val_data.json'))

    # Save class summary for train set
    from torch.utils.data import Dataset
    class_args = {'data_path': 'json', 'mode': 'mapped_train_data'}
    dataset = multimodal_empathetic_dialogue(class_args)
    dataset.save_full_class_summary(os.path.join('json', 'full_class_summary.json'))

    # Print sample conversations
    if train_set:
        print("\n### First and Last conversation in TRAIN set ###")
        print_conversation(train_set[0])
        print_conversation(train_set[-1])
    if test_set:
        print("\n### First and Last conversation in TEST set ###")
        print_conversation(test_set[0])
        print_conversation(test_set[-1])
    if val_set:
        print("\n### First and Last conversation in VALIDATION set ###")
        print_conversation(val_set[0])
        print_conversation(val_set[-1])

    # Save full dataset summary
    save_full_dataset_summary()
