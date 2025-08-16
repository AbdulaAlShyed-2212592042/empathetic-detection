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
        conv_labeled_audio = set()  # track labeled audio in this conversation

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

    train_set, test_set, val_set = [], [], []

    for conv_id, convs in conv_id_to_convs.items():
        if conv_id in train_ids:
            train_set.extend(convs)
        elif conv_id in test_ids:
            test_set.extend(convs)
        elif conv_id in val_ids:
            val_set.extend(convs)

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

if __name__ == "__main__":
    json_path = "data/train_audio/audio_v5_0/train.json"
    audio_dir = "data/train_audio/audio_v5_0"

    all_data = map_and_reconstruct_all(json_path, audio_dir)
    train_set, test_set, val_set = split_dataset(all_data)

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
