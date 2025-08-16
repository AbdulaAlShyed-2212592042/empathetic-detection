import os
import json
import random
from collections import Counter
from tqdm import tqdm
import re
import nltk

# Download nltk data if not already done
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# More complete contraction map for expansion
contraction_map = {
    "aren't": "are not",
    "can't": "cannot",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "isn't": "is not",
    "it's": "it is",
    "let's": "let us",
    "mightn't": "might not",
    "mustn't": "must not",
    "shan't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "shouldn't": "should not",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "we'd": "we would",
    "we're": "we are",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "where's": "where is",
    "who'd": "who would",
    "who'll": "who will",
    "who're": "who are",
    "who's": "who is",
    "who've": "who have",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "could've": "could have",
    "might've": "might have",
    "must've": "must have",
    "should've": "should have",
    "would've": "would have",
    "i've": "i have",
    "we've": "we have",
    "they've": "they have",
}

fillers = {"uh", "um", "ah", "er", "hmm", "mm", "like"}

def expand_contractions(text):
    pattern = re.compile(r'\b(' + '|'.join(re.escape(key) for key in contraction_map.keys()) + r')\b')
    def replace(match):
        return contraction_map[match.group(0)]
    return pattern.sub(replace, text)

def preprocess_text_for_bert(text):
    # Lowercase is okay but optional
    text = text.lower()
    text = expand_contractions(text)
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    # Remove digits (optional)
    text = re.sub(r"\d+", " ", text)
    # Remove fillers (noise)
    words = text.split()
    words = [w for w in words if w not in fillers]
    text = " ".join(words)
    # Don't aggressively remove punctuation or lemmatize - BERT tokenizer can handle that
    return text.strip()

def check_audio(audio_path, target_sr=16000):
    if audio_path is None or not os.path.isfile(audio_path):
        return False
    try:
        import librosa
        y, sr = librosa.load(audio_path, sr=target_sr)
        if len(y) < target_sr // 2:  # skip if audio shorter than 0.5s
            return False
        return True
    except Exception:
        return False

def preprocess_data_bert_wav2vec2(conversations):
    processed = []
    missing_audio_count = 0
    for conv in tqdm(conversations, desc="Preprocessing conversations"):
        if conv["speaker_emotion"] is None:
            continue

        valid_turns = []
        for turn in conv["turns"]:
            valid_utts = []
            for utt in turn["dialogue"]:
                utt["text"] = preprocess_text_for_bert(utt["text"])
                if utt["audio_name"] and check_audio(utt["audio_path"]):
                    valid_utts.append(utt)
                else:
                    missing_audio_count += 1
            if valid_utts:
                turn["dialogue"] = valid_utts
                valid_turns.append(turn)

        if valid_turns:
            conv["turns"] = valid_turns
            processed.append(conv)

    print(f"\nSkipped {missing_audio_count} utterances due to missing or invalid audio.")
    return processed

def map_emotions_to_ids(conversations):
    emotions = sorted(set(conv["speaker_emotion"] for conv in conversations if conv["speaker_emotion"] is not None))
    emotion2id = {em: idx for idx, em in enumerate(emotions)}
    for conv in conversations:
        conv["label_id"] = emotion2id.get(conv["speaker_emotion"], -1)
    return conversations, emotion2id

def map_and_reconstruct_all(json_path, audio_dir):
    with open(json_path, "r", encoding="utf-8") as f:
        conversations = json.load(f)

    all_audio_files = [f for f in os.listdir(audio_dir) if f.endswith(".wav") and not f.startswith("._")]
    all_audio_set = set(all_audio_files)

    total_audio_files = len(all_audio_files)
    mapped_audio_files = set()

    reconstructed_conversations = []

    for conv in conversations:
        conv_id = str(conv.get("conversation_id", "")).zfill(5)
        spk_id = conv.get("speaker_profile", {}).get("ID")
        lst_id = conv.get("listener_profile", {}).get("ID")

        reconstructed_turns = []

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

                turn_reconstructed.append({
                    "index": idx,
                    "role": role,
                    "text": text,
                    "audio_name": audio_name,
                    "audio_path": audio_path
                })

            last_idx = max([utt.get("index", -1) for utt in dialogue_history], default=-1)
            response_idx = last_idx + 1

            response_text = turn.get("response", "")
            response_audio_name = f"dia{conv_id}utt{response_idx}_{lst_id}.wav"
            response_audio_path = os.path.join(audio_dir, response_audio_name) if response_audio_name in all_audio_set else None
            if response_audio_name in all_audio_set:
                mapped_audio_files.add(response_audio_name)

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
            "speaker_emotion": speaker_emotion
        })

    print(f"Total audio files found: {total_audio_files}")
    print(f"Total mapped audio files: {len(mapped_audio_files)}")
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

    train_set = []
    test_set = []
    val_set = []

    for conv_id, convs in conv_id_to_convs.items():
        if conv_id in train_ids:
            train_set.extend(convs)
        elif conv_id in test_ids:
            test_set.extend(convs)
        elif conv_id in val_ids:
            val_set.extend(convs)

    def get_ids(dataset):
        return set(c["conversation_id"] for c in dataset)

    train_ids_check = get_ids(train_set)
    test_ids_check = get_ids(test_set)
    val_ids_check = get_ids(val_set)

    train_test_overlap = train_ids_check.intersection(test_ids_check)
    train_val_overlap = train_ids_check.intersection(val_ids_check)
    test_val_overlap = test_ids_check.intersection(val_ids_check)

    print(f"Total conversations: {len(conversations)}")
    print(f"Train set: {len(train_set)} conversations")
    print(f"Test set: {len(test_set)} conversations")
    print(f"Validation set: {len(val_set)} conversations\n")

    print("Data Leakage Report:")
    print(f"Train-Test overlap: {len(train_test_overlap)} conversations")
    print(f"Train-Val overlap: {len(train_val_overlap)} conversations")
    print(f"Test-Val overlap: {len(test_val_overlap)} conversations")
    if len(train_test_overlap) == 0 and len(train_val_overlap) == 0 and len(test_val_overlap) == 0:
        print("No Leakage detected. Clean splits.\n")
    else:
        print("Leakage Detected!\n")

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
        print(f"Chain of empathy: {turn['chain_of_empathy']}\n")

if __name__ == "__main__":
    json_path = "data/train_audio/audio_v5_0/train.json"
    audio_dir = "data/train_audio/audio_v5_0"

    all_data = map_and_reconstruct_all(json_path, audio_dir)

    all_data = preprocess_data_bert_wav2vec2(all_data)

    all_data, emotion_map = map_emotions_to_ids(all_data)

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
