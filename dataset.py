import os
import json
import re

class AudioDataset:
    def __init__(self, json_path, root_dir):
        self.root_dir = root_dir
        self.data = []

        # Load JSON data (list of conversations)
        with open(json_path, 'r', encoding='utf-8') as f:
            self.conversations = json.load(f)

        # Build lookup dict for fast access: {(conv_id, turn_id, speaker_id): turn_data}
        self.turn_lookup = {}
        for conv in self.conversations:
            conv_id = str(conv["conversation_id"])  # Convert to str for matching with filenames
            speaker_id = str(conv["speaker_profile"]["ID"])
            listener_id = str(conv["listener_profile"]["ID"])
            for turn in conv["turns"]:
                turn_id = str(turn["turn_id"])

                # Map speaker turn keyed by conv_id, turn_id, speaker_id
                self.turn_lookup[(conv_id, turn_id, speaker_id)] = {
                    "audio_role": "speaker",
                    "conversation": conv,
                    "turn": turn,
                }
                # Future: Add listener turns if needed

        # Regex to parse filenames: e.g. dia28194utt0_60.wav
        pattern = re.compile(r"dia(\d+)utt(\d+)_(\d+)\.wav")

        # Scan directory for valid audio files that match JSON info
        for filename in os.listdir(self.root_dir):
            if not filename.endswith('.wav'):
                continue
            m = pattern.match(filename)
            if not m:
                continue
            conv_id, turn_id, audio_id = m.group(1), m.group(2), m.group(3)

            key = (conv_id, turn_id, audio_id)
            if key in self.turn_lookup:
                audio_path = os.path.join(self.root_dir, filename)
                entry = {
                    "audio_path": audio_path,
                    "conversation_id": conv_id,
                    "turn_id": turn_id,
                    "speaker_id": audio_id,
                    "turn_data": self.turn_lookup[key]["turn"],
                    "conversation_info": {
                        "conversation_id": conv_id,
                        "speaker_profile": self.turn_lookup[key]["conversation"]["speaker_profile"],
                        "listener_profile": self.turn_lookup[key]["conversation"]["listener_profile"],
                        "topic": self.turn_lookup[key]["conversation"]["topic"]
                    }
                }
                self.data.append(entry)

        if not self.data:
            raise RuntimeError(f"No valid audio files found in {root_dir} based on {json_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def print_summary(label, dataset, index):
    """Prints concise summary for a specific dataset index."""
    item = dataset[index]
    conv_info = item['conversation_info']
    turn = item['turn_data']
    dialogue_history = turn.get('dialogue_history', [])
    audio_text = None
    for utt in reversed(dialogue_history):
        if utt['role'] == 'speaker':
            audio_text = utt['utterance']
            break

    print(f"=== {label} Sample (index {index}) ===")
    print(f"Conversation ID: {conv_info.get('conversation_id', 'N/A')}")
    speaker_profile = conv_info.get('speaker_profile', {})
    listener_profile = conv_info.get('listener_profile', {})
    print(f"Speaker Profile: Age={speaker_profile.get('age', 'N/A')}, Gender={speaker_profile.get('gender', 'N/A')}, "
          f"Timbre={speaker_profile.get('timbre', 'N/A')}, ID={speaker_profile.get('ID', 'N/A')}")
    print(f"Listener Profile: Age={listener_profile.get('age', 'N/A')}, Gender={listener_profile.get('gender', 'N/A')}, "
          f"Timbre={listener_profile.get('timbre', 'N/A')}, ID={listener_profile.get('ID', 'N/A')}")
    print(f"Topic: {conv_info.get('topic', 'N/A')}")
    print(f"Audio path: {item['audio_path']}")
    print(f"Turn ID: {item['turn_id']}")
    print(f"Speaker ID: {item['speaker_id']}")
    print(f"Audio text: {audio_text}")
    print(f"Response text: {turn.get('response', 'N/A')}")
    print(f"Speaker emotion: {turn.get('chain_of_empathy', {}).get('speaker_emotion', 'N/A')}")
    print("========================\n")


def main():
    dataset_path = "data/train_audio/audio_v5_0"
    json_path = "data/train_audio/audio_v5_0/train.json"

    print(f"Loading dataset from json: {json_path}")
    dataset = AudioDataset(json_path=json_path, root_dir=dataset_path)

    total_samples = len(dataset)
    print(f"Dataset loaded with {total_samples} samples\n")

    # Print detailed info for all samples
    for i, item in enumerate(dataset):
        conv_info = item['conversation_info']
        turn = item['turn_data']
        dialogue_history = turn.get('dialogue_history', [])

        audio_text = None
        for utt in reversed(dialogue_history):
            if utt['role'] == 'speaker':
                audio_text = utt['utterance']
                break

        print(f"Conversation ID: {conv_info.get('conversation_id', 'N/A')}")
        speaker_profile = conv_info.get('speaker_profile', {})
        listener_profile = conv_info.get('listener_profile', {})
        print(f"Speaker Profile: Age={speaker_profile.get('age', 'N/A')}, Gender={speaker_profile.get('gender', 'N/A')}, "
              f"Timbre={speaker_profile.get('timbre', 'N/A')}, ID={speaker_profile.get('ID', 'N/A')}")
        print(f"Listener Profile: Age={listener_profile.get('age', 'N/A')}, Gender={listener_profile.get('gender', 'N/A')}, "
              f"Timbre={listener_profile.get('timbre', 'N/A')}, ID={listener_profile.get('ID', 'N/A')}")
        print(f"Topic: {conv_info.get('topic', 'N/A')}")
        print(f"Sample {i}:")
        print(f"  Audio path: {item['audio_path']}")
        print(f"  Turn ID: {item['turn_id']}")
        print(f"  Speaker ID: {item['speaker_id']}")
        print(f"  Audio text (last speaker utterance): {audio_text}")
        print(f"  Response text: {turn.get('response', 'N/A')}")
        print(f"  Speaker emotion: {turn.get('chain_of_empathy', {}).get('speaker_emotion', 'N/A')}\n")

    # Print summaries for first, middle and last samples
    if total_samples > 0:
        print_summary("First", dataset, 0)
        print_summary("Middle", dataset, total_samples // 2)
        print_summary("Last", dataset, total_samples - 1)


if __name__ == "__main__":
    main()
# This code is designed to load an audio dataset from a JSON file and a directory of audio files,
# providing detailed summaries of the dataset's contents, including speaker and listener profiles,  
# conversation topics, and audio file paths. It also includes functionality to print concise summaries
# for specific samples in the dataset.      
# The dataset is structured to allow easy access to conversation turns and their associated audio files
# based on conversation and turn IDs, facilitating analysis of dialogue interactions.