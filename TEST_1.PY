import os
import json

def map_and_reconstruct_all(json_path, audio_dir):
    # Load JSON data
    with open(json_path, "r", encoding="utf-8") as f:
        conversations = json.load(f)

    # List all audio files, ignoring hidden/system files
    all_audio_files = [
        f for f in os.listdir(audio_dir)
        if f.endswith(".wav") and not f.startswith("._")
    ]
    all_audio_set = set(all_audio_files)

    total_audio_files = len(all_audio_files)
    mapped_audio_files = set()
    missing_audio_files = []

    reconstructed_conversations = []

    # Track turns missing emotion label
    unlabeled_emotion_turns = []

    for conv in conversations:
        conv_id = str(conv.get("conversation_id", "")).zfill(5)
        spk_id = conv.get("speaker_profile", {}).get("ID")
        lst_id = conv.get("listener_profile", {}).get("ID")

        reconstructed_turns = []

        for turn in conv.get("turns", []):
            dialogue_history = turn.get("dialogue_history", [])
            turn_reconstructed = []

            # Add dialogue history utterances with audio mapping
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

                if audio_name and audio_name in all_audio_set:
                    audio_path = os.path.join(audio_dir, audio_name)
                    mapped_audio_files.add(audio_name)
                else:
                    audio_path = None
                    if audio_name:
                        missing_audio_files.append(audio_name)

                # Check if emotion label exists
                emotion_label = utt.get("emotion")
                if audio_path and not emotion_label:
                    unlabeled_emotion_turns.append(audio_name)

                turn_reconstructed.append({
                    "index": idx,
                    "role": role,
                    "text": text,
                    "audio_name": audio_name,
                    "audio_path": audio_path,
                    "emotion": emotion_label
                })

            # Add the "response" as the listener_response utterance
            last_idx = max((utt.get("index", -1) for utt in dialogue_history), default=-1)
            response_idx = last_idx + 1

            response_text = turn.get("response", "")
            response_audio_name = f"dia{conv_id}utt{response_idx}_{lst_id}.wav"

            if response_audio_name in all_audio_set:
                response_audio_path = os.path.join(audio_dir, response_audio_name)
                mapped_audio_files.add(response_audio_name)
            else:
                response_audio_path = None
                missing_audio_files.append(response_audio_name)

            response_emotion = turn.get("emotion")
            if response_audio_path and not response_emotion:
                unlabeled_emotion_turns.append(response_audio_name)

            turn_reconstructed.append({
                "index": response_idx,
                "role": "listener_response",
                "text": response_text,
                "audio_name": response_audio_name,
                "audio_path": response_audio_path,
                "emotion": response_emotion
            })

            reconstructed_turns.append({
                "turn_id": turn.get("turn_id"),
                "context": turn.get("context", ""),
                "dialogue": sorted(turn_reconstructed, key=lambda x: x["index"]),
                "chain_of_empathy": turn.get("chain_of_empathy", {})
            })

        reconstructed_conversations.append({
            "conversation_id": conv_id,
            "speaker_profile": conv.get("speaker_profile", {}),
            "listener_profile": conv.get("listener_profile", {}),
            "topic": conv.get("topic", ""),
            "turns": reconstructed_turns
        })

    # Summary report
    print(f"Total audio files found: {total_audio_files}")
    print(f"Total mapped audio files: {len(mapped_audio_files)}")
    print(f"Total unmapped audio files: {total_audio_files - len(mapped_audio_files)}")
    if missing_audio_files:
        print(f"Missing audio files (sample): {missing_audio_files[:10]}")
        if len(missing_audio_files) > 10:
            print(f"...and {len(missing_audio_files) - 10} more")

    print(f"Total audio files missing emotion label: {len(unlabeled_emotion_turns)}")
    if unlabeled_emotion_turns:
        print(f"Sample unlabeled emotion files: {unlabeled_emotion_turns[:10]}")
        if len(unlabeled_emotion_turns) > 10:
            print(f"...and {len(unlabeled_emotion_turns) - 10} more\n")

    # Print all unlabeled audio files in first conversation
    if reconstructed_conversations:
        first_conv = reconstructed_conversations[0]
        print(f"\nUnlabeled audio files in first conversation (ID: {first_conv['conversation_id']}):")
        first_conv_unlabeled = []
        for turn in first_conv["turns"]:
            for utt in turn["dialogue"]:
                if utt["audio_path"] and not utt.get("emotion"):
                    first_conv_unlabeled.append(utt["audio_name"])
        for audio_name in first_conv_unlabeled:
            print(audio_name)
        print(f"Total unlabeled audio in first conversation: {len(first_conv_unlabeled)}")

    return reconstructed_conversations


if __name__ == "__main__":
    json_path = "data/train_audio/audio_v5_0/train.json"
    audio_dir = "data/train_audio/audio_v5_0"
    all_data = map_and_reconstruct_all(json_path, audio_dir)
