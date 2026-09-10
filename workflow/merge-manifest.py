import json
import argparse

def merge_manifests(main_path, new_path, output_path):
    main_data = {}
    with open(main_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                main_data[obj["audio_filepath"]] = obj

    with open(new_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                key = obj["audio_filepath"]
                if "text" in obj:
                    obj["pred_text"] = obj.pop("text")
                if key in main_data:
                    main_data[key].update(obj)
                else:
                    main_data[key] = obj

    with open(output_path, "w", encoding="utf-8") as f:
        for obj in main_data.values():
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Merged {len(main_data)} entries → {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge two manifest JSONL files")
    parser.add_argument("main_file", help="Main manifest file (keeps 'text' field)")
    parser.add_argument("new_file", help="New manifest file ('text' renamed to 'pred_text')")
    parser.add_argument("--output", default="gt_riva_merged_wer_manifest.jsonl", help="Output file path")
    args = parser.parse_args()

    merge_manifests(args.main_file, args.new_file, args.output)
