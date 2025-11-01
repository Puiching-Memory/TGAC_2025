from __future__ import annotations

import json
import os


def main() -> None:
    """Render final_dataset.json into the pretty-printed TXT variant."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(script_dir, os.pardir, "data"))
    json_path = os.path.join(data_dir, "final_dataset.json")
    txt_path = os.path.join(data_dir, "final_dataset.txt")

    with open(json_path, "r", encoding="utf-8") as fh:
        dataset = json.load(fh)

    rendered = json.dumps(dataset, ensure_ascii=False, indent=4) + "\n"

    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write(rendered)

    print(f"Wrote {len(dataset)} records to {os.path.basename(txt_path)}")


if __name__ == "__main__":
    main()
