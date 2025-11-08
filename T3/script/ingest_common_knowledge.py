import json
import sys
from pathlib import Path


def _load_markdown_entries(md_path: Path) -> list[dict[str, str]]:
    """Parse the domain knowledge markdown file into discrete entries."""
    if not md_path.exists():
        raise FileNotFoundError(f"Knowledge file not found: {md_path}")

    raw_lines = md_path.read_text(encoding="utf-8").splitlines()

    entries: list[dict[str, str]] = []
    current_section = ""
    current_subsection = ""

    for raw_line in raw_lines:
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("## "):
            current_section = line[3:].strip()
            current_subsection = ""
            continue

        if line.startswith("### "):
            current_subsection = line[4:].strip()
            continue

        if line.startswith("- "):
            content = line[2:].strip()
            entries.append(
                {
                    "section": current_section or "General",
                    "subsection": current_subsection,
                    "content": content,
                }
            )
            continue

        # Capture leftover lines (for multi-line bullets or descriptions).
        if entries:
            previous = entries[-1]
            previous["content"] = f"{previous['content']} {line}".strip()
        else:
            entries.append(
                {
                    "section": current_section or "General",
                    "subsection": current_subsection,
                    "content": line,
                }
            )

    return entries


def ingest_common_knowledge() -> None:
    print("Starting domain knowledge ingestion...")

    script_path = Path(__file__).resolve()
    t3_root = script_path.parents[1]

    md_path = t3_root / "data" / "common_knowledge.md"
    chroma_path = t3_root / "chroma_db"
    temp_dir = t3_root / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_output_path = temp_dir / "domain_knowledge_docs.json"

    try:
        entries = _load_markdown_entries(md_path)
    except (FileNotFoundError, OSError) as exc:
        print(f"Error while loading knowledge file: {exc}")
        sys.exit(1)

    if not entries:
        print("No knowledge entries found; aborting ingestion.")
        return

    documents: list[str] = []
    metadatas: list[dict[str, str]] = []
    ids: list[str] = []

    for idx, entry in enumerate(entries):
        section = entry.get("section", "General")
        subsection = entry.get("subsection", "")
        content = entry.get("content", "")

        header = section if not subsection else f"{section} - {subsection}"
        documents.append(f"{header}: {content}")
        metadatas.append(
            {
                "section": section,
                "subsection": subsection,
                "source": str(md_path),
                "entry_index": str(idx),
            }
        )
        ids.append(f"domain_knowledge_{idx}")

    try:
        import chromadb
    except ImportError:
        print("Error: chromadb library is required. Please install it before running this script.")
        sys.exit(1)

    try:
        client = chromadb.PersistentClient(path=str(chroma_path))
        collection = client.get_or_create_collection(name="domain_knowledge")
        print(f"Preparing to upsert {len(documents)} knowledge entries...")

        temp_payload = {
            "documents": documents,
            "metadatas": metadatas,
            "ids": ids,
        }
        temp_output_path.write_text(json.dumps(temp_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Serialized payload written to {temp_output_path} for inspection.")

        collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
        print("Knowledge ingestion complete.")
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Failed to ingest knowledge into ChromaDB: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    ingest_common_knowledge()
