"""Export KB documents to individual markdown files."""
import os
from knowledge_base.kb_docs import DOCUMENTS

def export_all():
    base_dir = os.path.dirname(__file__)
    for doc in DOCUMENTS:
        safe_topic = doc["topic"].replace(" ", "_").replace("/", "_")
        filename = os.path.join(base_dir, f"{doc['doc_id']}_{safe_topic}.md")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"# {doc['title']}\n\n")
            f.write(f"- **Document ID:** {doc['doc_id']}\n")
            f.write(f"- **Topic:** {doc['topic']}\n\n")
            f.write(f"{doc['content']}\n")
    print(f"Exported {len(DOCUMENTS)} documents to {base_dir}")

if __name__ == "__main__":
    export_all()
