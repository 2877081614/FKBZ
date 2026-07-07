import json
import pathlib
import re

import pypdf


PDF_PATH = pathlib.Path(
    r"D:\huang\Programs\防空编组\research_papers\04_heterogeneous_resource_coordination\P0_02_2023_HARL_Heterogeneous_Agent_RL.pdf"
)
OUT_DIR = PDF_PATH.parent / "P0_02_2023_HARL_Heterogeneous_Agent_RL_reader"


def paragraph_blocks(text):
    section_re = re.compile(
        r"^(?:Abstract|Keywords|[0-9]+(?:\.[0-9]+)*\.?\s+|[A-Z]\.?\s+|Appendix|References|Acknowledgments|Algorithm\s+\d+)"
    )
    paras = []
    cur = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if section_re.match(line) and cur:
            paras.append(" ".join(cur))
            cur = []
        if len(line) < 78 and (section_re.match(line) or line.isupper()):
            if cur:
                paras.append(" ".join(cur))
                cur = []
            paras.append(line)
        else:
            cur.append(line)
    if cur:
        paras.append(" ".join(cur))
    for para in paras:
        para = re.sub(r"\s+", " ", para).strip()
        if para:
            yield para


def main():
    OUT_DIR.mkdir(exist_ok=True)
    reader = pypdf.PdfReader(str(PDF_PATH))
    pages = []
    blocks = []
    raw_pages = []
    sid = 1

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        raw_pages.append(f"\n\n===== PAGE {page_num} =====\n{text}")
        page_block_ids = []
        for para in paragraph_blocks(text):
            block_id = f"S{sid:04d}"
            sid += 1
            blocks.append(
                {
                    "id": block_id,
                    "page": page_num,
                    "type": "text",
                    "original": para,
                    "translation": "",
                    "confidence": "text-layer",
                }
            )
            page_block_ids.append(block_id)
        pages.append(
            {
                "page": page_num,
                "text_chars": len(text),
                "block_ids": page_block_ids,
            }
        )

    (OUT_DIR / "raw_pages.txt").write_text("".join(raw_pages), encoding="utf-8")
    (OUT_DIR / "source_map.json").write_text(
        json.dumps(
            {"pdf": str(PDF_PATH), "pages": pages, "blocks": blocks},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"out={OUT_DIR}")
    print(f"pages={len(pages)} blocks={len(blocks)} chars={sum(p['text_chars'] for p in pages)}")


if __name__ == "__main__":
    main()
