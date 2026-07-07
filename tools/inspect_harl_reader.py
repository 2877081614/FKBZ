import json
from pathlib import Path


SOURCE_MAP = Path(
    r"D:\huang\Programs\防空编组\research_papers\04_heterogeneous_resource_coordination\P0_02_2023_HARL_Heterogeneous_Agent_RL_reader\source_map.json"
)


def main():
    data = json.loads(SOURCE_MAP.read_text(encoding="utf-8"))
    terms = [
        "Abstract",
        "Introduction",
        "Heterogeneous-Agent Trust Region Learning",
        "Heterogeneous-Agent Mirror Learning",
        "Lemma 4",
        "Lemma 6",
        "Theorem 7",
        "Theorem 8",
        "Lemma 13",
        "Theorem 14",
        "HATRPO",
        "HAPPO",
        "HAA2C",
        "HADDPG",
        "HATD3",
        "Experiments and Analysis",
        "Figure 1:",
        "Figure 2:",
        "Figure 3:",
        "Figure 4:",
        "Figure 5:",
        "Conclusion",
    ]
    for block in data["blocks"]:
        text = block["original"]
        if any(term in text for term in terms):
            print(f"\n### {block['id']} p{block['page']}\n{text[:1600]}")


if __name__ == "__main__":
    main()
