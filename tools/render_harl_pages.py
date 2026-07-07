from pathlib import Path

import pypdfium2 as pdfium


PDF_PATH = Path(
    r"D:\huang\Programs\防空编组\research_papers\04_heterogeneous_resource_coordination\P0_02_2023_HARL_Heterogeneous_Agent_RL.pdf"
)
OUT_DIR = PDF_PATH.parent / "P0_02_2023_HARL_Heterogeneous_Agent_RL_reader" / "assets"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = pdfium.PdfDocument(str(PDF_PATH))
    for page_num in [7, 10, 18, 21, 22, 23, 24, 25, 27, 28, 29, 30, 54]:
        page = pdf[page_num - 1]
        image = page.render(scale=2.0).to_pil()
        target = OUT_DIR / f"page_{page_num:02d}.png"
        image.save(target)
        print(target)


if __name__ == "__main__":
    main()
