import pathlib

import pypdfium2 as pdfium


PDF_PATH = pathlib.Path(
    r"D:\huang\Programs\防空编组\research_papers\04_heterogeneous_resource_coordination\P0_01_2021_HATRPO_HAPPO_Trust_Region_MARL.pdf"
)
OUT_DIR = PDF_PATH.parent / "P0_01_2021_HATRPO_HAPPO_Trust_Region_MARL_reader" / "assets"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = pdfium.PdfDocument(str(PDF_PATH))
    for page_num in [4, 8, 9, 26, 27]:
        page = pdf[page_num - 1]
        bitmap = page.render(scale=2.0)
        image = bitmap.to_pil()
        image.save(OUT_DIR / f"page_{page_num:02d}.png")
        print(OUT_DIR / f"page_{page_num:02d}.png")


if __name__ == "__main__":
    main()
