"""Package the readable drawings and their exact LaTeX caption replacements."""
from pathlib import Path
import json
import re
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parent
FOOTERS = {
    "base_pose_error": "先求相對位姿",
    "wholebody_motion_coupling": "底盤位姿改變時",
    "arm_setpoint_tracking": "虛線：位置設定點",
    "visual_time_alignment": "影像處理完成後",
    "grasp_state_transition": "抓持成立後才銜接",
    "command_execution_timeline": "這段既定命令",
    "tcp_target_compensation": "完成條件仍對",
    "mobile_approach_distance_profile": "實際進度另受",
}


def main():
    sources = sorted(ROOT.glob("03_0[1-7]_*.tex"))
    manuscript = "\n".join(p.read_text() for p in sources)
    images = [
        ROOT / p for p in re.findall(
            r"\\includegraphics(?:\[[^\]]*\])?\s*\{([^}]+)\}", manuscript
        )
    ]
    assert len(images) == len(set(images)) == 22
    own = [p for p in images if p.parent.name == "ch03"]
    assert len(own) == 19
    manifest = json.loads((ROOT / "figsrc/ch03/figure_manifest.json").read_text())
    manifest += json.loads((ROOT / "figsrc/chapter_additions_manifest.json").read_text())
    records = {r["file"]: r for r in manifest}
    checks = []
    for p in own:
        record = records[p.stem]
        assert record["minimum_print_font_pt"] >= 11
        assert record["print_width_cm"] == 14.73
        assert not record["explanatory_footer"]
        info = subprocess.check_output(["pdfinfo", str(p)], text=True)
        assert re.search(r"^Pages:\s+1\s*$", info, re.M)
        fonts = subprocess.check_output(["pdffonts", str(p)], text=True).splitlines()[2:]
        assert fonts and any("DFKai" in row for row in fonts)
        assert all(re.search(r"\s+yes\s+(?:yes|no)\s+(?:yes|no)\s+\d+\s+\d+\s*$", row) for row in fonts)
        rasters = subprocess.check_output(["pdfimages", "-list", str(p)], text=True).splitlines()[2:]
        assert not rasters, p
        pdf_text = subprocess.check_output(["pdftotext", str(p), "-"], text=True)
        plain = re.sub(r"\s+", "", pdf_text)
        assert FOOTERS.get(p.stem, "\0") not in plain, p
        panels = sorted(set(re.findall(r"\([a-z]\)", pdf_text)))
        if panels:
            block = next(b for b in re.findall(
                r"\\begin\{figure\}.*?\\end\{figure\}", manuscript, re.S
            ) if str(p.relative_to(ROOT)) in b)
            caption = block[block.index(r"\caption{"):]
            assert all(panel in caption for panel in panels), (p, panels)
        checks.append(dict(
            file=str(p.relative_to(ROOT)), embedded_kaishu=True, vector_only=True,
            target_print_width_cm=14.73, primary_font_at_target_width_pt=11.0,
            no_explanatory_footer=True, panels=panels, caption_panels_match=True,
        ))
    report = ROOT / "第三章_圖片字級與圖說核對.json"
    report.write_text(json.dumps(dict(
        date="2026-11-25", original_drawings=19, reference_drawings=3, checks=checks,
        xelatex_compile=False, scope="Figure typography, explanatory footers and panel captions.",
    ), ensure_ascii=False, indent=2) + "\n")
    files = images + [p.with_suffix(".png") for p in images]
    files += [ROOT / "第三章_圖說更新.tex", report]
    files += list((ROOT / "figsrc/ch03").glob("chapter3_figures_overview_*.png"))
    package = ROOT.parent / "第三章_圖片字級與圖說修訂.zip"
    note = """第三章圖片字級與圖說修訂（2026-11-25）
1. 將 figsrc 內同名 PDF／PNG 覆蓋上傳到學姐的 Overleaf 專案，維持目錄。
2. 開啟「第三章_圖說更新.tex」，依各段註解的節次，以整個 figure 取代原本同名圖。
   此檔供逐段複製，請勿整份 input，避免與正文重複。
3. 自繪圖使用 0.98\\textwidth。以約 14.7 cm 寬排版時，主要文字約 11 pt。
4. 圖檔內已移除額外長句；正式圖說仍由 LaTeX caption 放在圖片下方。
5. 自繪分圖內只留 (a)、(b) 等字母，各分圖意義寫入 caption。
6. 中文使用標楷體，保留黑白樣式。PDF 為向量圖並嵌入字型。
7. 19 張自繪圖已重製；3 張原文參考圖保留來源圖面。
8. 圖表與交叉引用已核對，整章分頁仍由 Overleaf 編譯確認。
"""
    with zipfile.ZipFile(package, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(set(files)):
            assert p.is_file(), p
            z.write(p, str(p.relative_to(ROOT)))
        z.writestr("使用方式.txt", note)
    with zipfile.ZipFile(package) as z:
        assert z.testzip() is None
    print(json.dumps(dict(
        original_drawings=len(checks), active_figures=len(images),
        package=str(package), bytes=package.stat().st_size,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
