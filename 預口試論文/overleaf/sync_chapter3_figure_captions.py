"""Update native LaTeX captions to match the readable chapter-3 drawings."""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "figsrc/ch03"))
from figure_style import CAPTIONS

REFERENCE_CAPTIONS = {
    "rotation_representation_tang2024":
        r"旋轉群與角度參數表示之比較：(a) $\mathrm{SO}(2)$ 旋轉群表示；"
        r"(b) 歐拉角參數表示（引自文獻\cite{tang2024gmpc}之圖~1）。",
}


def update(block):
    path = re.search(r"\\includegraphics(?:\[[^\]]*\])?\s*\{([^}]+)\}", block).group(1)
    caption = {**CAPTIONS, **REFERENCE_CAPTIONS}.get(Path(path).stem)
    if caption is None:
        return block
    start = block.index(r"\caption{") + len(r"\caption{")
    depth = 1
    end = start
    while depth:
        if block[end] == "{" and block[end - 1] != "\\":
            depth += 1
        elif block[end] == "}" and block[end - 1] != "\\":
            depth -= 1
        end += 1
    return block[:start] + caption + block[end - 1:]


def main():
    pieces = [
        "% !TeX program = xelatex",
        "% 第三章圖說更新：以每段完整 figure 取代 Overleaf 原本的同名圖。",
        "% 請勿整份 input，避免與正文重複；圖檔與 caption 需一起更新。",
        "",
    ]
    for source in sorted(ROOT.glob("03_0[1-7]_*.tex")):
        content = re.sub(r"\\begin\{figure\}.*?\\end\{figure\}",
                         lambda m: update(m.group()), source.read_text(), flags=re.S)
        source.write_text(content)
        sub = 0
        for match in re.finditer(
            r"\\subsection\{([^}]+)\}|\\begin\{figure\}.*?\\end\{figure\}", content, re.S
        ):
            if match.group(1):
                sub += 1
                continue
            pieces += [
                "% ==================================================",
                f"% 位置：3.{int(source.name[3:5])}.{sub}",
                f"% 正文來源：{source.name}",
                "% ==================================================",
                match.group(), "",
            ]
    (ROOT / "第三章_圖說更新.tex").write_text("\n".join(pieces) + "\n")
    print("Updated captions and exported 22 figure replacements.")


if __name__ == "__main__":
    main()
