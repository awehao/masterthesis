"""Audit the standalone replacement chapter and prepare delivery metadata."""
from pathlib import Path
import collections
import csv
import json
import re
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "預口試論文"
BUNDLE = BASE / "第三章_精簡重整版"
OLD = Path("/home/howardchen/.codex/attachments/f879d175-c68b-4d98-bca8-232e6c666fd5/Pasted text.txt")
TEX = BUNDLE / "03_ResearchMethodology.tex"


def clean(s):
    return re.sub(r"(?<!\\)%[^\n]*", "", s)


def stats(s):
    return {
        "source_lines": len(s.splitlines()),
        "source_characters": len(s),
        "han_characters": len(re.findall(r"[\u4e00-\u9fff]", s)),
        "numbered_equation_groups": s.count(r"\begin{equation}"),
        "figures": s.count(r"\begin{figure}"),
        "sections": s.count(r"\section{"),
        "subsections": s.count(r"\subsection{"),
    }


def audit():
    source = TEX.read_text()
    s = clean(source)
    original = OLD.read_text()
    issues = []
    stack = []
    for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", s):
        kind, env = m.groups()
        if kind == "begin":
            stack.append(env)
        elif not stack or stack[-1] != env:
            issues.append(f"Environment mismatch: {env}")
        else:
            stack.pop()
    if stack:
        issues.append(f"Unclosed environments: {stack}")
    braces = []
    for m in re.finditer(r"[{}]", s):
        j = m.start()-1
        escapes = 0
        while j >= 0 and s[j] == "\\":
            escapes += 1
            j -= 1
        if escapes % 2:
            continue
        if m.group() == "{":
            braces.append(m.start())
        elif not braces:
            issues.append("Unmatched closing brace")
        else:
            braces.pop()
    if braces:
        issues.append("Unclosed braces")
    dollars = re.findall(r"(?<!\\)\$", s)
    if len(dollars) % 2:
        issues.append("Unbalanced inline math delimiters")
    labels = re.findall(r"\\label\{([^}]+)\}", s)
    counts = collections.Counter(labels)
    issues.extend("Duplicate label: "+k for k, n in counts.items() if n > 1)
    refs = [x.strip() for m in re.finditer(r"\\(?:ref|eqref|cref|Cref|pageref)\{([^}]+)\}", s)
            for x in m.group(1).split(",")]
    issues.extend("Undefined reference: "+k for k in sorted(set(refs)-set(labels)))
    biblio = (BUNDLE / "第三章_參考文獻_完整.bib").read_text()
    bibkeys = re.findall(r"@\w+\s*\{\s*([^,\s]+)", biblio)
    keys = set(bibkeys)
    issues.extend("Duplicate bibliography key: "+k
                  for k, n in collections.Counter(bibkeys).items() if n > 1)
    cites = [x.strip() for m in re.finditer(r"\\cite\{([^}]+)\}", s)
             for x in m.group(1).split(",")]
    issues.extend("Undefined citation: "+k for k in sorted(set(cites)-keys))
    issues.extend("Uncited chapter bibliography entry: "+k for k in sorted(keys-set(cites)))
    masterkeys = re.findall(r"@\w+\s*\{\s*([^,\s]+)",
                           (BASE / "overleaf/thesis_references.bib").read_text())
    issues.extend("Duplicate master bibliography key: "+k
                  for k, n in collections.Counter(masterkeys).items() if n > 1)
    issues.extend("Chapter citation missing in master: "+k
                  for k in sorted(set(cites)-set(masterkeys)))
    graphics = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\s*\{([^}]+)\}", s)
    for x in graphics:
        if not x.endswith(".png"):
            issues.append("Non-PNG figure: "+x)
        if not (BUNDLE / x).is_file():
            issues.append("Missing figure: "+x)
    removed = {
        "visual_time_alignment.png", "tcp_target_compensation.png",
        "omni_wheel_velocity_feasible_set.png", "grasp_state_transition.png",
        "command_execution_timeline.png",
    }
    issues.extend("Requested deletion still present: "+x for x in graphics if Path(x).name in removed)
    if "figsrc/reference_figures/" in s:
        issues.append("Original reference image still included")
    for phrase in ["抓持證據", "接觸證據", "理想連接", "理想固定"]:
        if phrase in s:
            issues.append("Out-of-scope wording: "+phrase)
    for m in re.finditer(r"\\begin\{figure\}(.*?)\\end\{figure\}", s, re.S):
        b = m.group(1)
        if b.find(r"\label") < b.find(r"\caption"):
            issues.append("Figure label precedes caption")
        if re.search(r"依文獻|自行繪製|重繪|改繪", b):
            issues.append("Drawing attribution wording remains in figure")
    alg = []
    for m in re.finditer(r"\\(If|EndIf|While|EndWhile|ElsIf)\b", s):
        c = m.group(1)
        if c in ["If", "While"]:
            alg.append(c)
        elif c == "ElsIf":
            if not alg or alg[-1] != "If":
                issues.append("ElsIf outside If")
        elif not alg or alg.pop() != {"EndIf": "If", "EndWhile": "While"}[c]:
            issues.append("Pseudocode block mismatch")
    if alg:
        issues.append("Unclosed pseudocode blocks")
    # Preserve labels for all pre-existing chapters/sections/subsections.
    headings = r"\\(?:chapter|section|subsection)\{([^}]+)\}\s*\\label\{([^}]+)\}"
    old_headings = dict(re.findall(headings, clean(original)))
    for title, label in old_headings.items():
        if label not in labels:
            issues.append("Original section label removed: "+label)
    result = {
        "original": stats(original),
        "condensed": stats(source),
        "label_count": len(labels),
        "reference_count": len(refs),
        "citation_keys": sorted(set(cites)),
        "chapter_bibliography_entries": len(bibkeys),
        "png_files": graphics,
        "checks": "passed" if not issues else "failed",
        "issues": issues,
        "xelatex_compilation": "not_run: no local TeX compiler installed",
        "page_count": "not_measured; depends on the user's actual Overleaf template",
    }
    (BUNDLE / "核對結果.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    old_labels = set(re.findall(r"\\label\{([^}]+)\}", clean(original)))
    removed_labels = sorted(old_labels-set(labels))
    with (BUNDLE / "刪除標籤清單.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["removed_label", "note"])
        for label in removed_labels:
            writer.writerow([label, "本章已移除其引用；其他章若引用此標籤，須改寫敘述或改引對應小節。"])
    assert not issues, issues
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return graphics


def package(graphics):
    # Only figures referenced by this replacement chapter enter the Overleaf ZIP.
    zip_path = BASE / "第三章_精簡重整版_Overleaf.zip"
    names = [
        "03_ResearchMethodology.tex", "第三章_新增參考文獻.bib",
        "第三章_參考文獻_完整.bib",
        "README_替換方式.md", "核對結果.json", "刪除標籤清單.csv",
        "新繪圖片核對.json", "圖片空白裁切紀錄.json", "SOURCES.md",
    ]
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for x in names+graphics:
            z.write(BUNDLE / x, x)
        for f in sorted((BUNDLE / "figsrc/ch03").glob("*.svg")):
            z.write(f, "圖源/"+f.name)
    print(f"ZIP: {zip_path} ({zip_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    package(audit())
