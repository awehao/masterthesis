"""Regenerate copyable drafts and the outline from current section sources.

Run from any directory with Python 3. Only current section sources are assembled;
historical fragments and pending implementation proposals are excluded.
"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent


def read(name):
    return (ROOT / name).read_text()


def headings(text):
    return '\n'.join(re.findall(r'^\\(?:chapter|section|subsection)\{[^\n]+\}', text, re.M))


def manual_section(title, subs):
    return '\\section{' + title + '}\n' + '\n'.join('\\subsection{' + x + '}' for x in subs)


def main():
    sections = sorted(ROOT.glob('03_0[1-7]_*.tex'))
    assert len(sections) == 7
    for filename, nums in [('第三章_3.1至3.4完整稿.tex', range(1, 5)),
                           ('第三章_3.4至3.7完整稿.tex', range(4, 8))]:
        sources = [p for p in sections if int(p.name[3:5]) in nums]
        header = '% !TeX program = xelatex\n% 由 build_manuscript_materials.py 自動整併；修改請回各分節來源。\n'
        header += '% 來源：' + '、'.join(p.name for p in sources) + '\n'
        header += '% 3.5.4 採第 101 趟已執行之圓柱展開方法；整趟任務未通過。\n\n'
        (ROOT / filename).write_text(header + '\n\n'.join(p.read_text() for p in sources))

    old = read('chapters_outline.tex')
    c4 = old[old.index('\\chapter{實驗方法與結果分析}'):old.index('\\chapter{結論與未來工作}')]
    c4 = re.sub(r'\\section\{住宅場景整合任務\}.*?(?=\\section\{計算效能)',
        lambda m: manual_section('住宅場景整合任務', [
            '視覺導引之開門與穿門', '茶几圓柱之移動中抓取與車尾收納',
            '開抽屜取筆、收納與關閉抽屜', '回程推門通行、關門與返回起點',
            '任務銜接與整體完成情形']) + '\n\n', c4, flags=re.S)
    c5 = old[old.index('\\chapter{結論與未來工作}'):]
    c5 = c5.split('% 參考文獻')[0].rstrip()
    literature = manual_section('文獻回顧', [
        '預測導航與動態避障', '全身模型預測控制與移動中操作',
        '視覺導引與關節式物件操作', '執行動態建模與追蹤補償', '現有研究比較與研究缺口'])
    problem = manual_section('問題定義與基本假設', [
        '研究問題與控制目標', '環境、感測與已知物件資訊', '執行介面與操作模型假設'])
    platform = headings(read('02_platform_and_interfaces.tex'))
    outline = '\n\n'.join([
        '% !TeX root = main.tex\n% 章節骨架；有正文的章節名稱由來源自動同步。\n'
        '% 第一章 1.3.1--1.3.4、第二章 2.5、第四與第五章為規劃標題。\n'
        '% 載入正式正文時不要同時載入本骨架。\n% 產生器：build_manuscript_materials.py。',
        headings(read('第一章_前言與動機_含圖.tex')), literature,
        headings(read('第一章_本文架構.tex')), '\\chapter{研究系統介紹}',
        platform, headings(read('02_software_and_integration.tex')), problem,
        headings(read('02_workflow_and_recording.tex')),
        '\n\n'.join(headings(p.read_text()) for p in sections), c4.rstrip(), c5,
        '% 參考文獻統一使用 thesis_references.bib；附錄依需要加入。', ''])
    outline = outline.rstrip() + '\n'
    (ROOT / 'chapters_outline.tex').write_text(outline)

    plan = ROOT.parent / '論文章節架構.md'
    text = plan.read_text()
    lines, chap, sec, sub = [], 0, 0, 0
    numerals = ['一', '二', '三', '四', '五']
    for kind, title in re.findall(r'^\\(chapter|section|subsection)\{([^\n]+)\}', outline, re.M):
        if kind == 'chapter':
            chap += 1; sec = sub = 0
            lines.append(f'## 第{numerals[chap-1]}章　{title}')
        elif kind == 'section':
            sec += 1; sub = 0
            lines.append(f'### {chap}.{sec} {title}')
        else:
            sub += 1
            lines.append(f'#### {chap}.{sec}.{sub} {title}')
    start = text.index('## 第一章')
    end = text.index('## 論文前後置內容')
    text = text[:start] + '\n\n'.join(lines) + '\n\n' + text[end:]
    text = text.replace('架構版本：v0.6（2026-11-23，更新第三章 3.4～3.7）。',
        '架構版本：v0.7（2026-11-24，與目前 LaTeX 正文標題及住宅任務同步）。')
    text = text.replace('住宅整合任務以開門、穿門、茶几取物及持物返回為主線；抽屜保留於 4.4 的協同控制比較。',
        '住宅任務為開門通行、茶几圓柱移動中抓取與車尾收納、開抽屜取筆與關抽屜、回程推門通行與關門、返回起點；獨立抽屜比較另列於 4.4。')
    text = text.replace('現有分章 Markdown 初稿尚未同步重編。', '正文以 overleaf 內分節 LaTeX 為準，舊 Markdown 內容已留存歷史備份。')
    text = text.replace('- 本次同步第三章 3.4～3.7 的正文與章節骨架，未更動摘要、LaTeX 主檔或既有 ZIP 上傳包。',
        '- 摘要、2.3–2.4、2.6–2.7 與第三章分節檔為目前撰寫來源；合併稿由產生器更新。3.5.4 新展開法另存待驗證修訂，未併入正式稿。')
    text = text.replace('全身幾何模型預測控制方法 |', '研究方法與全身模型預測控制器設計 |')
    text = text.replace('3.5.4 新展開法另存待驗證修訂，未併入正式稿。', '3.5.4 新展開法已有第 101 趟圓柱階段證據並併入正文；整趟仍未通過。')
    plan.write_text(text)
    print('Updated 2 combined drafts and chapter outlines.')


if __name__ == '__main__':
    main()
