"""Source, asset and provenance checks for the current provided manuscript fragments."""
from pathlib import Path
from collections import Counter
import json, re, subprocess, zipfile

ROOT=Path(__file__).resolve().parent
PARTS=[ROOT/n for n in ['第一章_前言與動機_含圖.tex','01_03_05_research_comparison.tex',
    '第一章_本文架構.tex','02_platform_and_interfaces.tex','02_software_and_integration.tex',
    '02_workflow_and_recording.tex']]+sorted(ROOT.glob('03_0[1-7]_*.tex'))

def uncomment(s):
    return '\n'.join(re.split(r'(?<!\\)%',ln,maxsplit=1)[0] for ln in s.splitlines())

def check_braces(s):
    n=0
    for i,c in enumerate(s):
        if i and s[i-1]=='\\':continue
        n+=(c=='{')-(c=='}')
        assert n>=0
    assert n==0,n

def main():
    for p in PARTS+[ROOT/'algorithm_preamble.tex']:check_braces(uncomment(p.read_text()))
    active=uncomment('\n'.join(p.read_text() for p in PARTS))
    bib=uncomment((ROOT/'thesis_references.bib').read_text());check_braces(bib)
    keys=re.findall(r'^@\w+\s*\{([^,]+),',bib,re.M)
    assert len(keys)==len(set(keys))
    labels=re.findall(r'\\label\{([^}]+)\}',active)
    assert not [k for k,v in Counter(labels).items() if v>1]
    refs=[k.strip() for g in re.findall(r'\\(?:[cC]ref|eqref|ref|pageref)\*?\{([^}]+)\}',active) for k in g.split(',')]
    cites=[k.strip() for g in re.findall(r'\\cite\w*\*?(?:\[[^\]]*\])*\{([^}]+)\}',active) for k in g.split(',')]
    assert not set(refs)-set(labels),sorted(set(refs)-set(labels))
    assert not set(cites)-set(keys),sorted(set(cites)-set(keys))
    images=re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',active)
    assert all((ROOT/p).is_file() for p in images)
    stack=[]
    for action,env in re.findall(r'\\(begin|end)\{([^}]+)\}',active):
        if action=='begin':stack.append(env)
        else:assert stack and stack.pop()==env,env
    assert not stack
    ch3='\n'.join(p.read_text() for p in sorted(ROOT.glob('03_0[1-7]_*.tex')))
    assert ch3.count('\\begin{figure}')==22
    assert '\\label{alg:wholebody_predictive_control}' in ch3
    assert '\\label{eq:ch3_joint_space_unfold_reference}' in ch3
    assert 'w_{\\mathrm{ext}}=0' in ch3

    added=json.loads((ROOT/'figsrc/chapter_additions_manifest.json').read_text())
    added.append(dict(chapter='ch02',file='mask_rcnn_class_examples'))
    assets=[]
    for item in added:
        p=ROOT/'figsrc'/item['chapter']/(item['file']+'.pdf')
        info=subprocess.check_output(['pdfinfo',str(p)],text=True)
        assert re.search(r'^Pages:\s+1\s*$',info,re.M)
        fonts=subprocess.check_output(['pdffonts',str(p)],text=True).splitlines()[2:]
        assert fonts and all(re.search(r'\s+yes\s+(?:yes|no)\s+(?:yes|no)\s+\d+\s+\d+\s*$',ln) for ln in fonts)
        assert any('DFKai' in ln for ln in fonts)
        rasters=subprocess.check_output(['pdfimages','-list',str(p)],text=True).splitlines()[2:]
        is_raster=item['file']=='mask_rcnn_class_examples'
        assert bool(rasters)==is_raster,(p,rasters)
        assets.append(dict(file=str(p.relative_to(ROOT)),embedded_fonts=True,vector_only=not is_raster))

    data=json.loads((ROOT/'figsrc/ch02/run97_stage_times.json').read_text())
    assert abs(sum(s['duration_s'] for s in data['stages'])-387.7)<1e-6
    assert data['source'].startswith('evaluation/runs/m8house_d2_unified_97/')
    examples=json.loads((ROOT/'figsrc/ch02/mask_examples_evidence.json').read_text())
    assert len(examples['examples'])==6 and all(e['run'] in [85,86,87] for e in examples['examples'])
    profile=json.loads((ROOT/'figsrc/ch03/approach_profile_evidence.json').read_text())
    assert profile['run']=='m8house_d2_unified_101' and profile['branch_event']['what']=='pick_jblend_start'

    outputs=[ROOT/'第三章_3.1至3.4完整稿.tex',ROOT/'第三章_3.4至3.7完整稿.tex',
             ROOT/'chapters_outline.tex',ROOT.parent/'論文章節架構.md']
    before=[p.read_bytes() for p in outputs]
    subprocess.run(['python3',str(ROOT/'build_manuscript_materials.py')],check=True,stdout=subprocess.DEVNULL)
    assert before==[p.read_bytes() for p in outputs], 'generated drafts were stale'
    outline=(ROOT/'chapters_outline.tex').read_text()
    expected='\n'.join(re.findall(r'^\\(?:chapter|section|subsection)\{[^\n]+\}',ch3,re.M))
    actual=outline[outline.index('\\chapter{研究方法與全身模型預測控制器設計}'):outline.index('\\chapter{實驗方法與結果分析}')]
    assert '\n'.join(x for x in actual.splitlines() if x.strip())==expected
    assert '\\input{algorithm_preamble}' in (ROOT/'preamble.tex').read_text()
    assert '\\subsection{現有研究比較與研究缺口}' in outline

    handoff=ROOT/'第三章_現行22張圖_插入片段.tex'
    if handoff.is_file():
        s=uncomment(handoff.read_text())
        fig_labels=re.findall(r'\\label\{([^}]+)\}',s)
        assert len(fig_labels)==len(set(fig_labels))==22
        assert s.count('\\begin{figure}')==22
        assert set(fig_labels).issubset(labels)
        figrefs=[k.strip() for g in re.findall(r'\\(?:[cC]ref|eqref|ref)\{([^}]+)\}',s) for k in g.split(',')]
        assert set(figrefs).issubset(labels)
        guide=(ROOT/'Overleaf_從第三章插圖開始_修改清單.md').read_text()
        assert all(ord(c)>=32 or c in '\n\t' for c in guide)
        assert '\\input{algorithm_preamble}' in guide
        assert '\\label{eq:ch3_condensed_augmented_prediction}' in guide
        for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)',guide):
            assert (ROOT/target).is_file(),target

    report=dict(date='2026-11-24',source_files=[p.name for p in PARTS],
        bibliography_entries=len(keys),unique_labels=len(labels),missing_citations=[],missing_references=[],
        graphics_paths=len(images),all_graphics_exist=True,chapter3_figure_count=22,
        new_figure_count=len(assets),new_assets=assets,
        chapter3_headings_match=True,generated_drafts_current=True,
        joint_space_revision_promoted=True,joint_space_validation='run101 cylinder stage only; full mission failed',
        extension_term_enabled_in_existing_runs=False,
        run97_quality_limitation='residual drawer opening 11.1 mm',
        xelatex_compile=False,bibtex_compile=False,
        omitted_content=['第一章 1.3.1--1.3.4 完整 LaTeX','第二章 2.5 完整 LaTeX','第四、第五章完整 LaTeX'],
        scope='文件與圖表修訂、固定模型離線示例推論；未啟動模擬或訓練，未修改控制程式。')
    (ROOT/'20261124_修訂核對紀錄.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['source_files','bibliography_entries','unique_labels',
        'graphics_paths','chapter3_figure_count','new_figure_count','missing_citations','missing_references']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
