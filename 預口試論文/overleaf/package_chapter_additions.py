"""Build review/upload bundles from current sources after validation."""
from pathlib import Path
import json,re,zipfile
from PIL import Image,ImageDraw,ImageFont
from validate_chapter_additions import PARTS,uncomment,main as validate
from build_overleaf_handoff import main as make_handoff

ROOT=Path(__file__).resolve().parent

def package(path,files,note):
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(set(files)):
            assert p.is_file(),p
            z.write(p,str(p.relative_to(ROOT)))
        z.writestr('使用順序.txt',note)
    with zipfile.ZipFile(path) as z:assert z.testzip() is None
    print(path.name,path.stat().st_size)

def main():
    make_handoff()
    validate()
    manifest=json.loads((ROOT/'figsrc/chapter_additions_manifest.json').read_text())
    manifest.insert(2,dict(chapter='ch02',file='mask_rcnn_class_examples'))
    tiles=Image.new('RGB',(2000,2600),'white');d=ImageDraw.Draw(tiles)
    font=ImageFont.truetype(str(ROOT/'fonts/kaiu.ttf'),30)
    titles=['2.1.2 整機座標','2.2.3 命令鏈','2.4.2 六類預測遮罩','2.6 任務狀態機',
            '2.6 第 97 趟甘特圖','3.3.2 輪級速度可行域','3.5.1 視覺目標流程','3.5.4 距離剖面設計']
    for i,(item,title) in enumerate(zip(manifest,titles)):
        im=Image.open(ROOT/'figsrc'/item['chapter']/(item['file']+'.png')).convert('RGB')
        im.thumbnail((980,590));x=(i%2)*1000+(1000-im.width)//2;y=(i//2)*650+45
        tiles.paste(im,(x,y));d.text(((i%2)*1000+20,(i//2)*650+8),title,font=font,fill='black')
    preview=ROOT/'figsrc/新增圖表預覽.png';tiles.save(preview)
    images=re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',uncomment('\n'.join(p.read_text() for p in PARTS)))
    shared=PARTS+[ROOT/n for n in ['thesis_references.bib','algorithm_preamble.tex',
        '第一至三章_新增內容與插圖位置.md','Overleaf_從第三章插圖開始_修改清單.md',
        '第三章_現行22張圖_插入片段.tex','build_overleaf_handoff.py','overleaf_handoff_manifest.json',
        'abstract_zh.tex','abstract_en.tex','20261124_修訂核對紀錄.json','README.md',
        'validate_chapter_additions.py','package_chapter_additions.py','build_manuscript_materials.py',
        '第三章_3.1至3.4完整稿.tex','第三章_3.4至3.7完整稿.tex','chapters_outline.tex']]
    shared+=[ROOT/p for p in images]
    preserved=[ROOT/n for n in ['第二章_架構圖插入.tex','第二章_模擬環境地圖插入.tex','第二章_新增照片插入.tex']]
    shared+=preserved
    shared+=[ROOT/p for src in preserved for p in re.findall(
        r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',uncomment(src.read_text()))]
    shared+=list((ROOT/'figsrc').rglob('SOURCES*.md'))
    shared+=[preview,ROOT/'figsrc/build_chapter_additions.py',ROOT/'figsrc/make_mask_examples.py',
        ROOT/'figsrc/chapter_additions_manifest.json',ROOT/'figsrc/ch02/mask_examples_evidence.json',
        ROOT/'figsrc/ch02/run97_stage_times.json',ROOT/'figsrc/ch03/approach_profile_evidence.json']
    shared+=[ROOT/'figsrc'/i['chapter']/(i['file']+ext) for i in manifest for ext in ('.png',)]
    shared+=[ROOT/'figsrc'/i['chapter']/(i['file']+'.svg') for i in manifest if i['file']!='mask_rcnn_class_examples']
    # The original drawing helpers are needed to reproduce the new vector drawings.
    shared+=[ROOT/'figsrc/ch03'/n for n in ['method_diagrams.py','method_scene_geometry.py']]
    note='''先看 Overleaf_從第三章插圖開始_修改清單.md。
第三章_現行22張圖_插入片段.tex 含全部現行圖的複製格式與位置；整節替換後不重貼。
更新學姐專案的對應小節並上傳 figsrc，不要同時載入正文與目錄骨架。
第一章提供 1.1–1.2、1.3.5、1.4；未包含 1.3.1–1.3.4 全文。
第二章 2.5 全文亦未收錄。這不是完整五章論文。
演算法框需在導言區載入一次 algorithm_preamble.tex；詳見說明。
第三章合併稿在 3.4 重疊，不同時整份載入。
第 97 趟為流程完成但抽屜殘留 11.1 mm；第 101 趟僅圓柱階段完成，整趟未通過。
接近剖面為設計曲線；逐類 IoU 表為驗證集，測試摘要 0.754 另述。
本機未提供 XeLaTeX/BibTeX，整章排版需在 Overleaf 檢查。
重製腳本需完整研究儲存庫中的模型、權重與實驗資料；排版只需 PDF。
'''
    package(ROOT.parent/'第一至三章_新增圖表與Overleaf內容.zip',shared,note)

    full=shared+list(ROOT.glob('*.tex'))+[ROOT/n for n in ['preamble.tex','nthuthesis.cls',
        'IEEEtran.bst','latexmkrc','20261124_內容修訂說明.md','fonts/kaiu.ttf','fonts/times.ttf']]
    full+=list((ROOT/'figsrc/reference_figures').glob('*.json'))
    full+=list((ROOT/'figsrc/reference_figures').glob('*.py'))
    full+=list((ROOT/'figsrc/reference_figures').glob('*.png'))
    package(ROOT.parent/'20261124_計畫書修訂稿.zip',full,note+'main.tex 仍預設只編譯摘要。\n')
    abstracts=[ROOT/n for n in ['main.tex','thesis.tex','preamble.tex','algorithm_preamble.tex',
        'nthuvars.tex','nthuthesis.cls','00_abstract.tex','abstract_zh.tex','abstract_en.tex',
        'latexmkrc','fonts/kaiu.ttf','fonts/times.ttf']]
    package(ROOT.parent/'thesis_abstract_overleaf.zip',abstracts,
        '摘要專案 v0.8。選 XeLaTeX，主檔 main.tex；目前僅編譯中英文摘要。\n')

if __name__=='__main__':main()
