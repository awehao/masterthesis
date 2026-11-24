"""Extract cited figures from original papers, preserving the PDF artwork.

Coordinates are PDF points measured from the top left. Downloaded source
papers remain in originals/ for provenance; distributable assets are only
the selected figures, without surrounding article text or captions.
"""
from pathlib import Path
import hashlib
import json
import re
import subprocess

HERE = Path(__file__).resolve().parent
FIGURES = [
    dict(file='mask_rcnn_framework_he2017', source_file='mask_rcnn_iccv2017.pdf',
         url='https://openaccess.thecvf.com/content_ICCV_2017/papers/He_Mask_R-CNN_ICCV_2017_paper.pdf',
         source_figure='Figure 1', pdf_page=1, version='CVF-hosted ICCV 2017 paper',
         crop_top_left_pt=[306,209,550,322], cite='he2017maskrcnn',
         placement='3.5.1 模型與訓練資料介紹之後、遮罩深度點選取之前',
         caption='Mask R-CNN 實例分割方法之原理示意',
         scope='原方法流程概念；本文實作採 ResNet-50-FPN v2，並非逐層照搬原圖網路。'),
    dict(file='rotation_representation_tang2024', source_file='gmpc_arxiv2403_07317v1.pdf',
         url='https://arxiv.org/pdf/2403.07317v1',
         source_figure='Fig. 1', pdf_page=1, version='arXiv:2403.07317v1; bibliography cites the RA-L journal version',
         crop_top_left_pt=[308,171,571,290], cite='tang2024gmpc',
         placement='3.1.1 底盤位姿表示及李群定義之後',
         caption='旋轉群與角度參數表示之比較',
         scope='只說明旋轉表示；不使用原論文差速底盤或實驗結果代表本研究全向底盤。'),
    dict(file='kalman_predict_update_welch2004', source_file='welch_bishop_kalman_intro.pdf',
         url='https://www.cs.utexas.edu/~pstone/Courses/393Rfall15/readings/Welch%2BBishop-TR-95.pdf',
         source_figure='Figure 1-2', pdf_page=6, version='TR 95-041, revised March 1, 2004',
         crop_top_left_pt=[110,320,498,540], cite='welch2004kalman',
         placement='3.4.1 卡爾曼等速度狀態模型及符號說明之後',
         caption='卡爾曼濾波之預測與量測更新流程',
         scope='一般線性濾波循環；本研究障礙模型無外部控制輸入，圖中 B u 項取零。'),
    dict(file='nav2_architecture_macenski2020', source_file='nav2_arxiv2003_00368v2.pdf',
         url='https://arxiv.org/pdf/2003.00368v2',
         source_figure='Fig. 2', pdf_page=3, version='arXiv:2003.00368v2; bibliography cites IROS 2020',
         crop_top_left_pt=[48,277,299,440], cite='macenski2020nav2',
         placement='2.4.1 Nav2 及定位來源介紹之後',
         caption='Navigation2 既有系統之模組分工',
         scope='既有架構背景；本文使用全域規劃與代價地圖，操作及局部控制由本文方法處理。'),
]


def main():
    manifest=[]
    for item in FIGURES:
        src=HERE/'originals'/item['source_file']
        info=subprocess.check_output(['pdfinfo',str(src)],text=True)
        height=float(re.search(r'Page size:\s+[\d.]+ x ([\d.]+)',info)[1])
        x0,y0,x1,y1=item['crop_top_left_pt']
        width,cropheight=x1-x0,y1-y0
        out=HERE/(item['file']+'.pdf')
        subprocess.run(['gs','-q','-dSAFER','-dBATCH','-dNOPAUSE',
            '-sDEVICE=pdfwrite','-dAutoRotatePages=/None',
            '-dFirstPage='+str(item['pdf_page']),'-dLastPage='+str(item['pdf_page']),
            '-dFIXEDMEDIA','-dDEVICEWIDTHPOINTS='+str(width),
            '-dDEVICEHEIGHTPOINTS='+str(cropheight),'-sOutputFile='+str(out),
            '-c',f'<</PageOffset [{-x0} {-(height-y1)}]>> setpagedevice',
            '-f',str(src)],check=True)
        subprocess.run(['pdftoppm','-r','220','-png','-singlefile',str(out),str(HERE/item['file'])],check=True)
        manifest.append(dict(**item, source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
            file_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
            treatment='原圖局部擷取；未改寫標籤、重繪、改色或更改幾何。',
            accessed='2026-11-24',original_caption_excluded=True))
    (HERE/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print('Extracted',len(manifest),'reference figures.')


if __name__=='__main__':main()
