"""Render every final PDF page and make contact sheets for visual inspection."""
from pathlib import Path
import json,hashlib,math
import fitz
from PIL import Image,ImageDraw,ImageOps
ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'tmp/pdf_qa';out.mkdir(parents=True,exist_ok=True)
    records=[]
    paths=['manuscripts/manuscript_en.pdf','manuscripts/supplementary_en.pdf',
           'manuscripts/第11步_成果与投稿说明.pdf','submission/cover_letter.pdf',
           'submission/graphical_abstract.pdf']
    names=['main','si','progress','cover','graphical']
    for path,name in zip(paths,names):
        doc=fitz.open(ROOT/path);rendered=[];outside=[]
        for i,page in enumerate(doc):
            target=out/f'{name}_{i+1:02d}.png'
            page.get_pixmap(matrix=fitz.Matrix(1.7,1.7),alpha=False).save(target)
            rendered.append(target)
            for block in page.get_text('dict')['blocks']:
                if block.get('type')!=0:continue
                b=fitz.Rect(block['bbox'])*page.rotation_matrix
                if b.x0 < -1 or b.y0 < -1 or b.x1 > page.rect.width+1 or b.y1 > page.rect.height+1:
                    outside.append(dict(page=i+1,bbox=list(b)))
        for begin in range(0,len(rendered),9):
            images=rendered[begin:begin+9]
            width,height=430,650
            sheet=Image.new('RGB',(width*3,height*math.ceil(len(images)/3)),(225,230,235))
            drawer=ImageDraw.Draw(sheet)
            for j,img in enumerate(images):
                pageimg=Image.open(img).convert('RGB')
                thumb=ImageOps.contain(pageimg,(width-18,height-40))
                x=(j%3)*width+(width-thumb.width)//2;y=(j//3)*height+30
                sheet.paste(thumb,(x,y))
                drawer.text(((j%3)*width+10,(j//3)*height+8),f'{name} page {begin+j+1}',fill='#223344')
            sheet.save(out/f'{name}_contact_{begin//9+1:02d}.png')
        records.append(dict(path=path,pages=len(doc),sha256=hashlib.sha256((ROOT/path).read_bytes()).hexdigest(),
                            all_pages_rendered=True,out_of_page_text_blocks=outside))
        print(name,len(doc),'outside blocks:',len(outside))
    report=dict(passed=False,documents=records,all_pages_rendered=True,
                all_pages_rendered_and_visually_reviewed=False,
                visual_review_status='pending human-readable inspection of rendered images',
                rendering_directory='tmp/pdf_qa')
    (ROOT/'reports/pdf_quality_assurance.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')


if __name__=='__main__':main()
