"""Render every current PDF page and check rotated text/image bounds."""
from pathlib import Path
import hashlib
import json
import fitz
from PIL import Image, ImageOps, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
REVIEW=Path('/workspace/scratch/8bea8356e6d0/tmp/step10_pdf_review')
PDFS=[ROOT/'manuscripts/manuscript_en.pdf',ROOT/'manuscripts/supplementary_en.pdf',
      ROOT/'manuscripts/第10步_真实结果与一区差距.pdf']+sorted((ROOT/'figures').glob('step10_*.pdf'))


def main():
    REVIEW.mkdir(parents=True,exist_ok=True)
    result={};contacts=[]
    for file in PDFS:
        name=file.stem;out=REVIEW/name;out.mkdir(exist_ok=True)
        document=fitz.open(file);bad=[];image_bad=[];pages=[]
        for i,page in enumerate(document):
            for block in page.get_text('dict')['blocks']:
                for line in block.get('lines',[]):
                    for span in line.get('spans',[]):
                        box=fitz.Rect(span['bbox'])*page.rotation_matrix
                        allowed=fitz.Rect(page.rect.x0-2,page.rect.y0-2,page.rect.x1+2,page.rect.y1+2)
                        if not allowed.contains(box):
                            bad.append(dict(page=i+1,text=span['text'][:100],bounds=list(box)))
            for occurrence in page.get_image_info():
                box=fitz.Rect(occurrence['bbox'])*page.rotation_matrix
                if not (page.rect+(-2,-2,2,2)).contains(box):
                    image_bad.append(dict(page=i+1,bounds=list(box)))
            path=out/f'page_{i+1:03d}.png'
            page.get_pixmap(matrix=fitz.Matrix(1.4,1.4),alpha=False).save(path)
            pages.append(path)
        result[str(file.relative_to(ROOT))]=dict(pages=len(document),sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
           every_page_rendered=True,rendered_pages=list(range(1,len(document)+1)),out_of_page_spans=bad,
           out_of_page_images=image_bad,render_directory=str(out))
        for start in range(0,len(pages),12):
            width,height=400,570;rows=(len(pages[start:start+12])+3)//4
            sheet=Image.new('RGB',(width*4,height*rows+45),'#dae1e5');draw=ImageDraw.Draw(sheet)
            draw.text((12,12),f'{name} | pages {start+1}-{min(start+12,len(pages))}',fill='#172b3a')
            for j,path in enumerate(pages[start:start+12]):
                img=Image.open(path);img.thumbnail((width-14,height-27))
                x=(j%4)*width+(width-img.width)//2;y=45+(j//4)*height+18
                sheet.paste(img,(x,y));draw.text(((j%4)*width+10,y-15),str(start+j+1),fill='black')
            path=REVIEW/f'{name}_contact_{start//12+1}.png';sheet.save(path);contacts.append(str(path))
    (ROOT/'reports/step10_pdf_layout_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (REVIEW/'contacts.json').write_text(json.dumps(contacts,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(pdf_count=len(PDFS),pages=sum(r['pages'] for r in result.values()),
          out_of_page_spans=sum(len(r['out_of_page_spans']) for r in result.values()),
          out_of_page_images=sum(len(r['out_of_page_images']) for r in result.values()),contacts=contacts),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
