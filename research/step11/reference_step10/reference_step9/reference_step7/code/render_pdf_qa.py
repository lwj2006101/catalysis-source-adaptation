"""Render all current PDF pages and check text/glyph/page bounds for review."""
from pathlib import Path
import argparse,json,subprocess
import fitz
from PIL import Image,ImageOps,ImageDraw
ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--render-dir',type=Path,required=True);args=parser.parse_args();out=args.render_dir;out.mkdir(parents=True,exist_ok=True)
    files=[ROOT/'manuscripts'/n for n in ['manuscript_en.pdf','supplementary_en.pdf','第7步_结果与一区差距.pdf']]+sorted((ROOT/'figures').glob('step7_*.pdf'))
    inventory=[];page_images=[];violations=[]
    for i,path in enumerate(files):
        doc=fitz.open(path);pages=[]
        for page in doc:
            text=page.get_text();assert '\ufffd' not in text,('replacement_glyph',path.name,page.number)
            for block in page.get_text('dict')['blocks']:
                for line in block.get('lines',[]):
                    for span in line['spans']:
                        b=fitz.Rect(span['bbox'])*page.rotation_matrix;pr=page.rect
                        if b.x0 < -.5 or b.y0 < -.5 or b.x1 > pr.width+.5 or b.y1 > pr.height+.5:violations.append((path.name,page.number+1,span['text'],span['bbox']))
            prefix=out/f'pdf{i+1:02d}';pages.append(dict(page=page.number+1,characters=len(text),width=page.rect.width,height=page.rect.height))
        subprocess.run(['pdftoppm','-r','105','-png',str(path),str(prefix)],check=True,capture_output=True)
        for image in sorted(out.glob(prefix.name+'-*.png'),key=lambda p:int(p.stem.split('-')[-1])):page_images.append((path.name,int(image.stem.split('-')[-1]),image))
        inventory.append(dict(path=str(path.relative_to(ROOT)),pages=len(doc),page_inventory=pages))
    assert not violations,violations
    sheets=[]
    for start in range(0,len(page_images),6):
        chunk=page_images[start:start+6];sheet=Image.new('RGB',(1060,2360),'#e6eaf0');draw=ImageDraw.Draw(sheet)
        for index,(name,page,image) in enumerate(chunk):
            im=Image.open(image).convert('RGB');im.thumbnail((500,725));x=20+(index%2)*520;y=30+(index//2)*780
            draw.text((x,y-21),f'{name} | page {page}',fill='#172c3f');sheet.paste(im,(x,y))
        destination=out/f'sheet{start//6+1:02d}.png';sheet.save(destination);sheets.append(str(destination))
    (ROOT/'verification/pdf_page_inventory.json').write_text(json.dumps(dict(files=inventory,total_pages=len(page_images),contact_sheets=sheets,render_dpi=105,bounds_violations=violations,replacement_glyphs=0),ensure_ascii=False,indent=2))
    print(json.dumps(dict(pdf_count=len(files),pages=len(page_images),contact_sheets=sheets),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
