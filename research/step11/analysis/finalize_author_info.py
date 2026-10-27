"""Fill genuine author metadata and build a separately named submission PDF."""
from pathlib import Path
import json,re,subprocess
ROOT=Path(__file__).resolve().parents[1]


def tex(s):
    translations={'&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}',
                  '~':r'\textasciitilde{}','^':r'\textasciicircum{}'}
    if '\\' in s:raise ValueError('Use plain text rather than LaTeX in author metadata')
    return ''.join(translations.get(c,c) for c in s)


def main():
    f=json.loads((ROOT/'submission/author_metadata.json').read_text())
    required=['corresponding_email','contributions','conflicts_of_interest','funding']
    if not f['authors'] or any(not f[k].strip() for k in required):
        raise ValueError('Fill actual authors, email, contributions, conflict statement and funding in author_metadata.json')
    if not f['all_authors_approved'] or not f['originality_and_exclusive_submission_confirmed']:
        raise ValueError('The submitting authors need to confirm approval and exclusive submission in their metadata')
    for author in f['authors']:
        if not author.get('name') or not author.get('affiliation'):raise ValueError('Every author needs an English name and affiliation')
    names=', '.join(tex(x['name']) for x in f['authors'])
    affiliations='\\\\'.join(tex(x['name'])+': '+tex(x['affiliation']) for x in f['authors'])
    block='\\author{'+names+'\\\\{\\small '+affiliations+'}\\\\{\\small Correspondence: '+tex(f['corresponding_email'])+'}}'
    text=(ROOT/'manuscripts/manuscript_en.tex').read_text().replace('\\author{}',block)
    start=text.index('\\section*{Author information and declarations}')
    end=text.index('\\section*{Data availability}',start)
    declarations=''
    for title,key in [('Author contributions','contributions'),('Conflicts of interest','conflicts_of_interest'),
                      ('Funding','funding'),('AI assistance','ai_assistance')]:
        declarations+='\\section*{'+title+'}\n'+tex(f[key])+'\n\n'
    if f['acknowledgements'].strip():declarations+='\\section*{Acknowledgements}\n'+tex(f['acknowledgements'])+'\n\n'
    text=text[:start]+declarations+text[end:]
    if f['repository_url'].strip():
        url=f['repository_url'].strip()
        if not url.startswith('https://'):raise ValueError('Repository URL must be a real HTTPS deposit link')
        text=text.replace('A persistent repository record and identifier are to be supplied by the authors when arranging the journal deposit; no repository DOI is asserted in this manuscript.',
                          'The computational archive is deposited at \\url{'+url+'}.')
    p=ROOT/'manuscripts/manuscript_submission.tex';p.write_text(text)
    si=(ROOT/'manuscripts/supplementary_en.tex').read_text().replace('\\author{}',block)
    s=ROOT/'manuscripts/supplementary_submission.tex';s.write_text(si)
    for target in [p,s]:
        for _ in range(2):subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error',target.name],cwd=target.parent,check=True)
        print('Built',target.with_suffix('.pdf'))
    letter=(ROOT/'submission/cover_letter.txt').read_text()
    letter=letter.replace('Author identities and disclosures will accompany the final submission record.',
                          'The submitting authors have confirmed approval of the manuscript and exclusive submission. Author information and disclosures are included in the manuscript.')
    letter=letter.replace('\nThe submitting authors\n','\n'+'\n'.join(x['name']+'; '+x['affiliation'] for x in f['authors'])+'\nCorrespondence: '+f['corresponding_email']+'\n')
    q=ROOT/'submission/cover_letter_submission.txt';q.write_text(letter)
    from build_submission import pdf,p as paragraph,ENT,EH
    pdf(q.with_suffix('.pdf'),[paragraph('Cover letter',EH)]+[paragraph(x,ENT) for x in letter.strip().split('\n\n')],
        'Cover letter: What few target labels learn under publication shift in catalysis')
    print('Built',q.with_suffix('.pdf'))


if __name__=='__main__':main()
