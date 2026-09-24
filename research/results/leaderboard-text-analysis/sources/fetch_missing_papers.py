from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import subprocess

root=Path('/data/hack_audit')
urls={
'linkvla':'https://arxiv.org/pdf/2603.01441',
'steervla':'https://arxiv.org/pdf/2602.08440',
'five_vla':'https://arxiv.org/pdf/2609.18623',
'rog_dagger':'https://arxiv.org/pdf/2608.24525',
'kyber_e2e':'https://arxiv.org/pdf/2405.01394',
'drivefuture':'https://arxiv.org/pdf/2605.09701',
'omnispace':'https://arxiv.org/pdf/2606.22617',
'lvldrive':'https://arxiv.org/pdf/2512.24331',
'ntr':'https://arxiv.org/pdf/2605.31116',
'poutine':'https://storage.googleapis.com/waymo-uploads/files/research/2025%20Technical%20Reports/2025%20WOD%20E2E%20Driving%20Challenge%20-%20Special%20Mention%20-%20Poutine.pdf'
}
def get(item):
 slug,url=item
 pdf=root/'papers'/f'{slug}.pdf'
 txt=root/'papers'/f'{slug}.txt'
 if not pdf.exists():
  p=subprocess.run(['curl','-fLsS','--retry','3','--max-time','100','-o',str(pdf),url],capture_output=True,text=True)
  if p.returncode or not pdf.exists() or pdf.read_bytes()[:4]!=b'%PDF':
   pdf.unlink(missing_ok=True)
   return slug,'download_failed',p.stderr.strip()[-300:]
 if not txt.exists():
  p=subprocess.run(['pdftotext','-layout',str(pdf),str(txt)],capture_output=True,text=True)
  if p.returncode:
   return slug,'conversion_failed',p.stderr.strip()[-300:]
 return slug,'ok',str(pdf.stat().st_size)
with ThreadPoolExecutor(max_workers=4) as pool:
 for result in as_completed([pool.submit(get,x) for x in urls.items()]):
  print(*result.result(),flush=True)
