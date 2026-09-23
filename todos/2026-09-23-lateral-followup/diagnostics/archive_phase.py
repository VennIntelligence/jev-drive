"""Inventory every closed phase directory; preserve raw evidence without rewriting it."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path.cwd())
    parser.add_argument('--raw-root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    source=args.repo/'scripts/b2d_controller_archive.py'
    spec=importlib.util.spec_from_file_location('phase_archive',str(source))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    roots=sorted(p for p in args.raw_root.iterdir() if p.is_dir())
    def inventory(root):
        output=args.out/(root.name+'.json')
        result=module.inventory(root,output)
        parsed=json.loads(output.read_text())
        entries=parsed['files']
        for row in entries:
            path=root/row['path']
            if module.digest(path)!=row['sha256']:
                raise RuntimeError('Post-inventory mismatch: '+str(path))
        return dict(root=str(root),inventory=str(output),files=len(entries),
                    bytes=sum(r['bytes'] for r in entries),
                    inventory_sha256=module.digest(output),verification='all_file_hashes_match')
    with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(inventory,roots))
    summary=dict(created_at=time.time(),raw_root=str(args.raw_root),roots=rows,
        total_files=sum(r['files'] for r in rows),total_bytes=sum(r['bytes'] for r in rows),
        counting='Readable inventory entries, including repeated source/fixture dependency links; not deduplicated storage.',
        helper_sha256=module.digest(source),script_sha256=module.digest(Path(__file__)))
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('roots',)},indent=2))


if __name__=='__main__':main()
