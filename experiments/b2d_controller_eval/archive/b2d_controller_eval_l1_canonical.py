"""Build a symlink-only L1 view, preserving every failed infrastructure attempt."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--sources',type=Path,nargs='+',required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    manifest={}
    for seed in ('p01','p02','p03','p04','p05'):
        for route in ('24816','25845','24252','26990','26944','25863','25928','3364'):
            for arm in 'ABCD':
                key=f'{seed}/{route}/{arm}'
                valid=[];failed=[]
                for source in a.sources:
                    candidate=source/seed/route/arm/('author_route' if arm=='A' else
                        'author_waypoint' if arm=='B' else 'pursuit')
                    path=candidate/'validation.json'
                    if not path.exists():continue
                    summary=json.loads(path.read_text())
                    if (summary['exception'] or summary['cleanup_errors'] or summary['telemetry_parse_errors']
                            or not summary['gates']['telemetry_complete']):
                        failed.append(str(candidate));continue
                    valid.append(candidate)
                if len(valid)!=1:
                    raise ValueError(f'{key}: expected exactly one valid attempt, got {valid}; failures {failed}')
                link=a.out/key/valid[0].name
                link.parent.mkdir(parents=True,exist_ok=True)
                if link.is_symlink():
                    if link.resolve()!=valid[0].resolve():raise ValueError(f'Existing link changed: {link}')
                elif link.exists():
                    raise ValueError(f'Existing nonsymlink: {link}')
                else:
                    link.symlink_to(valid[0].resolve(),target_is_directory=True)
                manifest[key]={'selected':str(valid[0]),'failed_infrastructure':failed}
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Linked {len(manifest)} valid L1 cases in {a.out}')


if __name__=='__main__':main()
