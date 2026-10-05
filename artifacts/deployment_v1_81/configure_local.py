"""Create ignored local production config from the user's existing CSV; never log secrets."""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[2]
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--credentials',type=Path,help='Existing two-column local CSV (apiKey/openAiCompatible/dashScope).')
    args=parser.parse_args()
    target=PROJECT/'.env'
    if target.exists():
        print(json.dumps({'created':False,'existing_env_preserved':True,'env_sha256':hashlib.sha256(target.read_bytes()).hexdigest()}))
        return
    candidates=[args.credentials] if args.credentials else list(PROJECT.parents[1].glob('*.csv'))
    matches=[]
    for candidate in candidates:
        try:
            values=dict(csv.reader(candidate.read_text(encoding='utf-8-sig').splitlines()))
        except (OSError,ValueError):continue
        if all(isinstance(values.get(k),str) and values[k].strip() for k in ('apiKey','openAiCompatible','dashScope')):
            matches.append((candidate,values))
    if len(matches)!=1:
        raise SystemExit('Provide --credentials with exactly one existing local credentials CSV. No key values are printed.')
    _,values=matches[0]
    for k in ('apiKey','openAiCompatible','dashScope'):
        if any(c in values[k] for c in ('\r','\n','\x00','$')):
            raise SystemExit('Credentials contain unsupported control/interpolation characters; existing files remain unchanged.')
    if not all(values[k].startswith('https://') for k in ('openAiCompatible','dashScope')):
        raise SystemExit('Configured model endpoints must use HTTPS.')
    result=subprocess.run([sys.executable,str(PROJECT/'scripts/setup_env.py'),'--mode','production'],
        cwd=PROJECT,capture_output=True,text=True)
    if result.returncode:
        raise SystemExit('Local setup failed; secrets not printed. Preserve any created config and inspect locally.')
    replacements={'MODEL_PROVIDER':'dashscope','DASHSCOPE_API_KEY':values['apiKey'],
        'DASHSCOPE_CHAT_BASE_URL':values['openAiCompatible'],'DASHSCOPE_HTTP_BASE_URL':values['dashScope']}
    lines=[]
    remaining=dict(replacements)
    for line in target.read_text(encoding='utf-8').splitlines():
        key=line.split('=',1)[0]
        if key in replacements:
            line=key+'='+json.dumps(replacements[key],ensure_ascii=False)
            remaining.pop(key,None)
        lines.append(line)
    lines += [k+'='+json.dumps(v,ensure_ascii=False) for k,v in remaining.items()]
    target.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'created':True,'provider':'dashscope','model_names':['qwen-plus','text-embedding-v4','gte-rerank-v2'],
        'env_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'env_ignored_and_not_for_archive':True,
        'model_calls':0,'existing_files_overwritten':False},ensure_ascii=False))

if __name__=='__main__':main()
