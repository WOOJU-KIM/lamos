import json

with open('data/system_logs.jsonl', 'r', encoding='utf-8') as f:
    for i in range(10):
        line = f.readline()
        if not line: break
        d = json.loads(line)
        print(f"Row {i}: keys={list(d.keys())}")
        if i == 0:
            for k, v in d.items():
                print(f"  {k}: {str(v)[:100]}")
