import json
import sys

transcript_path = 'C:/Users/chabo/.gemini/antigravity/brain/6bd29a15-5cae-4a5f-8c66-b631de55eb9b/.system_generated/logs/transcript.jsonl'
with open(transcript_path, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        obj = json.loads(line)
        idx = obj.get('step_index')
        if 740 <= idx <= 765:
            tool_calls = obj.get('tool_calls', [])
            if tool_calls:
                for tc in tool_calls:
                    name = tc.get('name') or tc.get('function', {}).get('name')
                    args = tc.get('args') or tc.get('function', {}).get('arguments')
                    print(f"Step {idx}: {name}")
                    if isinstance(args, dict):
                        if 'CommandLine' in args:
                            print(f"  CMD: {args['CommandLine']}")
                        elif 'TargetFile' in args:
                            print(f"  File: {args['TargetFile']}")
            content = obj.get('content', '')
            if '14,077,559' in content:
                print(f"Step {idx} content contains 14,077,559!")
