import json

try:
    with open('c:/project/bot/pyright.json', 'r', encoding='utf-16le') as f:
        data = json.load(f)

    for diag in data.get('generalDiagnostics', []):
        msg = diag['message']
        if "Could not find import" in msg and "type: ignore" in msg:
            continue
        print(f"{diag['file']}:{diag['range']['start']['line']+1} - {msg}")
except Exception as e:
    print("Error:", e)
