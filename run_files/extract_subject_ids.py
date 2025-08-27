#!/usr/bin/env python3
import sys, json

if not (3 <= len(sys.argv) <= 4):
    print(f"usage: {sys.argv[0]} INPUT.jsonl OUTPUT.txt [LIMIT]")
    sys.exit(2)

src, dst = sys.argv[1], sys.argv[2]
limit = int(sys.argv[3]) if len(sys.argv) == 4 else None

seen = set(); wrote = 0
with open(src, "r", encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as out:
    for i, line in enumerate(f, 1):
        line = line.strip()
        if not line: continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        for k in ("SUBJECT_ID", "subject_id"):
            v = obj.get(k)
            if v is None: continue
            s = str(v).strip()
            if s.isdigit() and s not in seen:
                seen.add(s); out.write(s + "\n"); wrote += 1
                if limit and wrote >= limit:
                    print(f"wrote {wrote} IDs to {dst} (limit reached)")
                    sys.exit(0)
print(f"wrote {wrote} IDs to {dst}")
