from pathlib import Path

file_path = Path("core/live_runner.py")
content = file_path.read_text(encoding="utf-8")
lines = content.splitlines()

new_lines = []
for l in lines:
    if "tp_res = self.broker.send_order" in l:
        if new_lines and "거래소 호가창에 동적 ATR" in new_lines[-1]:
            new_lines.pop()
        continue
    if "LimitTPOrder" in l or "동적ATR 예약매도 등록" in l:
        continue
    new_lines.append(l)

file_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
print("Cleaned live_runner.py successfully!")
