from pathlib import Path

app_path = Path("app.py")
content = app_path.read_text(encoding="utf-8")

old_str = '        grounding_label = (\n            "Verified"\n            if result.grounding_verified\n            else ("Insufficient" if not result.evidence_sufficient else "Not Grounded")\n        )'
new_str = '        grounding_label = (\n            "Grounded"\n            if result.grounding_verified\n            else ("Insufficient" if not result.evidence_sufficient else "Not Grounded")\n        )'

if old_str in content:
    content = content.replace(old_str, new_str, 1)
    app_path.write_text(content, encoding="utf-8")
    print("Updated grounding label to 'Grounded'")
else:
    # Try with CRLF
    old_crlf = old_str.replace("\n", "\r\n")
    new_crlf = new_str.replace("\n", "\r\n")
    if old_crlf in content:
        content = content.replace(old_crlf, new_crlf, 1)
        app_path.write_text(content, encoding="utf-8")
        print("Updated grounding label to 'Grounded' (CRLF)")
    else:
        print("Target string not found")
