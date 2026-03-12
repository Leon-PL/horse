"""Fix min_child_samples placement in model.py factories."""
with open("src/model.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

count = 0
new_lines = []
for line in lines:
    s = line.strip()
    # Match the lines we added: "min_child_samples": config.XXX_PARAMS.get("min_child_samples", 10),
    if ('"min_child_samples": config.' in s
        and "PARAMS.get(" in s
        and s.endswith(",")):
        count += 1
    else:
        new_lines.append(line)

print(f"Removed {count} min_child_samples lines from hp dicts")

with open("src/model.py", "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print("Done")
