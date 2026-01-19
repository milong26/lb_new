import re
import ast
import csv

# -------- 工具函数：解析一行内容 --------
def parse_line(line):
    import re, ast

    # --- 解析 ee-action ---
    ee_pattern = r"ee-action(?:是)?=(\{.*?\})"
    ee_match = re.search(ee_pattern, line)
    ee_action = ast.literal_eval(ee_match.group(1)) if ee_match else {}

    # --- 解析 state ---
    state_pattern = r"state_joint_obs[:：](.*?),joint_action"
    state_match = re.search(state_pattern, line)
    state = {}
    if state_match:
        state_raw = state_match.group(1)
        pairs = state_raw.split(',')
        for p in pairs:
            if '=' in p:
                k, v = p.split('=')
                try:
                    state[k] = float(v)
                except:
                    state[k] = "空"

    # --- 解析 joint_action ---
    joint_pattern = r"joint_action=(\{.*\})"
    joint_match = re.search(joint_pattern, line)
    joint_action = ast.literal_eval(joint_match.group(1)) if joint_match else {}

    return ee_action, state, joint_action



# -------- 主程序：读取两个文件 --------
def load_file(fname, tag):
    data = []
    with open(fname, 'r', encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(parse_line(line, tag))
    return data


code1 = load_file("code1.txt", "code1")
code2 = load_file("code2.txt", "code2")

# 对齐：每一行视为一步
steps = list(zip(code1, code2))
# 若长度不同，则补齐
max_len = max(len(code1), len(code2))
if len(code1) < max_len:
    code1 += [None] * (max_len - len(code1))
if len(code2) < max_len:
    code2 += [None] * (max_len - len(code2))
steps = [(code1[i], code2[i]) for i in range(max_len)]


# -------- 写入 CSV --------
with open("output.csv", "w", newline='', encoding="utf-8-sig") as f:
    writer = csv.writer(f)

    writer.writerow([" ", "ee-action", "state", "joint-action"])  # 表头

    for i, (c1, c2) in enumerate(steps, start=1):
        for item in (c1, c2):
            if item is None:
                continue

            # 第一行：stepX:code?
            writer.writerow([f"step{i}:{item['file']}", "", "", ""])

            # ee-action
            for k, v in item["ee"].items():
                writer.writerow(["", k, "", v])

            # state (obs)
            for k, v in item["state"].items():
                writer.writerow(["", "", k, v])

            # joint-action
            for k, v in item["joint"].items():
                writer.writerow(["", "", "", f"{k}={v}"])

print("CSV 写入完成：output.csv")
