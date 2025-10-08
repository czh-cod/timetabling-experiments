import xml.etree.ElementTree as ET
import pandas as pd
import os

# === 1️⃣ 自动定位项目根目录 ===
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# === 2️⃣ 需要批量解析的数据集列表 ===
datasets = [
    "pu-spr07-cs.xml",
    "pu-spr07-sa.xml",
    "pu-spr07-ecet.xml"
]

# === 3️⃣ 批量解析循环 ===
for file in datasets:
    xml_path = os.path.join(base_dir, "datasets", file)
    print(f"\n📂 Parsing {file} ...")
    print("Loading XML from:", xml_path)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    print("Days:", root.get("nrDays"))
    print("Slots per day:", root.get("slotsPerDay"))
    print("-" * 50)

    # === 4️⃣ 解析房间信息 ===
    rooms = []
    for r in root.findall(".//room"):
        rooms.append({
            "id": r.get("id"),
            "capacity": r.get("capacity"),
            "location": r.get("location")
        })
    print(f"Parsed {len(rooms)} rooms.")

    # === 5️⃣ 解析教师信息 ===
    instructors = []
    for ins in root.findall(".//instructor"):
        instructors.append({
            "id": ins.get("id"),
            "name": ins.get("name")
        })
    print(f"Parsed {len(instructors)} instructors.")

    # === 6️⃣ 解析课程信息 ===
    classes = []
    for c in root.findall(".//class"):
        classes.append({
            "id": c.get("id"),
            "course": c.get("course"),
            "limit": c.get("limit"),
            "instructor": c.get("instructor")
        })
    print(f"Parsed {len(classes)} classes.")

    # === 7️⃣ 导出为 CSV 文件 ===
    output_dir = os.path.join(base_dir, "datasets")
    os.makedirs(output_dir, exist_ok=True)

    prefix = file.replace(".xml", "")
    pd.DataFrame(rooms).to_csv(os.path.join(output_dir, f"{prefix}_rooms.csv"), index=False)
    pd.DataFrame(instructors).to_csv(os.path.join(output_dir, f"{prefix}_instructors.csv"), index=False)
    pd.DataFrame(classes).to_csv(os.path.join(output_dir, f"{prefix}_classes.csv"), index=False)

    print("-" * 50)
    print(f"✅ CSV files for {file} successfully saved to:", output_dir)

print("\n🎯 All datasets parsed successfully!")
