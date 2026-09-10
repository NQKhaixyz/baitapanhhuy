import json, csv, os

here = os.path.dirname(__file__)
qa = [json.loads(l) for l in open(os.path.join(here, "..", "qa.sample.jsonl"), encoding="utf-8")]

# ---- KIỂU A: 1 file phẳng, nhồi mảng vào 1 ô bằng ký tự ngăn ----
with open(os.path.join(here, "A_flat.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["id", "type_id", "type", "specialty", "question",
                "retrieval_chunks", "retrieval_points", "answer_gt", "is_answerable"])
    for it in qa:
        chunks = "|".join(cid for c in it["retrieval_gt"] for cid in c["chunk_ids"])
        pts = " ;; ".join(p["text"] for c in it["retrieval_gt"] for p in c["points"])
        w.writerow([it["id"], it["type_id"], it["type"], it["specialty"], it["question"],
                    chunks, pts, it["answer_gt"], it["meta"]["is_answerable"]])

# ---- KIỂU B: nhiều file chuẩn hóa (như bảng DB), nối bằng ID ----
with open(os.path.join(here, "B_items.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["item_id", "type_id", "type", "specialty", "question", "answer_gt", "is_answerable"])
    for it in qa:
        w.writerow([it["id"], it["type_id"], it["type"], it["specialty"],
                    it["question"], it["answer_gt"], it["meta"]["is_answerable"]])

with open(os.path.join(here, "B_citations.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["citation_id", "item_id", "chunk_id"])
    for it in qa:
        for i, c in enumerate(it["retrieval_gt"], 1):
            cite_id = "{}_c{}".format(it["id"], i)
            for cid in c["chunk_ids"]:
                w.writerow([cite_id, it["id"], cid])

with open(os.path.join(here, "B_points.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["point_id", "citation_id", "text", "verbatim_quote"])
    for it in qa:
        for i, c in enumerate(it["retrieval_gt"], 1):
            cite_id = "{}_c{}".format(it["id"], i)
            for p in c["points"]:
                w.writerow(["{}_{}".format(cite_id, p["id"]), cite_id,
                            p["text"], p.get("verbatim_quote", "")])

print("Da tao xong.")
