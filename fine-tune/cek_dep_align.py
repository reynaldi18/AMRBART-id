# coding: utf-8
"""Cek perataan label dependensi dengan tokenizer ASLI checkpoint, sebelum training M3/M4.

Menjalankan perataan yang sama persis dengan dataset.py (input konkatenasi, potong 400, token
AMR penutup) pada seluruh train.jsonl, mencetak statistik, dan satu contoh kalimat negasi.
CPU saja, beberapa menit.

  python cek_dep_align.py --model ../models/mbart-concat-pretrained \
      --data ../ds/amrbart-id-concat-train-m3-opus
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_interface.tokenization_bart import AMRBartTokenizer
from data_interface_concat.dep_align import align_example

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--data", required=True, help="folder berisi train.jsonl, train.dep.jsonl, dep_labels.json")
ap.add_argument("--max_source_length", type=int, default=400)
ap.add_argument("--n", type=int, default=0, help="0 = semua baris")
a = ap.parse_args()

tok = AMRBartTokenizer.from_pretrained(a.model)
labels = json.load(open(f"{a.data}/dep_labels.json"))["labels"]
r2i = {r: i for i, r in enumerate(labels)}
en_id = tok.lang_code_to_id["en_XX"]
st, contoh, t0 = {}, None, time.time()
L = a.max_source_length
with open(f"{a.data}/train.jsonl") as ft, open(f"{a.data}/train.dep.jsonl") as fd:
    for k, (lt, ld) in enumerate(zip(ft, fd)):
        if a.n and k >= a.n:
            break
        x, d = json.loads(lt), json.loads(ld)
        ids = tok("id_ID " + x["id"] + " en_XX " + x["en"], max_length=L, truncation=True)["input_ids"]
        ids = ids[:L - 3] + [tok.amr_bos_token_id, tok.mask_token_id, tok.amr_eos_token_id]
        words = [[s, e, h, r2i.get(r, -1)] for s, e, h, r in d["w"]]
        H, R = align_example(ids, x["en"], words, tok, en_id, stats=st)
        if contoh is None and " not " in f" {x['en']} " and len(x["en"]) < 90 and ":polarity -" in x["amr"]:
            contoh = (x, ids, H, R)
        if k % 20000 == 0:
            print(f"  {k} baris, {time.time()-t0:.0f} dtk", flush=True)

print(json.dumps(st, indent=1))
kata = max(st.get("kata", 0), 1)
print(f"kata diawasi : {100*st.get('kata_diawasi',0)/kata:.2f}%")
lewat = sum(v for k, v in st.items() if k.startswith("lewat_"))
print(f"kalimat dilewati: {lewat} dari {st.get('kalimat',0)}")
if contoh:
    x, ids, H, R = contoh
    t = tok.convert_ids_to_tokens(ids)
    print("\ncontoh:", x["en"])
    for p, (h, r) in enumerate(zip(H, R)):
        if h >= 0:
            print(f"  {t[p]:>14} -> {t[h]:<14} {labels[r]}")
