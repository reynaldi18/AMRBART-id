# coding: utf-8
"""Praproses label dependensi untuk M3/M4: parsing kolom `en` dengan Stanza (UD v2).

Kolom `en` identik antara data OPUS dan NLLB, jadi SATU keluaran dipakai M3 dan M4.
Keluaran (berkas pendamping; train.jsonl tidak diubah sama sekali):

  <out>.dep.jsonl   satu baris JSON per baris masukan, urutan sama:
                    {"i": no_baris, "h": md5(en)[:12], "w": [[awal, akhir, head, deprel], ...]}
                    awal/akhir = offset karakter di `en`; head 1-based, 0 = root
  <out>.meta.json   versi stanza, paket model, jumlah kalimat/kata, kalimat ganda-root, durasi
  dep_labels.json   (hanya untuk --labels-out) daftar label dari data latih, urutan tetap

Cache per potongan (--chunk baris) di <out>.parts/ sehingga bisa dilanjutkan bila sesi mati.

Pemakaian (Kaggle, Python bawaan dengan GPU; BUKAN env conda `amr`):
  pip install -q stanza
  python prep_dep.py --input train.jsonl --out train --labels-out dep_labels.json
  python prep_dep.py --input dev.jsonl   --out dev
  python prep_dep.py --input test.jsonl  --out test
"""
import argparse
import collections
import glob
import hashlib
import json
import os
import time


def en_hash(s):
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:12]


def word_spans(token):
    """Rentang karakter tiap word dalam satu token. Multi-word token (mis. 'cannot' -> can + not)
    dibagi dengan mencari teks tiap word secara berurutan di dalam teks token."""
    ws = token.words
    if len(ws) == 1:
        return [(token.start_char, token.end_char)]
    text = token.text.lower()
    out, pos = [], 0
    for w in ws:
        k = text.find(w.text.lower(), pos)
        if k < 0:
            return [(token.start_char, token.end_char)] * len(ws)
        out.append((token.start_char + k, token.start_char + k + len(w.text)))
        pos = k + len(w.text)
    return out


def parse_doc(doc):
    """Gabungkan semua kalimat menjadi satu daftar kata; indeks head digeser per kalimat."""
    words, n_root, offset = [], 0, 0
    for sent in doc.sentences:
        spans = {}
        for tok in sent.tokens:
            for w, sp in zip(tok.words, word_spans(tok)):
                spans[w.id] = sp
        for w in sent.words:
            s, e = spans[w.id]
            head = 0 if w.head == 0 else w.head + offset
            n_root += w.head == 0
            words.append([s, e, head, w.deprel])
        offset += len(sent.words)
    return words, n_root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True, help="prefiks keluaran, mis. ds/.../train")
    ap.add_argument("--labels-out", default=None)
    ap.add_argument("--package", default="default", help="paket model Stanza bahasa Inggris")
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=64, help="batch kalimat untuk depparse")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    import stanza
    rows = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    texts = [r["en"] for r in rows]
    print(f"{len(texts)} kalimat dari {args.input}")

    stanza.download("en", package=args.package, processors="tokenize,mwt,pos,lemma,depparse",
                    verbose=False)
    nlp = stanza.Pipeline("en", package=args.package, processors="tokenize,mwt,pos,lemma,depparse",
                          tokenize_no_ssplit=True, use_gpu=not args.cpu, verbose=False,
                          depparse_batch_size=args.batch * 50, pos_batch_size=args.batch * 50)

    part_dir = args.out + ".parts"
    os.makedirs(part_dir, exist_ok=True)
    t0 = time.time()
    for c0 in range(0, len(texts), args.chunk):
        part = os.path.join(part_dir, f"{c0:07d}.jsonl")
        if os.path.exists(part):
            continue
        chunk = texts[c0:c0 + args.chunk]
        # newline -> spasi (panjang sama, offset tetap); string kosong diparse sebagai kosong
        docs_in = [stanza.Document([], text=t.replace("\n", " ").replace("\r", " ")) for t in chunk]
        nonempty = [k for k, t in enumerate(chunk) if t.strip()]
        docs = nlp([docs_in[k] for k in nonempty]) if nonempty else []
        res = {k: parse_doc(d) for k, d in zip(nonempty, docs)}
        tmp = part + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for k, t in enumerate(chunk):
                w, nr = res.get(k, ([], 0))
                f.write(json.dumps({"i": c0 + k, "h": en_hash(t), "w": w, "nroot": nr},
                                   ensure_ascii=False) + "\n")
        os.replace(tmp, part)
        done = min(c0 + args.chunk, len(texts))
        el = time.time() - t0
        print(f"  {done}/{len(texts)}  {el/60:.1f} mnt", flush=True)

    # gabung + statistik
    lab = collections.Counter()
    n_words = n_multi = n_empty = 0
    with open(args.out + ".dep.jsonl", "w", encoding="utf-8") as fo:
        n = 0
        for part in sorted(glob.glob(os.path.join(part_dir, "*.jsonl"))):
            for line in open(part, encoding="utf-8"):
                d = json.loads(line)
                assert d["i"] == n, f"urutan rusak di {part}: {d['i']} != {n}"
                assert d["h"] == en_hash(texts[n]), f"hash en tidak cocok baris {n}"
                n_words += len(d["w"])
                n_multi += d.pop("nroot") > 1
                n_empty += not d["w"]
                lab.update(w[3] for w in d["w"])
                fo.write(json.dumps(d, ensure_ascii=False) + "\n")
                n += 1
    assert n == len(texts), f"baris keluaran {n} != masukan {len(texts)}"

    meta = {
        "input": os.path.abspath(args.input), "n": n, "kata": n_words,
        "kalimat_multi_root": n_multi, "kalimat_kosong": n_empty,
        "stanza": stanza.__version__, "package": args.package,
        "processors": "tokenize,mwt,pos,lemma,depparse", "tokenize_no_ssplit": True,
        "label_counts": dict(lab.most_common()),
    }
    with open(args.out + ".meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    if args.labels_out:
        labels = sorted(lab)
        with open(args.labels_out, "w") as f:
            json.dump({"labels": labels, "sumber": os.path.abspath(args.input),
                       "stanza": stanza.__version__, "package": args.package}, f, indent=1)
        print(f"{len(labels)} label -> {args.labels_out}")
    print(f"selesai: {n} kalimat, {n_words} kata, multi-root {n_multi}, kosong {n_empty}")


if __name__ == "__main__":
    main()
