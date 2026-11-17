# coding: utf-8
"""Perataan kata hasil parser dependensi (Stanza) ke sub-token SentencePiece pada input konkatenasi.

Input encoder: id_ID <indo> en_XX <eng> </s> <AMR> <mask> </AMR>
(tokenization_mbart50.py versi repo ini mengosongkan prefix_tokens, jadi tidak ada kode bahasa di
depan). Tokenizer slow memecah teks di kode bahasa, jadi potongan setelah `en_XX` adalah
SentencePiece atas kalimat Inggris saja. Modul ini:
  1. menghitung rentang karakter tiap sub-token Inggris (tanpa bergantung versi sentencepiece),
  2. memetakan tiap kata ke sub-token PERTAMA yang tumpang-tindih dan belum dipakai kata lain,
  3. menghasilkan dua larik sepanjang input_ids:
       dep_heads: -100 = bukan kandidat;  -1 = kandidat kepala yang bukan dependen (ROOT, kata
                  tak terawasi);  >=0 = posisi kepala emas untuk dependen ini
       dep_rels : id label untuk dependen terawasi, -100 selain itu.
ROOT = posisi `en_XX`.
"""
import unicodedata

IGNORE = -100
CAND = -1

_SKIPPABLE = {"Cc", "Cf", "Zs", "Zl", "Zp", "Mn"}


def _norm_stream(text):
    """Karakter non-spasi teks setelah NFKC per karakter, beserta indeks karakter aslinya."""
    out = []
    for i, ch in enumerate(text):
        for c in unicodedata.normalize("NFKC", ch):
            if not c.isspace():
                out.append((c, i))
    return out


def piece_char_spans(pieces, text, unk_piece="<unk>"):
    """Rentang karakter [awal, akhir) tiap piece di `text`; None untuk piece tanpa karakter
    (mis. '▁' tunggal). Mengembalikan None bila penjajaran gagal."""
    stream = _norm_stream(text)
    j = 0
    spans = []
    for k, p in enumerate(pieces):
        surf = p.replace("▁", "")
        if p == unk_piece or surf == unk_piece:
            # unk: makan karakter sampai karakter pertama piece berikutnya cocok
            nxt = None
            for q in pieces[k + 1:]:
                s = q.replace("▁", "")
                if s and s != unk_piece:
                    nxt = s[0]
                    break
            start = j
            while j < len(stream) and (nxt is None or stream[j][0] != nxt or j == start):
                j += 1
                if nxt is None:
                    break
            idx = [stream[t][1] for t in range(start, j)]
            spans.append((min(idx), max(idx) + 1) if idx else None)
            continue
        idx = []
        for c in surf:
            if c.isspace():
                continue
            # lewati karakter yang dibuang normalisasi SentencePiece (kontrol, format, combining)
            while j < len(stream) and stream[j][0] != c and unicodedata.category(stream[j][0]) in _SKIPPABLE:
                j += 1
            if j >= len(stream) or stream[j][0] != c:
                return None
            idx.append(stream[j][1])
            j += 1
        spans.append((min(idx), max(idx) + 1) if idx else None)
    return spans


def align_example(input_ids, en_text, words, tokenizer, en_lang_id, pieces=None, stats=None):
    """words: list [start_char, end_char, head(1-based, 0=root), rel_id(-1 = label tak dikenal)].

    Mengembalikan (dep_heads, dep_rels) sepanjang input_ids, atau (None, None) bila kalimat dilewati.
    """
    n = len(input_ids)
    heads = [IGNORE] * n
    rels = [IGNORE] * n

    def bump(key, v=1):
        if stats is not None:
            stats[key] = stats.get(key, 0) + v

    bump("kalimat")
    bump("kata", len(words))
    if not words:
        bump("kalimat_tanpa_kata")
        return heads, rels

    # posisi en_XX penanda segmen Inggris (yang terakhir, kalau-kalau ada prefix kode bahasa)
    occ = [i for i, t in enumerate(input_ids) if t == en_lang_id]
    if not occ:
        bump("lewat_tanpa_segmen_inggris")
        bump("kata_hilang", len(words))
        return heads, rels
    root = occ[-1]
    seg = root + 1

    if pieces is None:
        # tokenize() penuh, bukan sp_model saja: token AMR tambahan (mis. '-15', '-01') ikut
        # memecah teks Inggris persis seperti di dalam input konkatenasi
        pieces = tokenizer.tokenize(en_text.strip())
    lead = len(en_text) - len(en_text.lstrip())
    spans = piece_char_spans(pieces, en_text.strip(), unk_piece=getattr(tokenizer, "unk_token", "<unk>"))
    if spans is None:
        bump("lewat_penjajaran_gagal")
        bump("kata_hilang", len(words))
        return heads, rels
    spans = [None if s is None else (s[0] + lead, s[1] + lead) for s in spans]

    # bagian segmen yang benar-benar ada di input (setelah pemotongan) harus sama persis
    expected = tokenizer.convert_tokens_to_ids(pieces)
    avail = 0
    while avail < len(expected) and seg + avail < n and input_ids[seg + avail] == expected[avail]:
        avail += 1
    if avail < len(expected):
        nxt = input_ids[seg + avail] if seg + avail < n else None
        # berhenti wajar hanya bila input habis atau terpotong ke token AMR penutup
        if nxt is not None and nxt not in (tokenizer.amr_bos_token_id,):
            bump("lewat_id_tidak_cocok")
            bump("kata_hilang", len(words))
            return heads, rels
        bump("kalimat_terpotong")

    # 1) kata -> piece pertama yang tumpang-tindih & belum dipakai
    claimed = set()
    wpos = [None] * len(words)        # posisi untuk diawasi
    anchor = [None] * len(words)      # posisi yang memuat awal kata (dipakai sebagai target kepala)
    for w, (ws, we, _, _) in enumerate(words):
        cand = [k for k, s in enumerate(spans) if s is not None and s[0] < we and s[1] > ws and k < avail]
        if not cand:
            bump("kata_tanpa_piece")
            continue
        anchor[w] = seg + cand[0]
        free = [k for k in cand if k not in claimed]
        if not free:
            bump("kata_bertabrakan")
            continue
        claimed.add(free[0])
        wpos[w] = seg + free[0]
        if free[0] != cand[0]:
            bump("kata_digeser")

    heads[root] = CAND
    for w, p in enumerate(wpos):
        if p is not None:
            heads[p] = CAND

    # 2) isi kepala emas
    for w, (ws, we, h, r) in enumerate(words):
        p = wpos[w]
        if p is None:
            continue
        if h == 0:
            tgt = root
        else:
            tgt = wpos[h - 1] if wpos[h - 1] is not None else anchor[h - 1]
        if tgt is None or tgt == p or r < 0:
            bump("kata_kepala_hilang")
            continue
        if heads[tgt] == IGNORE:
            heads[tgt] = CAND
        heads[p] = tgt
        rels[p] = r
        bump("kata_diawasi")
    return heads, rels
