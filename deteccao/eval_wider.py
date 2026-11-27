"""Comparação dos detetores no WIDER FACE (validação) vs ground truth:
exatidão (AP, curva P-R, P/R no threshold de operação), tempo, e robustez a
caras pequenas, pose, oclusão parcial, blur, iluminação e nº de caras por imagem.

Estrutura esperada:
    data/WIDER/WIDER_val/images/<evento>/*.jpg
    data/WIDER/wider_face_split/wider_face_val_bbx_gt.txt

Uso:
    python deteccao/eval_wider.py                                # validação completa, 3 detetores
    python deteccao/eval_wider.py --max-images 300               # amostra aleatória (seed fixa)
    python deteccao/eval_wider.py --detectors scrfd retinaface_mobile

As deteções ficam em cache (data/WIDER/cache_<detetor>.pkl): voltar a correr só recalcula as métricas.
Matching: IoU >= 0.5, cada cara GT casa com no máximo 1 deteção (por ordem de score);
caras marcadas "invalid" no GT são ignoradas (deteções sobre elas nem TP nem FP).
"""
import argparse
import os
import pickle
import random
import sys
import time

import cv2 as cv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # raiz do projeto
from deteccao.face_detector import DETECTORS, create_detector  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WIDER = os.path.join(ROOT, "data", "WIDER")
OUT = os.path.join(ROOT, "deteccao", "resultados")
IOU = 0.5
LOW_THR = 0.02                                               # para traçar a curva P-R
OP_THR = {"scrfd": 0.5, "retinaface_mobile": 0.9, "retinaface": 0.9}   # thresholds da aplicação
NAMES = {"retinaface": "RetinaFace R50", "retinaface_mobile": "RetinaFace MobileNet0.25",
         "scrfd": "SCRFD-500MF"}

# atributos do GT do WIDER: x y w h blur expression illumination invalid occlusion pose
CONDITIONS = {
    "Tamanho (√(w·h) px)": ("size", [("< 16", 0, 16), ("16–32", 16, 32), ("32–64", 32, 64),
                                     ("64–128", 64, 128), ("≥ 128", 128, 1e9)]),
    "Pose": (9, [("típica", 0), ("atípica", 1)]),
    "Oclusão": (8, [("nenhuma", 0), ("parcial", 1), ("forte", 2)]),
    "Blur": (4, [("nítida", 0), ("normal", 1), ("forte", 2)]),
    "Iluminação": (6, [("normal", 0), ("extrema", 1)]),
    "Expressão": (5, [("típica", 0), ("exagerada", 1)]),
}
FACE_GROUPS = [("1", 1, 1), ("2–5", 2, 5), ("6–20", 6, 20), ("> 20", 21, 10 ** 6)]


def parse_gt(path):
    out, lines, i = {}, open(path).read().split("\n"), 0
    while i < len(lines) and lines[i].strip():
        name, n = lines[i].strip(), int(lines[i + 1])
        rows = [list(map(int, l.split())) for l in lines[i + 2:i + 2 + max(n, 1)]]
        out[name] = np.array(rows[:n]).reshape(-1, 10) if n else np.zeros((0, 10), int)
        i += 2 + max(n, 1)
    return out


def iou_matrix(a, b):
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)


def run_detector(name, images):
    """Corre o detetor (threshold baixo) e guarda em cache: {img: (boxes Nx4, scores N, ms)}."""
    cache = os.path.join(WIDER, f"cache_{name}.pkl")
    res = pickle.load(open(cache, "rb")) if os.path.exists(cache) else {}
    todo = [n for n in images if n not in res]
    if todo:
        det = create_detector(name, threshold=LOW_THR)
        det.detect(cv.imread(os.path.join(WIDER, "WIDER_val", "images", todo[0])))   # aquecimento
        for k, n in enumerate(todo):
            img = cv.imread(os.path.join(WIDER, "WIDER_val", "images", n))
            t0 = time.perf_counter()
            faces = det.detect(img)
            ms = (time.perf_counter() - t0) * 1000
            res[n] = (np.array([f["box"] for f in faces], float).reshape(-1, 4),
                      np.array([f["score"] for f in faces], float), ms)
            if (k + 1) % 100 == 0 or k + 1 == len(todo):
                print(f"  [{name}] {k + 1}/{len(todo)}")
                pickle.dump(res, open(cache, "wb"))
    return {n: res[n] for n in images}


def evaluate(dets, gt, op_thr):
    """Matching por imagem. Devolve dados para AP e para robustez no threshold de operação."""
    all_scores, all_tp, n_valid = [], [], 0
    face_rows = []     # por cara GT válida: (atributos..., size, detetada@op, nº caras da imagem)
    img_rows = []      # por imagem: (nº caras válidas, TP@op, FP@op)
    for name, ann in gt.items():
        boxes, scores, _ = dets[name]
        g = ann[:, :4].astype(float).copy()
        g[:, 2:] += g[:, :2]
        valid = ann[:, 7] == 0
        n_valid += int(valid.sum())
        I = iou_matrix(boxes, g) if len(boxes) and len(g) else np.zeros((len(boxes), len(g)))
        used = np.zeros(len(g), bool)
        hit_op = np.zeros(len(g), bool)
        fp_op = tp_op = 0
        for j in np.argsort(-scores):
            cand = np.where(~used & (I[j] >= IOU))[0] if len(g) else []
            if len(cand):
                b = cand[np.argmax(I[j, cand])]
                used[b] = True
                if not valid[b]:
                    continue
                all_scores.append(scores[j]); all_tp.append(1)
                if scores[j] >= op_thr:
                    hit_op[b] = True; tp_op += 1
            else:
                all_scores.append(scores[j]); all_tp.append(0)
                fp_op += int(scores[j] >= op_thr)
        size = np.sqrt(np.maximum(ann[:, 2], 0) * np.maximum(ann[:, 3], 0))
        for i in np.where(valid)[0]:
            face_rows.append((*ann[i], size[i], hit_op[i], int(valid.sum())))
        img_rows.append((int(valid.sum()), tp_op, fp_op))
    return np.array(all_scores), np.array(all_tp), n_valid, np.array(face_rows), np.array(img_rows)


def pr_curve(scores, tp, n_gt):
    order = np.argsort(-scores)
    tp = tp[order]
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    rec, prec = ctp / max(n_gt, 1), ctp / np.maximum(ctp + cfp, 1e-9)
    mrec, mpre = np.r_[0, rec, 1], np.r_[0, prec, 0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])), rec, prec, scores[order]


def plot_pr(curves, path):
    """Curva P-R desenhada com OpenCV (sem matplotlib)."""
    W, H, L, B = 700, 560, 70, 60
    img = np.full((H, W, 3), 255, np.uint8)
    pw, ph = W - L - 20, H - B - 40
    to = lambda r, p: (int(L + r * pw), int(40 + (1 - p) * ph))
    for v in np.linspace(0, 1, 6):
        cv.line(img, to(v, 0), to(v, 1), (230, 230, 230), 1)
        cv.line(img, to(0, v), to(1, v), (230, 230, 230), 1)
        cv.putText(img, f"{v:.1f}", (to(v, 0)[0] - 12, H - B + 22), cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        cv.putText(img, f"{v:.1f}", (L - 40, to(0, v)[1] + 5), cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    cv.rectangle(img, to(0, 1), to(1, 0), (0, 0, 0), 1)
    colors = [(200, 80, 40), (40, 160, 40), (40, 40, 220)]
    for k, (label, (ap, rec, prec, sc, op)) in enumerate(curves.items()):
        c = colors[k % 3]
        pts = np.array([to(r, p) for r, p in zip(rec, prec)], np.int32)
        cv.polylines(img, [pts], False, c, 2, cv.LINE_AA)
        j = np.searchsorted(-sc, -op, side="right") - 1          # ponto do threshold de operação
        if j >= 0:
            cv.circle(img, to(rec[j], prec[j]), 6, c, -1)
        cv.putText(img, f"{label}  AP={ap:.3f}", (L + 10, H - B - 15 - 22 * (len(curves) - 1 - k)),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv.LINE_AA)
    cv.putText(img, "WIDER FACE val - precisao vs recall (IoU 0.5); ponto = threshold da app",
               (L - 30, 25), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
    cv.putText(img, "recall", (W // 2, H - 12), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    cv.putText(img, "precisao", (5, 30 + ph // 2), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv.imwrite(path, img)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--detectors", nargs="+", default=list(DETECTORS), choices=list(DETECTORS))
    p.add_argument("--max-images", type=int, default=0, help="0 = validação completa")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    os.makedirs(OUT, exist_ok=True)

    gt = parse_gt(os.path.join(WIDER, "wider_face_split", "wider_face_val_bbx_gt.txt"))
    names = sorted(gt)
    if a.max_images:
        random.Random(a.seed).shuffle(names)
        names = sorted(names[:a.max_images])
    gt = {n: gt[n] for n in names}
    n_faces = sum(int((v[:, 7] == 0).sum()) for v in gt.values())
    print(f"{len(gt)} imagens, {n_faces} caras válidas")

    results, curves = {}, {}
    for d in a.detectors:
        print(f"a correr {d}...")
        dets = run_detector(d, names)
        s, tp, n_valid, faces, imgs = evaluate(dets, gt, OP_THR[d])
        ap, rec, prec, sc = pr_curve(s, tp, n_valid)
        curves[NAMES[d]] = (ap, rec, prec, sc, OP_THR[d])
        ms = np.array([v[2] for v in dets.values()])
        results[d] = dict(ap=ap, faces=faces, imgs=imgs, ms=ms,
                          p=imgs[:, 1].sum() / max(imgs[:, 1].sum() + imgs[:, 2].sum(), 1),
                          r=imgs[:, 1].sum() / max(n_valid, 1))

    plot_pr(curves, os.path.join(OUT, "deteccao_curva_PR.png"))

    # ------------------------------------------------------------- relatório
    D = a.detectors
    hdr = "| | " + " | ".join(NAMES[d] for d in D) + " |"
    sep = "|---|" + "---|" * len(D)
    md = [f"# Comparação de detetores — WIDER FACE validação\n",
          f"- {len(gt)} imagens{' (amostra aleatória, seed %d)' % a.seed if a.max_images else ' (validação completa)'}, "
          f"{n_faces} caras válidas (GT `invalid` ignoradas)",
          f"- Matching IoU ≥ {IOU}; AP com o detetor a threshold {LOW_THR}; restantes métricas no **threshold da "
          f"aplicação**: " + ", ".join(f"{NAMES[d]} {OP_THR[d]}" for d in D),
          "- Só CPU (Apple M1, Python x86_64 via Rosetta)\n",
          "## 1. Exatidão e tempo\n", hdr, sep,
          "| **AP** (IoU 0.5) | " + " | ".join(f"**{results[d]['ap']:.3f}**" for d in D) + " |",
          "| Precisão @ threshold app | " + " | ".join(f"{results[d]['p']:.3f}" for d in D) + " |",
          "| Recall @ threshold app | " + " | ".join(f"{results[d]['r']:.3f}" for d in D) + " |",
          "| ms/imagem (média) | " + " | ".join(f"{results[d]['ms'].mean():.1f}" for d in D) + " |",
          "| ms/imagem (p95) | " + " | ".join(f"{np.percentile(results[d]['ms'], 95):.1f}" for d in D) + " |",
          "| FPS | " + " | ".join(f"{1000 / results[d]['ms'].mean():.1f}" for d in D) + " |",
          "\nCurva P-R: `deteccao_curva_PR.png`\n",
          "## 2. Robustez — recall por condição (threshold da app)\n",
          "Recall = fração das caras GT dessa condição que foram detetadas; a coluna \"caras\" indica quantas caras GT há nessa condição.\n"]
    for title, (col, bins) in CONDITIONS.items():
        md += [f"### {title}\n", "| " + title.split(" (")[0] + " | caras | " + " | ".join(NAMES[d] for d in D) + " |",
               "|---|---|" + "---|" * len(D)]
        for b in bins:
            f0 = results[D[0]]["faces"]
            if col == "size":
                mask = lambda F: (F[:, 10] >= b[1]) & (F[:, 10] < b[2])
            else:
                mask = lambda F, c=col, v=b[1]: F[:, c] == v
            n = int(mask(f0).sum())
            cells = [f"{results[d]['faces'][mask(results[d]['faces']), 11].mean():.3f}" if n else "—" for d in D]
            md.append(f"| {b[0]} | {n} | " + " | ".join(cells) + " |")
        md.append("")
    md += ["## 3. Robustez — várias caras na mesma imagem (threshold da app)\n",
           "| Caras por imagem | imagens | " + " | ".join(f"{NAMES[d]} R / P" for d in D) + " |",
           "|---|---|" + "---|" * len(D)]
    for label, lo, hi in FACE_GROUPS:
        cells, n_img = [], 0
        for d in D:
            I = results[d]["imgs"]
            m = (I[:, 0] >= lo) & (I[:, 0] <= hi)
            n_img = int(m.sum())
            tp, fp, g = I[m, 1].sum(), I[m, 2].sum(), I[m, 0].sum()
            cells.append(f"{tp / max(g, 1):.3f} / {tp / max(tp + fp, 1):.3f}" if n_img else "—")
        md.append(f"| {label} | {n_img} | " + " | ".join(cells) + " |")

    with open(os.path.join(OUT, "deteccao_comparacao.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\n-> {OUT}/deteccao_comparacao.md, deteccao_curva_PR.png")


if __name__ == "__main__":
    main()
