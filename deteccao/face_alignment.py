"""Alinhamento de caras a partir dos 5 landmarks do detetor.

Estima uma transformação de semelhança (rotação + escala + translação, 4 graus de liberdade)
que leva os 5 landmarks detetados para o template canónico do ArcFace (112x112) e aplica-a
com warpAffine. É o pré-processamento esperado pelo modelo de reconhecimento (ArcFace).

Ordem dos landmarks (igual nos 3 detetores de face_detector.py), em coordenadas da imagem:
    olho mais à esquerda, olho mais à direita, nariz, canto da boca esq., canto da boca dir.

Uso como módulo:
    from deteccao.face_alignment import align_face
    aligned, M = align_face(frame, face["landmarks"])          # 112x112 BGR

Uso como teste (gera figuras e tabela em deteccao/resultados/):
    python deteccao/face_alignment.py
"""
import os
import sys

import cv2 as cv
import numpy as np

# template ArcFace para 112x112 (InsightFace)
ARCFACE_TEMPLATE = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                             [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


def similarity_transform(src, dst):
    """Semelhança por mínimos quadrados sobre TODOS os pontos (Umeyama, 1991) -> matriz 2x3.

    Minimiza sum ||s·R·src_i + t - dst_i||². (Não usar cv.estimateAffinePartial2D com LMEDS/RANSAC:
    com 5 pontos esses métodos ajustam só 2 pontos e o resultado fica instável.)
    """
    src, dst = np.asarray(src, np.float64), np.asarray(dst, np.float64)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s_c, d_c = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(d_c.T @ s_c / len(src))
    D = np.diag([1.0, np.sign(np.linalg.det(U) * np.linalg.det(Vt))])
    R = U @ D @ Vt
    scale = np.trace(np.diag(S) @ D) / (s_c ** 2).sum(1).mean()
    t = mu_d - scale * R @ mu_s
    return np.hstack([scale * R, t[:, None]]).astype(np.float32)


def align_face(image, landmarks, size=112):
    """-> (cara alinhada size x size, matriz afim 2x3)."""
    src = np.asarray(landmarks, dtype=np.float32).reshape(5, 2)
    M = similarity_transform(src, ARCFACE_TEMPLATE * (size / 112.0))
    return cv.warpAffine(image, M, (size, size), borderValue=0), M


def transform_points(points, M):
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    return pts @ M[:, :2].T + M[:, 2]


def eye_angle(landmarks):
    """Inclinação (graus) da linha entre os olhos; 0 = olhos na horizontal."""
    (x1, y1), (x2, y2) = np.asarray(landmarks, dtype=np.float32)[:2]
    return float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))


def simple_crop(image, box, size=112, margin=0.0):
    """Alternativa SEM alinhamento: crop da caixa (quadrado) redimensionado."""
    x1, y1, x2, y2 = box
    cx, cy, s = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1) * (1 + margin)
    M = np.float32([[size / s, 0, size / 2 - cx * size / s], [0, size / s, size / 2 - cy * size / s]])
    return cv.warpAffine(image, M, (size, size), borderValue=0)


# --------------------------------------------------------------------------- #
# Teste / demonstração
# --------------------------------------------------------------------------- #
WIDER_5 = ["0--Parade/0_Parade_Parade_0_246.jpg",
           "1--Handshaking/1_Handshaking_Handshaking_1_158.jpg",
           "11--Meeting/11_Meeting_Meeting_11_Meeting_Meeting_11_206.jpg",
           "12--Group/12_Group_Team_Organized_Group_12_Group_Team_Organized_Group_12_617.jpg",
           "2--Demonstration/2_Demonstration_Demonstration_Or_Protest_2_360.jpg"]
ANGLES = (-45, -30, -15, 0, 15, 30, 45)


def _draw_points(img, pts, color=(0, 0, 255), r=2):
    for p in np.asarray(pts).astype(int):
        cv.circle(img, tuple(p), r, color, -1)
    return img


def _label(img, text, h=22):
    bar = np.full((h, img.shape[1], 3), 255, np.uint8)
    cv.putText(bar, text, (4, h - 7), cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv.LINE_AA)
    return cv.vconcat([bar, img])


class ArcFaceEmbedder:
    """Só para o teste: descritor ArcFace (w600k_mbf.onnx) de uma cara 112x112."""

    def __init__(self, path):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name

    def __call__(self, face_bgr):
        blob = cv.dnn.blobFromImage(face_bgr, 1 / 127.5, (112, 112), (127.5,) * 3, swapRB=True)
        e = self.sess.run(None, {self.inp: blob})[0][0]
        return e / np.linalg.norm(e)


def _nearest(faces, point):
    if not faces:
        return None
    c = lambda f: np.array([(f["box"][0] + f["box"][2]) / 2, (f["box"][1] + f["box"][3]) / 2])
    f = min(faces, key=lambda f: np.linalg.norm(c(f) - point))
    return f if np.linalg.norm(c(f) - point) < 0.5 * (f["box"][2] - f["box"][0]) else None


def _frontality(f):
    """~0.4+ = frontal (nariz a meio dos olhos, olhos afastados); perto de 0 = perfil."""
    le, re, nose = np.asarray(f["landmarks"], float)[:3]
    d = np.linalg.norm(re - le) + 1e-6
    off = abs(np.dot(nose - (le + re) / 2, re - le)) / d ** 2
    return d / (f["box"][2] - f["box"][0]) * max(0.0, 1 - 2 * off)


def _patch(img, box, S=320, k=2.5):
    """Janela k x caixa à volta da cara, redimensionada para S x S (evita caras vizinhas)."""
    cx, cy, side = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2, k * (box[2] - box[0])
    T = np.float32([[S / side, 0, S / 2 - cx * S / side], [0, S / side, S / 2 - cy * S / side]])
    return cv.warpAffine(img, T, (S, S), flags=cv.INTER_CUBIC, borderValue=(127, 127, 127))


def pick_frontal_faces(det, wider_root, n=10, max_scan=150):
    """Caras do WIDER val com GT >= 150 px, pose típica, sem oclusão/blur/iluminação extrema,
    ordenadas pela frontalidade medida nos landmarks. -> [(patch 320x320, caixa no patch)]"""
    lines = open(os.path.join(wider_root, "wider_face_split", "wider_face_val_bbx_gt.txt")).read().split("\n")
    i, cands = 0, []
    while i < len(lines) and lines[i].strip() and len(cands) < max_scan:
        name, m = lines[i].strip(), int(lines[i + 1])
        rows = [list(map(int, l.split())) for l in lines[i + 2:i + 2 + max(m, 1)]][:m]
        i += 2 + max(m, 1)
        good = [r for r in rows if min(r[2], r[3]) >= 150 and r[4] == 0 and r[6] == 0 and r[7] == 0
                and r[8] == 0 and r[9] == 0]
        if good:
            x, y, w, h = good[0][:4]
            cands.append((name, (x, y, x + w, y + h)))
    scored = []
    for name, box in cands:
        img = cv.imread(os.path.join(wider_root, "WIDER_val", "images", name))
        p = _patch(img, box)
        f = _nearest(det.detect(p), np.array([160, 160]))
        if f is not None and len(f["landmarks"]) == 5:
            scored.append((_frontality(f), p, f["box"]))
    scored.sort(key=lambda t: -t[0])
    return [(p, b) for _, p, b in scored[:n]]


def rotation_series(det, emb, patch, box):
    """Roda o patch e mede: inclinação dos olhos antes/depois de alinhar e similaridade ArcFace vs 0°."""
    S = patch.shape[0]
    center = np.array([S / 2, S / 2])
    out = {}
    for a in ANGLES:
        R = cv.getRotationMatrix2D((S / 2, S / 2), a, 1.0)
        rot = cv.warpAffine(patch, R, (S, S), borderValue=(127, 127, 127))
        f = _nearest(det.detect(rot), center)
        if f is None or len(f["landmarks"]) != 5:
            out[a] = None; continue
        al, M = align_face(rot, f["landmarks"])
        cr = simple_crop(rot, f["box"])
        out[a] = [eye_angle(f["landmarks"]), eye_angle(transform_points(f["landmarks"], M)), emb(cr), emb(al), cr, al]
    if out[0] is None:
        return {a: None for a in ANGLES}
    e_cr, e_al = out[0][2], out[0][3]
    for a in ANGLES:
        if out[a]:
            out[a][2], out[a][3] = float(out[a][2] @ e_cr), float(out[a][3] @ e_al)
    return out


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, root)
    from deteccao.face_detector import create_detector
    wider = os.path.join(root, "data", "WIDER", "WIDER_val", "images")
    out = os.path.join(root, "deteccao", "resultados")
    os.makedirs(out, exist_ok=True)
    det = create_detector("scrfd")
    emb = ArcFaceEmbedder(os.path.join(root, "models", "w600k_mbf.onnx"))

    # ---- Figura 1: frame + landmarks -> alinhada 112 -> normalizada 224 (como na pág. 3 do enunciado)
    rows, residuals, angles_before = [], [], []
    for name in WIDER_5:
        img = cv.imread(os.path.join(wider, name))
        faces = [f for f in det.detect(img) if len(f["landmarks"]) == 5]
        if not faces:
            print(f"(sem deteção) {name}")
            continue
        for f in faces:     # erro de alinhamento em TODAS as caras detetadas
            aligned, M = align_face(img, f["landmarks"])
            residuals.append(np.sqrt(((transform_points(f["landmarks"], M) - ARCFACE_TEMPLATE) ** 2)
                                     .sum(1).mean()))
            angles_before.append(abs(eye_angle(f["landmarks"])))
        big = [f for f in faces if f["box"][2] - f["box"][0] >= 40] or faces
        f = max(big, key=_frontality)                              # cara mais frontal para a figura
        aligned, M = align_face(img, f["landmarks"])
        x1, y1, x2, y2 = map(int, f["box"])
        m = int(0.4 * (x2 - x1))
        raw = img[max(0, y1 - m):y2 + m, max(0, x1 - m):x2 + m].copy()
        _draw_points(raw, np.asarray(f["landmarks"]) - [max(0, x1 - m), max(0, y1 - m)],
                     r=max(2, raw.shape[1] // 60))
        raw = cv.resize(raw, (224, int(224 * raw.shape[0] / raw.shape[1])))
        raw = cv.resize(raw, (224, 224)) if raw.shape[0] > 224 else cv.copyMakeBorder(
            raw, 0, 224 - raw.shape[0], 0, 0, cv.BORDER_CONSTANT, value=(255, 255, 255))
        al = _draw_points(aligned.copy(), transform_points(f["landmarks"], M))
        al = cv.copyMakeBorder(al, 56, 56, 56, 56, cv.BORDER_CONSTANT, value=(255, 255, 255))
        norm = cv.resize(aligned, (224, 224))
        ev = name.split("/")[0]
        rows.append(cv.hconcat([_label(raw, f"{ev} ({eye_angle(f['landmarks']):+.0f} deg)"),
                                _label(al, "alinhada 112x112"), _label(norm, "normalizada 224x224")]))
    fig1 = cv.vconcat(rows)
    cv.imwrite(os.path.join(out, "alinhamento_exemplos.jpg"), fig1)

    # ---- Figura 2 + tabela: robustez a rotação no plano (10 caras grandes e frontais do WIDER val)
    faces_rot = pick_frontal_faces(det, os.path.join(root, "data", "WIDER"), n=10)
    per_angle = {a: [] for a in ANGLES}
    strip_al, strip_cr = [], []
    for k, (img, box) in enumerate(faces_rot):
        res = rotation_series(det, emb, img, box)
        for a in ANGLES:
            per_angle[a].append(res[a][:4] if res[a] else None)
        if k == 0:                                   # figura com a 1.ª cara
            for a in ANGLES:
                if res[a]:
                    strip_cr.append(_label(cv.resize(res[a][4], (150, 150)), f"{a:+d} deg"))
                    strip_al.append(_label(cv.resize(res[a][5], (150, 150)), f"{a:+d} deg"))
    fig2 = cv.vconcat([_label(cv.hconcat(strip_cr), "SEM alinhamento (crop da caixa)", 26),
                       _label(cv.hconcat(strip_al), "COM alinhamento (landmarks -> template ArcFace)", 26)])
    cv.imwrite(os.path.join(out, "alinhamento_rotacao.jpg"), fig2)
    table = []
    for a in ANGLES:
        ok = [r for r in per_angle[a] if r is not None]
        if not ok:
            table.append((a, 0, len(per_angle[a]), None, None, None, None)); continue
        r = np.array(ok)
        table.append((a, len(ok), len(per_angle[a]), np.abs(r[:, 0]).mean(), np.abs(r[:, 1]).mean(),
                      r[:, 2].mean(), r[:, 3].mean()))

    md = ["# Alinhamento de caras com landmarks\n",
          "Detetor: SCRFD-500MF (5 landmarks) · alinhamento: semelhança (4 GL) para o template ArcFace 112×112 · "
          "descritor: ArcFace MobileFaceNet (`w600k_mbf.onnx`)\n",
          "## 1. Erro de alinhamento (todas as caras detetadas nas 5 imagens)\n",
          "| Caras | Inclinação dos olhos antes (média \\|°\\|) | Erro residual landmarks↔template (px, em 112×112) |",
          "|---|---|---|",
          f"| {len(residuals)} | {np.mean(angles_before):.1f}° | {np.mean(residuals):.2f} ± {np.std(residuals):.2f} |",
          "\nFigura: `alinhamento_exemplos.jpg`\n",
          f"## 2. Robustez à rotação no plano ({len(faces_rot)} caras grandes e frontais do WIDER val)\n",
          "Cada cara é rodada de -45° a +45°. Similaridade de cosseno entre o descritor ArcFace da cara rodada e o "
          "da mesma cara a 0° (média das caras detetadas). Perto de 1 = o descritor não muda com a rotação.\n",
          "| Rotação | Detetadas | \\|inclinação olhos\\| detetada | \\|inclinação\\| após alinhar | "
          "Similaridade SEM alinhamento | Similaridade COM alinhamento |",
          "|---|---|---|---|---|---|"]
    for a, nd, nt, before, after, s_cr, s_al in table:
        md.append(f"| {a:+d}° | {nd}/{nt} | — | — | — | — |" if before is None else
                  f"| {a:+d}° | {nd}/{nt} | {before:.1f}° | {after:.1f}° | {s_cr:.3f} | **{s_al:.3f}** |")
    md.append("\nFigura: `alinhamento_rotacao.jpg`")
    with open(os.path.join(out, "alinhamento.md"), "w") as fp:
        fp.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
