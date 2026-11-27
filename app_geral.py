"""App com o modelo geral: uma rede por cara dá nome (ou Unknown), idade, género e expressão.

    python app_geral.py                        webcam (modelo do geral/resultados/config.json)
    python app_geral.py --modelo multitarefa   rede das notas da aula (um ramo por tarefa)
    python app_geral.py --source video.mp4     vídeo

Teclas: n registar a cara maior (escreve o nome e Enter; recolhe 10 amostras: vai mexendo a cabeça)
        r remover uma pessoa (escreve o nome)   L limpar a galeria   1/2/3 liga/desliga identidade / idade+género / expressão
        s liga/desliga a média pesada dos frames anteriores (multitarefa/temporal.py)   p pausa   q sair
Regista só quem deu consentimento. A galeria guarda apenas descritores (data/galeria.npz).
"""
import argparse
import threading
import time

import cv2 as cv
import numpy as np

from deteccao.face_detector import DETECTORS, create_detector
from geral.galeria import Galeria
from geral.modelo import UNKNOWN, GeralEstimator
from multitarefa.temporal import Smoother

N_SAMPLES, SAMPLE_GAP = 10, 0.35


class Worker:
    """Deteção + modelo geral numa thread; o vídeo mostra sempre o frame mais recente."""

    def __init__(self, det, est, gal):
        self.det, self.est, self.gal = det, est, gal
        self.smoother = Smoother()
        self.smooth = True
        self.lock, self.frame, self.faces, self.ms, self.running = threading.Lock(), None, [], (0, 0), True
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def loop(self):
        while self.running:
            with self.lock:
                frame, self.frame = self.frame, None
            if frame is None:
                time.sleep(0.005)
                continue
            t0 = time.perf_counter()
            faces = self.det.detect(frame)
            t1 = time.perf_counter()
            for f, r in zip(faces, self.est.predict(frame, faces, self.gal, self.smoother if self.smooth else None)):
                f["r"] = r
            with self.lock:
                self.faces, self.ms = faces, ((t1 - t0) * 1000, (time.perf_counter() - t1) * 1000)


def put(img, text, org, color, scale=0.6):
    cv.putText(img, text, org, cv.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(img, text, org, cv.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv.LINE_AA)


def draw(img, faces, on):
    GREY = (170, 170, 170)
    for f in faces:
        x1, y1, x2, y2 = f["box"]
        r = f.get("r")
        known = bool(r) and on[0] and r["name"] != UNKNOWN
        col = (0, 220, 0) if known else (0, 140, 255)
        cv.rectangle(img, (x1, y1), (x2, y2), col, 2)
        if not r:
            put(img, "sem landmarks", (x1, y1 - 8), GREY)
            continue
        if on[0]:
            put(img, f"{r['name']} {r['sim']:.2f}", (x1, max(y1 - 8, 18)), col, 0.75)
        y = y2 + 20
        if on[1]:
            g = "?" if r["gender_uncertain"] else f"{r['gender']} {r['gender_conf']:.2f}"
            put(img, f"~{r['age']:.0f} ({r['age_range']}) {g}", (x1, y), GREY if r["gender_uncertain"] else (0, 255, 255))
            y += 22
        if on[2]:
            t = f"{r['expr']} {r['expr_conf']:.2f}"
            put(img, f"? ({t})" if r["expr_uncertain"] else t, (x1, y), GREY if r["expr_uncertain"] else (255, 200, 0))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="índice da câmara ou ficheiro de vídeo")
    ap.add_argument("--detector", choices=list(DETECTORS), default="scrfd")
    ap.add_argument("--modelo", default=None, help="nome em models/ sem .onnx (p.ex. multitarefa, geral_ramo)")
    ap.add_argument("--sem-suavizacao", action="store_true")
    ap.add_argument("--galeria", choices=["media", "clusters", "vizinho"], default="clusters")
    a = ap.parse_args()
    cap = cv.VideoCapture(int(a.source) if a.source.isdigit() else a.source)
    if not cap.isOpened():
        raise SystemExit(f"não consegui abrir {a.source}")
    gal = Galeria.load(a.galeria)
    est = GeralEstimator(a.modelo)
    w = Worker(create_detector(a.detector), est, gal)
    w.smooth = not a.sem_suavizacao
    on, typing, text, enrol, msg, msg_t, paused, frame = [True] * 3, None, "", None, "", 0.0, False, None

    def say(t):
        nonlocal msg, msg_t
        msg, msg_t = t, time.time()

    def change(fn):
        """Altera uma cópia da galeria e troca-a de uma vez (a thread nunca vê um estado a meio)."""
        nonlocal gal
        g = Galeria(gal.people, gal.mode, gal.path)
        out = fn(g)
        g.save()
        gal = w.gal = g
        return out

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break
            with w.lock:
                w.frame = frame
        with w.lock:
            faces, ms = w.faces, w.ms
        if enrol and time.time() - enrol["t"] > SAMPLE_GAP:
            c = [f for f in faces if f.get("r")]
            f = max(c, key=lambda f: f["box"][2] - f["box"][0]) if c else None
            # outra pessoa a entrar em cena a meio do registo: só aceita caras coerentes com as já recolhidas
            if f is not None and f["r"] is not enrol["last"]:
                e = f["r"]["emb"]
                if not enrol["embs"] or float(np.mean(enrol["embs"], 0) @ e) > 0.35:
                    enrol["embs"].append(e)
                enrol["t"], enrol["last"] = time.time(), f["r"]
            if len(enrol["embs"]) >= N_SAMPLES:
                try:
                    change(lambda g: g.add(enrol["name"], enrol["embs"]))
                    say(f"{enrol['name']} registado")
                except ValueError as ex:
                    say(str(ex))
                enrol = None
        view = draw(frame.copy(), faces, on)
        put(view, f"SCRFD {ms[0]:.0f} ms  {est.name} {ms[1]:.0f} ms  suav. {'on' if w.smooth else 'off'}  caras {len(faces)}  galeria {len(gal.people)}", (10, 26), (255, 255, 0))
        put(view, "[n] registar [r] remover [L] limpar [1] id [2] idade/genero [3] expressao [s] suavizar [p] pausa [q] sair", (10, 48), (255, 255, 255), 0.45)
        if typing:
            put(view, f"{typing}: {text}_  (Enter confirma, Esc cancela)", (10, 80), (0, 255, 0))
        elif enrol:
            put(view, f"a registar {enrol['name']}: {len(enrol['embs'])}/{N_SAMPLES}  (mexe a cabeca; [x] cancela)", (10, 80), (0, 255, 0))
        elif time.time() - msg_t < 3:
            put(view, msg, (10, 80), (0, 255, 0))
        cv.imshow("Modelo geral", view)
        key = cv.waitKey(1) & 0xFF
        if key == 255:
            continue
        if typing:
            if key in (13, 10):
                name, text_mode = text.strip(), typing
                typing = None
                if name and text_mode == "Nome a registar":
                    enrol = {"name": name, "embs": [], "t": 0.0, "last": None}
                elif name and text_mode == "Nome a remover":
                    say("removido" if change(lambda g: g.remove(name)) else "não existe")
                elif text_mode.startswith("Limpar") and name == "SIM":
                    change(lambda g: g.clear())
                    say("galeria vazia")
            elif key == 27:
                typing = None
            elif key in (8, 127):
                text = text[:-1]
            elif 32 <= key < 127:
                text += chr(key)
            continue
        ch = chr(key)
        if ch == "q":
            break
        if ch == "x":
            enrol = None
        if ch == "p":
            paused = not paused
        if ch == "s":
            w.smooth = not w.smooth
            w.smoother.reset()
        if ch in "123":
            on[int(ch) - 1] = not on[int(ch) - 1]
        if ch in "nrL":
            typing, text = {"n": "Nome a registar", "r": "Nome a remover", "L": "Limpar tudo? escreve SIM"}[ch], ""
    w.running = False
    w.thread.join(timeout=2)
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
