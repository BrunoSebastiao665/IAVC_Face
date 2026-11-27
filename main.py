import argparse
import os
import threading
import time

import cv2 as cv

from deteccao.face_detector import DETECTORS, FaceDetector, create_detector
from combinacoes import AGE_GENDER, COMBOS, EXPRESSION, OFF, ModelPool, menu
from expressao.expression import ExpressionEstimator, app_config
from idade_genero.age_gender import AgeGenderEstimator


class AsyncDetector:
    """Corre o detetor numa thread à parte para o vídeo não ficar lento.

    O loop principal mostra sempre o frame mais recente com as últimas caras conhecidas;
    a thread vai buscando o frame mais recente e atualiza as caras quando termina.
    """

    def __init__(self, detector, age_gender=None, expression=None):
        self.detector = detector
        self.expression = expression      # ExpressionEstimator opcional (corre na mesma thread)
        self.age_gender = age_gender      # AgeGenderEstimator opcional (corre na mesma thread)
        self.lock = threading.Lock()
        self.frame = None
        self.faces = []
        self.det_fps = 0.0
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def submit(self, frame):
        with self.lock:
            self.frame = frame

    def results(self):
        with self.lock:
            return self.faces, self.det_fps

    def _loop(self):
        while self.running:
            with self.lock:
                frame, self.frame = self.frame, None
            if frame is None:
                time.sleep(0.005)
                continue
            start = time.time()
            faces = self.detector.detect(frame)
            age_gender = self.age_gender      # referência local: as teclas podem trocá-lo a meio
            if age_gender is not None:
                for face, attr in zip(faces, age_gender.predict(frame, faces)):
                    face["attrs"] = attr
            expression = self.expression
            if expression is not None:
                for face, ex in zip(faces, expression.predict(frame, faces)):
                    face["expr"] = ex
            with self.lock:
                self.faces = faces
                self.det_fps = 1 / (time.time() - start)

    def stop(self):
        self.running = False
        self.thread.join(timeout=2)


parser = argparse.ArgumentParser(description="Deteção de caras, idade/género e expressão na webcam")
parser.add_argument("--model", choices=list(DETECTORS), default="scrfd",
                    help="detetor a usar (por omissão: scrfd)")
parser.add_argument("--camera", type=int, default=0, help="índice da câmara")
parser.add_argument("--menu", action="store_true", help="escolher os modelos num menu ao arrancar")
parser.add_argument("--combo", choices=list(COMBOS), default=None,
                    help="combinação pronta: " + "; ".join(f"{k} = {a} + {e}" for k, (a, e, _) in COMBOS.items()))
parser.add_argument("--age-gender", nargs="?", const="fairface", default=None, metavar="MODELO",
                    help="idade aparente e género: " + ", ".join(AGE_GENDER) + " (por omissão fairface)")
parser.add_argument("--expression", nargs="?", const="auto", default=None, choices=["auto", *EXPRESSION], metavar="MODELO",
                    help="expressão facial visível, não a emoção sentida: " + ", ".join(EXPRESSION) +
                         " (sem valor = o escolhido no AffectNet; por omissão desligada)")
parser.add_argument("--list", action="store_true", help="listar modelos e combinações e sair")
args = parser.parse_args()

if args.list:
    print("--combo"); [print(f"  {k:<18} {a} + {e}  —  {d}") for k, (a, e, d) in COMBOS.items()]
    print("--age-gender"); [print(f"  {k:<18} {d}") for k, d in AGE_GENDER.items()]
    print("--expression"); [print(f"  {k:<18} {d}") for k, d in EXPRESSION.items()]
    raise SystemExit
if args.menu:
    ag_name, ex_name = menu()
else:                                    # --age-gender / --expression têm prioridade sobre --combo
    base = COMBOS[args.combo][:2] if args.combo else ("fairface", OFF)
    ag_name, ex_name = args.age_gender or base[0], args.expression or base[1]
if ex_name == "auto":
    ex_name = app_config()["modelo"]

pool = ModelPool()
try:
    age_gender, expression = pool.age_gender(ag_name), pool.expression(ex_name)
except FileNotFoundError as e:
    parser.error(str(e))
print(f"Detetor: {args.model} | idade/género: {ag_name} | expressão: {ex_name}")
AG_CYCLE, EX_CYCLE, COMBO_CYCLE = list(AGE_GENDER), list(EXPRESSION), list(COMBOS.items())
topk = 1
detector = AsyncDetector(create_detector(args.model), age_gender, expression)


def switch(new_ag, new_ex):
    """Troca os modelos com a app a correr (a primeira vez que um modelo é usado, demora a carregar)."""
    global ag_name, ex_name
    try:
        detector.age_gender, detector.expression = pool.age_gender(new_ag), pool.expression(new_ex)
        ag_name, ex_name = new_ag, new_ex
        print(f"Idade/género: {ag_name} | expressão: {ex_name}")
    except FileNotFoundError as e:
        print(e)


def following(options, current):
    return options[(options.index(current) + 1) % len(options)] if current in options else options[0]


cap = cv.VideoCapture(args.camera)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

prev = time.time()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Our operations on the frame come here
    detector.submit(frame)
    faces, det_fps = detector.results()
    displayframe = FaceDetector.draw(frame.copy(), faces)
    for face in faces:
        if "attrs" in face:
            x1, _, _, y2 = face["box"]
            color = (160, 160, 160) if face["attrs"]["gender_uncertain"] else (0, 255, 255)
            cv.putText(displayframe, AgeGenderEstimator.label(face["attrs"]), (x1, y2 + 20),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if face.get("expr"):
            x1, _, _, y2 = face["box"]
            color = (160, 160, 160) if face["expr"]["uncertain"] else (255, 200, 0)
            cv.putText(displayframe, ExpressionEstimator.label(face["expr"], topk), (x1, y2 + (44 if "attrs" in face else 20)),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    now = time.time()
    fps = 1 / max(now - prev, 1e-6)
    prev = now
    cv.putText(displayframe, f"{args.model} | Video: {fps:.0f} FPS  Detecao: {det_fps:.1f} FPS  Caras: {len(faces)}",
               (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv.putText(displayframe, f"[g] idade/genero: {ag_name}   [e] expressao: {ex_name}",
               (10, 58), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv.putText(displayframe, "[c] combinacao   [t] top-3   [q] sair", (10, 82), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # Display the resulting frame
    cv.imshow('frame', displayframe)
    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('t'):
        topk = 3 if topk == 1 else 1
    if key == ord('g'):
        switch(following(AG_CYCLE, ag_name), ex_name)
    if key == ord('e'):
        switch(ag_name, following(EX_CYCLE, ex_name))
    if key == ord('c'):
        pairs = [v[:2] for _, v in COMBO_CYCLE]
        switch(*following(pairs, (ag_name, ex_name)))

# When everything done, release the capture
detector.stop()
cap.release()
cv.destroyAllWindows()
