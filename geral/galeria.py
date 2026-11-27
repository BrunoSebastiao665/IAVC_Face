"""Galeria de pessoas conhecidas (menos de 20, com consentimento). Guarda SÓ descritores e nomes, nunca imagens."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from expressao.comum import ROOT                      # noqa: E402

PATH = os.path.join(ROOT, "data", "galeria.npz")
UNKNOWN = "Unknown"
MAX_PEOPLE = 19


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def clusters(E, k_max=4, per=5, iters=20, seed=0):
    """k-means esférico (cosseno) dos descritores de uma pessoa -> K protótipos unitários [K,512].
    K = min(k_max, N // per): cada protótipo resume pelo menos ~5 amostras (pose / expressão / distância parecidas)."""
    E = unit(np.asarray(E, np.float32))
    k = max(1, min(k_max, len(E) // per))
    if k == 1:
        return unit(E.mean(0))[None]
    rng = np.random.RandomState(seed)
    C = [E[rng.randint(len(E))]]
    for _ in range(1, k):                                   # k-means++: começa por amostras afastadas
        d = np.clip(1 - np.max(E @ np.array(C).T, 1), 0, None)   # arredondamentos podem dar < 0
        C.append(E[rng.choice(len(E), p=d / d.sum()) if d.sum() > 0 else rng.randint(len(E))])
    C = np.array(C)
    for _ in range(iters):
        a = (E @ C.T).argmax(1)
        C = unit(np.stack([E[a == j].mean(0) if (a == j).any() else C[j] for j in range(k)]))
    return C


class Galeria:
    def __init__(self, people=None, mode="media", path=PATH):
        """people: {nome: descritores [N,512]}. mode: 'media' (um template por pessoa), 'vizinho' (o mais parecido
        de todas as amostras) ou 'clusters' (K protótipos por pessoa, notas da aula; score = o protótipo mais parecido)."""
        self.people, self.mode, self.path = dict(people or {}), mode, path
        self._build()

    def _build(self):
        self.names = list(self.people)
        if self.names:
            T = [unit(self.people[n].mean(0))[None] if self.mode == "media" else clusters(self.people[n])
                 if self.mode == "clusters" else self.people[n] for n in self.names]
            self.T, self.owner = np.concatenate(T), np.repeat(np.arange(len(T)), [len(t) for t in T])

    @classmethod
    def load(cls, mode="media", path=PATH):
        if not os.path.exists(path):
            return cls(mode=mode, path=path)
        with np.load(path) as d:
            return cls({k: d[k] for k in d.files}, mode, path)

    def save(self):
        if self.people:
            np.savez(self.path, **self.people)
            os.chmod(self.path, 0o600)
        elif os.path.exists(self.path):
            os.remove(self.path)

    def add(self, name, embs):
        name = name.strip()
        if not name or name == UNKNOWN or "/" in name:
            raise ValueError(f"nome inválido: {name!r}")
        if name not in self.people and len(self.people) >= MAX_PEOPLE:
            raise ValueError("a galeria é limitada a menos de 20 pessoas")
        embs = unit(np.atleast_2d(np.asarray(embs, np.float32)))
        self.people[name] = np.concatenate([self.people[name], embs]) if name in self.people else embs
        self._build()

    def remove(self, name):
        found = self.people.pop(name, None) is not None
        self._build()
        return found

    def clear(self):
        self.people = {}
        self._build()

    def scores(self, E):
        """[N,512] -> similaridade com cada pessoa [N,P] (cosseno; máximo por pessoa nos modos 'vizinho' e 'clusters')."""
        S = np.atleast_2d(E) @ self.T.T
        return np.stack([S[:, self.owner == p].max(1) for p in range(len(self.names))], 1)

    def match(self, e, thr):
        if not self.people:
            return UNKNOWN, 0.0
        s = self.scores(e)[0]
        p = int(s.argmax())
        return (self.names[p] if s[p] >= thr else UNKNOWN), float(s[p])
