# -*- coding: utf-8 -*-
"""
MATLAB 3D 시각화를 위한 데이터 내보내기
=========================================
clip_embedding_viz.py 가 저장한 embeddings.npz 를 읽어, MATLAB 에서 바로
쓸 수 있는 embeddings.mat 으로 변환한다.

MATLAB 에 Statistics & ML Toolbox(tsne/pca)가 없어도 .m 스크립트가 그대로
동작하도록, t-SNE/PCA 3D 좌표를 여기서 미리 계산해 함께 저장한다.
"""
import sys
import numpy as np
from scipy.io import savemat
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

RANDOM_STATE = 42
NPZ = "embeddings.npz"
OUT_MAT = "embeddings.mat"

CAT_KEYS = ["cat", "car", "robot"]
CAT_LABELS = {"cat": "고양이", "car": "자동차", "robot": "로봇 팔"}


def hex2rgb(h):
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]


def tsne3d(X, perplexity=5):
    return TSNE(n_components=3, perplexity=perplexity, random_state=RANDOM_STATE,
                init="pca", max_iter=2000).fit_transform(X).astype(np.float64)


def main():
    import os
    if not os.path.exists(NPZ):
        sys.exit(f"'{NPZ}' 없음 — 먼저 clip_embedding_viz.py 실행")

    d = np.load(NPZ, allow_pickle=True)
    emb = d["embeddings"].astype(np.float64)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    names = [str(x) for x in d["names"]]
    cats = [str(x) for x in d["categories"]]
    mods = [str(x) for x in d["modalities"]]
    colors = [hex2rgb(str(x)) for x in d["colors"]]

    # (A) 원본 t-SNE 3D
    tsne_raw = tsne3d(emb)

    # (B) modality-gap 보정 t-SNE 3D
    mods_arr = np.array(mods)
    img_mask = mods_arr == "image"
    centered = emb.copy()
    centered[img_mask] -= emb[img_mask].mean(0, keepdims=True)
    centered[~img_mask] -= emb[~img_mask].mean(0, keepdims=True)
    tsne_corr = tsne3d(centered)

    # (C) PCA -> 단위 구면 (각도 = 코사인 유사도)
    pca = PCA(n_components=3, random_state=RANDOM_STATE)
    sphere = pca.fit_transform(emb)
    sphere = sphere / np.linalg.norm(sphere, axis=1, keepdims=True)
    evr = pca.explained_variance_ratio_.astype(np.float64)

    # 512차원 원본 코사인 유사도 행렬 (시각화/검증용)
    cosmat = (emb @ emb.T).astype(np.float64)

    savemat(OUT_MAT, {
        "emb": emb,
        "tsne_raw": tsne_raw,
        "tsne_corr": tsne_corr,
        "sphere": sphere,
        "evr": evr,
        "cosmat": cosmat,
        "names": np.array(names, dtype=object),
        "categories": np.array(cats, dtype=object),
        "modalities": np.array(mods, dtype=object),
        "colors": np.array(colors, dtype=np.float64),
        "cat_keys": np.array(CAT_KEYS, dtype=object),
        "cat_labels": np.array([CAT_LABELS[c] for c in CAT_KEYS], dtype=object),
    }, do_compression=True)

    print(f"[mat] {OUT_MAT} 저장")
    print(f"  임베딩 {emb.shape}, PCA 3D 설명분산 {evr.sum():.3f} "
          f"{tuple(round(float(e), 3) for e in evr)}")


if __name__ == "__main__":
    main()
