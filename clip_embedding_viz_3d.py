# -*- coding: utf-8 -*-
"""
CLIP 멀티모달 임베딩 3D 시각화 (코사인 유사도 = 벡터 각도)
============================================================
clip_embedding_viz.py 가 저장한 embeddings.npz 를 읽어 3D로 시각화한다.
(없으면 먼저 clip_embedding_viz.py 를 실행할 것)

핵심 아이디어:
  코사인 유사도 = 두 단위벡터 사이 각도의 코사인.
  따라서 임베딩을 3D로 줄이고 단위 구면에 올리면, 원점에서 뻗는 벡터들
  사이의 '각도'가 곧 코사인 유사도가 된다(가까울수록 유사).

출력:
  - embedding_visualization_3d.png   (3패널 정적 이미지, 300dpi)
  - embedding_3d_interactive.html    (회전 가능한 인터랙티브 뷰, plotly 있을 때)
"""
import sys
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (3D projection 등록)

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

RANDOM_STATE = 42
NPZ = "embeddings.npz"
OUT_PNG = "embedding_visualization_3d.png"
OUT_HTML = "embedding_3d_interactive.html"

CAT_LABELS = {"cat": "고양이", "car": "자동차", "robot": "로봇 팔"}
MARKER = {"image": "o", "text": "^"}


def setup_korean_font():
    import matplotlib.font_manager as fm
    avail = {f.name for f in fm.fontManager.ttflist}
    for name in ["Malgun Gothic", "NanumGothic", "AppleGothic", "Gulim"]:
        if name in avail:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


def load():
    d = np.load(NPZ, allow_pickle=True)
    emb = d["embeddings"].astype(np.float32)
    # 안전하게 재정규화 (코사인 = 내적)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    return {
        "emb": emb,
        "names": list(d["names"]),
        "cats": list(d["categories"]),
        "mods": list(d["modalities"]),
        "colors": list(d["colors"]),
    }


def tsne3d(X, perplexity=5):
    return TSNE(n_components=3, perplexity=perplexity,
                random_state=RANDOM_STATE, init="pca",
                max_iter=2000).fit_transform(X)


def pca_unit_sphere(X):
    """PCA로 3D 축소 후 각 점을 단위 구면에 사영.
    선형 사영이라 3D상의 각도가 원본 코사인 각도를 근사한다."""
    p = PCA(n_components=3, random_state=RANDOM_STATE)
    Y = p.fit_transform(X)
    Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
    return Y, p.explained_variance_ratio_


def draw_scatter3d(ax, coords, data, title, connect_centroids=True):
    cats = list(CAT_LABELS.keys())
    for k in range(len(coords)):
        x, y, z = coords[k]
        ax.scatter(x, y, z, c=data["colors"][k],
                   marker=MARKER[data["mods"][k]],
                   s=130 if data["mods"][k] == "text" else 90,
                   edgecolors="black", linewidths=0.6, alpha=0.9, depthshade=True)
        ax.text(x, y, z, "  " + data["names"][k], fontsize=7)

    if connect_centroids:
        for c in cats:
            color = next(data["colors"][k] for k in range(len(coords))
                         if data["cats"][k] == c)
            ip = [coords[k] for k in range(len(coords))
                  if data["cats"][k] == c and data["mods"][k] == "image"]
            tp = [coords[k] for k in range(len(coords))
                  if data["cats"][k] == c and data["mods"][k] == "text"]
            if not ip or not tp:
                continue
            ic, tc = np.mean(ip, axis=0), np.mean(tp, axis=0)
            ax.plot([ic[0], tc[0]], [ic[1], tc[1]], [ic[2], tc[2]],
                    "--", color=color, linewidth=1.6, alpha=0.7)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2"); ax.set_zlabel("dim 3")


def draw_sphere_vectors(ax, coords, data, title):
    """원점에서 뻗는 단위벡터 + 반투명 단위 구면. 각도 = 코사인 유사도."""
    # 와이어프레임 단위 구
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    sx = np.outer(np.cos(u), np.sin(v))
    sy = np.outer(np.sin(u), np.sin(v))
    sz = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(sx, sy, sz, color="gray", alpha=0.12, linewidth=0.5)

    for k in range(len(coords)):
        x, y, z = coords[k]
        ax.quiver(0, 0, 0, x, y, z, color=data["colors"][k],
                  alpha=0.75, linewidth=1.6, arrow_length_ratio=0.08)
        ax.scatter(x, y, z, c=data["colors"][k], marker=MARKER[data["mods"][k]],
                   s=120 if data["mods"][k] == "text" else 80,
                   edgecolors="black", linewidths=0.6)
        ax.text(x * 1.08, y * 1.08, z * 1.08, data["names"][k], fontsize=7)

    ax.set_title(title, fontsize=11)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")


def make_static_png(data):
    emb = data["emb"]
    coords_raw = tsne3d(emb)

    # modality-gap 제거
    mods = np.array(data["mods"])
    img_mask = mods == "image"
    centered = emb.copy()
    centered[img_mask] -= emb[img_mask].mean(0, keepdims=True)
    centered[~img_mask] -= emb[~img_mask].mean(0, keepdims=True)
    coords_corr = tsne3d(centered)

    sphere, evr = pca_unit_sphere(emb)
    print(f"[pca] 3D 설명분산 비율 합 = {evr.sum():.3f} {tuple(round(float(e),3) for e in evr)}")

    fig = plt.figure(figsize=(22, 8))
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")

    draw_scatter3d(ax1, coords_raw, data,
                   "(A) 3D t-SNE — 원본\n(modality gap: 이미지·텍스트 분리)")
    draw_scatter3d(ax2, coords_corr, data,
                   "(B) 3D t-SNE — modality-gap 보정\n(같은 카테고리 군집)")
    draw_sphere_vectors(ax3, sphere, data,
                        "(C) 단위 구면 벡터 (PCA 3D)\n각도 = 코사인 유사도")

    cat_handles = [Line2D([0], [0], marker="s", color="w", label=CAT_LABELS[c],
                          markerfacecolor=next(data["colors"][k] for k in
                          range(len(emb)) if data["cats"][k] == c),
                          markersize=11, markeredgecolor="black")
                   for c in CAT_LABELS]
    mod_handles = [
        Line2D([0], [0], marker="o", color="w", label="이미지", markersize=10,
               markerfacecolor="gray", markeredgecolor="black"),
        Line2D([0], [0], marker="^", color="w", label="텍스트", markersize=11,
               markerfacecolor="gray", markeredgecolor="black"),
    ]
    fig.legend(handles=cat_handles + mod_handles, loc="lower center",
               ncol=5, fontsize=10, frameon=True)
    fig.suptitle("CLIP 멀티모달 임베딩 — 3D 시각화", fontsize=15, y=0.99)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[png] {OUT_PNG} 저장 (300dpi)")
    return coords_corr, sphere


def make_interactive_html(data, coords_corr, sphere):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:
        print(f"[html] plotly 없음 — 인터랙티브 뷰 건너뜀 ({e})")
        return

    def traces(coords, with_vectors=False):
        out = []
        for c in CAT_LABELS:
            for m in ("image", "text"):
                idx = [k for k in range(len(coords))
                       if data["cats"][k] == c and data["mods"][k] == m]
                if not idx:
                    continue
                color = data["colors"][idx[0]]
                out.append(go.Scatter3d(
                    x=[coords[k][0] for k in idx],
                    y=[coords[k][1] for k in idx],
                    z=[coords[k][2] for k in idx],
                    mode="markers+text",
                    text=[data["names"][k] for k in idx],
                    textposition="top center",
                    textfont=dict(size=9),
                    marker=dict(
                        size=6 if m == "image" else 8,
                        color=color,
                        symbol="circle" if m == "image" else "diamond",
                        line=dict(width=1, color="black")),
                    name=f"{CAT_LABELS[c]} · {'이미지' if m=='image' else '텍스트'}",
                ))
        if with_vectors:
            for k in range(len(coords)):
                x, y, z = coords[k]
                out.append(go.Scatter3d(
                    x=[0, x], y=[0, y], z=[0, z], mode="lines",
                    line=dict(color=data["colors"][k], width=4),
                    opacity=0.5, showlegend=False, hoverinfo="skip"))
        return out

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("3D t-SNE (modality-gap 보정)",
                        "단위 구면 벡터 — 각도 = 코사인 유사도"))
    for t in traces(coords_corr):
        fig.add_trace(t, row=1, col=1)
    for t in traces(sphere, with_vectors=True):
        fig.add_trace(t, row=1, col=2)

    # 단위 구 표면(반투명)
    u = np.linspace(0, 2 * np.pi, 40); v = np.linspace(0, np.pi, 20)
    sx = np.outer(np.cos(u), np.sin(v))
    sy = np.outer(np.sin(u), np.sin(v))
    sz = np.outer(np.ones_like(u), np.cos(v))
    fig.add_trace(go.Surface(x=sx, y=sy, z=sz, opacity=0.08,
                             colorscale="Greys", showscale=False,
                             hoverinfo="skip"), row=1, col=2)

    fig.update_layout(
        title="CLIP 멀티모달 임베딩 — 인터랙티브 3D (드래그하여 회전)",
        height=750, legend=dict(itemsizing="constant"))
    fig.write_html(OUT_HTML, include_plotlyjs="cdn")
    print(f"[html] {OUT_HTML} 저장 (브라우저에서 열어 회전/확대)")


def main():
    setup_korean_font()
    import os
    if not os.path.exists(NPZ):
        sys.exit(f"'{NPZ}' 가 없습니다. 먼저 clip_embedding_viz.py 를 실행하세요.")
    data = load()
    print(f"[load] {len(data['emb'])} 개 임베딩 (dim={data['emb'].shape[1]})")
    coords_corr, sphere = make_static_png(data)
    make_interactive_html(data, coords_corr, sphere)
    print("\n완료.")


if __name__ == "__main__":
    main()
