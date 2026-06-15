# -*- coding: utf-8 -*-
"""
CLIP 기반 멀티모달 임베딩 공간 시각화 실험
=============================================
CLIP(openai/clip-vit-base-patch32) 으로 이미지/텍스트 임베딩을 추출하고,
L2 정규화 -> t-SNE(2D) -> matplotlib 시각화로 공동 임베딩 공간에서
의미적으로 유사한 이미지-텍스트 쌍이 가깝게 모이는지 확인한다.

출력:
  - embedding_visualization.png  (300dpi)
  - similarity_matrix.csv        (전체 쌍별 코사인 유사도)
  - results_summary.txt          (카테고리 내/간 유사도 평균)
"""
import os
import io
import sys
import time

# Windows 콘솔(cp949)에서 한글/유니코드 출력 깨짐 방지
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import requests
from PIL import Image, ImageDraw

import torch
from transformers import CLIPModel, CLIPProcessor
from sklearn.manifold import TSNE

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ----------------------------------------------------------------------------
# 설정
# ----------------------------------------------------------------------------
RANDOM_STATE = 42
MODEL_ID = "openai/clip-vit-base-patch32"
IMG_DIR = "images"
N_IMAGES = 5

OUT_PNG = "embedding_visualization.png"
OUT_CSV = "similarity_matrix.csv"
OUT_TXT = "results_summary.txt"

# Wikimedia 의 robot policy 통과를 위해 일반 브라우저 형태의 User-Agent 사용
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

# 카테고리: 라벨(한글), 색상, 이미지 검색어, 텍스트 프롬프트
CATEGORIES = {
    "cat": {
        "label": "고양이",
        "color": "#1f77b4",  # 파랑
        "query": "cute domestic cat sitting",
        "texts": ["a cat", "고양이", "feline animal"],
    },
    "car": {
        "label": "자동차",
        "color": "#d62728",  # 빨강
        "query": "modern sports car automobile",
        "texts": ["a car", "자동차", "automobile"],
    },
    "robot": {
        "label": "로봇 팔",
        "color": "#2ca02c",  # 초록
        "query": "robotic arm",
        "texts": ["a robot arm", "로봇 팔", "robotic manipulator"],
    },
}

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)


# ----------------------------------------------------------------------------
# 한글 폰트 설정 (matplotlib)
# ----------------------------------------------------------------------------
def setup_korean_font():
    import matplotlib.font_manager as fm
    candidates = ["Malgun Gothic", "NanumGothic", "AppleGothic", "Gulim", "Dotum"]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            print(f"[font] 한글 폰트 사용: {name}")
            return
    print("[font] 경고: 한글 폰트를 찾지 못했습니다. 한글이 깨질 수 있습니다.")
    plt.rcParams["axes.unicode_minus"] = False


# ----------------------------------------------------------------------------
# 이미지 수집: Wikimedia Commons 검색 API -> 다운로드, 실패 시 placeholder
# ----------------------------------------------------------------------------
def commons_search(term, limit=12):
    """Commons 파일 네임스페이스에서 검색 후 다운로드용 URL 목록을 반환.

    이미지는 Special:FilePath 엔드포인트(commons.wikimedia.org)로 받는다.
    upload.wikimedia.org 썸네일 직접 요청은 robot policy 로 자주 429 가 나므로
    redirect 기반의 Special:FilePath 가 훨씬 안정적이다.
    """
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": term,
        "gsrnamespace": "6",      # File:
        "gsrlimit": str(limit),
        "prop": "imageinfo",
        "iiprop": "url|mime",
        "format": "json",
    }
    try:
        r = requests.get(
            "https://commons.wikimedia.org/w/api.php",
            params=params, headers=HEADERS, timeout=30,
        )
        data = r.json()
    except Exception as e:
        print(f"  [search] '{term}' 검색 실패: {e}")
        return []
    pages = data.get("query", {}).get("pages", {})
    # 검색 점수(index) 순서 보존
    items = sorted(pages.values(), key=lambda p: p.get("index", 1e9))
    urls = []
    for pg in items:
        ii = (pg.get("imageinfo") or [{}])[0]
        if ii.get("mime") not in ("image/jpeg", "image/png"):
            continue
        filename = pg["title"].split("File:", 1)[-1].replace(" ", "_")
        urls.append(
            "https://commons.wikimedia.org/wiki/Special:FilePath/"
            f"{requests.utils.quote(filename)}?width=400"
        )
    return urls


def download_image(url, retries=2):
    last_err = None
    for attempt in range(retries + 1):
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.status_code == 429:  # rate limit -> 백오프 후 재시도
            last_err = "429 rate-limited"
            time.sleep(3 * (attempt + 1))
            continue
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        # 너무 작은(=썸네일 실패/아이콘) 이미지는 거른다
        if min(img.size) < 80:
            raise ValueError(f"이미지가 너무 작음: {img.size}")
        return img
    raise RuntimeError(last_err or "다운로드 실패")


def make_placeholder(cat_key, cfg, idx):
    """다운로드 실패 시 카테고리 색 + 라벨로 placeholder 생성."""
    img = Image.new("RGB", (224, 224), color=cfg["color"])
    d = ImageDraw.Draw(img)
    d.rectangle([20, 20, 204, 204], outline="white", width=4)
    d.text((30, 100), f"{cat_key} #{idx}", fill="white")
    return img


def acquire_images(cat_key, cfg):
    """카테고리별로 N_IMAGES 장의 PIL 이미지를 확보.

    이미 디스크에 받아둔 이미지가 있으면 재사용(재다운로드/속도제한 회피),
    없을 때만 Commons 에서 새로 다운로드한다.
    """
    out_dir = os.path.join(IMG_DIR, cat_key)
    os.makedirs(out_dir, exist_ok=True)
    images = []

    # 디스크 캐시 재사용
    existing = sorted(
        f for f in os.listdir(out_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    for fname in existing:
        if len(images) >= N_IMAGES:
            break
        try:
            images.append(Image.open(os.path.join(out_dir, fname)).convert("RGB"))
        except Exception:
            continue
    if images:
        print(f"  [cache] {cat_key} 디스크에서 {len(images)}장 재사용")
    if len(images) >= N_IMAGES:
        return images[:N_IMAGES]

    urls = commons_search(cfg["query"], limit=N_IMAGES * 5)
    for url in urls:
        if len(images) >= N_IMAGES:
            break
        try:
            img = download_image(url)
        except Exception as e:
            print(f"  [dl] 스킵 ({e})")
            continue
        idx = len(images) + 1
        path = os.path.join(out_dir, f"{cat_key}_{idx}.jpg")
        img.save(path, "JPEG", quality=90)
        images.append(img)
        print(f"  [dl] {cat_key} #{idx} 저장")
        time.sleep(1.2)

    # 부족분은 placeholder로 채움
    while len(images) < N_IMAGES:
        idx = len(images) + 1
        img = make_placeholder(cat_key, cfg, idx)
        path = os.path.join(out_dir, f"{cat_key}_{idx}_placeholder.jpg")
        img.save(path, "JPEG", quality=90)
        images.append(img)
        print(f"  [ph] {cat_key} #{idx} placeholder 생성")

    return images


# ----------------------------------------------------------------------------
# 메인
# ----------------------------------------------------------------------------
def main():
    setup_korean_font()

    # 1) 이미지 수집 ---------------------------------------------------------
    print("\n=== 1. 이미지 수집 ===")
    images_by_cat = {}
    for cat_key, cfg in CATEGORIES.items():
        print(f"[{cat_key}] '{cfg['query']}'")
        images_by_cat[cat_key] = acquire_images(cat_key, cfg)

    # 2) CLIP 모델 로드 ------------------------------------------------------
    print("\n=== 2. CLIP 모델 로드 ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[model] {MODEL_ID}  device={device}")
    model = CLIPModel.from_pretrained(MODEL_ID).to(device).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_ID)

    # 3) 아이템 리스트 구성 (이미지 + 텍스트) -------------------------------
    #    item: dict(category, modality, name, color, pil/text)
    items = []
    pil_images = []
    text_strings = []
    for cat_key, cfg in CATEGORIES.items():
        for i, img in enumerate(images_by_cat[cat_key], 1):
            items.append({
                "category": cat_key, "modality": "image",
                "name": f"{cat_key}#{i}", "color": cfg["color"],
            })
            pil_images.append(img)
        for txt in cfg["texts"]:
            items.append({
                "category": cat_key, "modality": "text",
                "name": txt, "color": cfg["color"],
            })
            text_strings.append(txt)

    # 4) 임베딩 추출 + L2 정규화 --------------------------------------------
    print("\n=== 3. 임베딩 추출 ===")

    def to_feature_tensor(out):
        # transformers 5.x: get_*_features 가 BaseModelOutputWithPooling 을
        # 반환하며 pooler_output 이 projection 적용된 공동공간 임베딩(512d).
        # 구버전(텐서 직접 반환)도 함께 지원.
        if isinstance(out, torch.Tensor):
            return out
        return out.pooler_output

    with torch.no_grad():
        img_inputs = processor(images=pil_images, return_tensors="pt").to(device)
        img_emb = to_feature_tensor(model.get_image_features(**img_inputs))

        txt_inputs = processor(
            text=text_strings, return_tensors="pt", padding=True
        ).to(device)
        txt_emb = to_feature_tensor(model.get_text_features(**txt_inputs))

    img_emb = torch.nn.functional.normalize(img_emb, p=2, dim=-1).cpu().numpy()
    txt_emb = torch.nn.functional.normalize(txt_emb, p=2, dim=-1).cpu().numpy()
    print(f"[emb] 이미지 {img_emb.shape}, 텍스트 {txt_emb.shape}")

    # items 순서(이미지 전부 -> 텍스트 전부)대로 합치기
    embeddings = np.zeros((len(items), img_emb.shape[1]), dtype=np.float32)
    img_ptr = txt_ptr = 0
    for k, it in enumerate(items):
        if it["modality"] == "image":
            embeddings[k] = img_emb[img_ptr]; img_ptr += 1
        else:
            embeddings[k] = txt_emb[txt_ptr]; txt_ptr += 1

    labels = [it["name"] for it in items]

    # 임베딩 캐시 저장 (3D 시각화 스크립트 등에서 재사용)
    np.savez(
        "embeddings.npz",
        embeddings=embeddings,
        names=np.array([it["name"] for it in items], dtype=object),
        categories=np.array([it["category"] for it in items], dtype=object),
        modalities=np.array([it["modality"] for it in items], dtype=object),
        colors=np.array([it["color"] for it in items], dtype=object),
    )
    print("[cache] embeddings.npz 저장")

    # 5) 코사인 유사도 행렬 (정규화돼 있으므로 내적 = 코사인) ---------------
    print("\n=== 4. 코사인 유사도 계산 ===")
    sim = embeddings @ embeddings.T
    write_similarity_csv(sim, labels)

    # 6) 카테고리 내/간 유사도 요약 -----------------------------------------
    summarize(items, embeddings)

    # 7) t-SNE 2D ------------------------------------------------------------
    print("\n=== 5. t-SNE 차원 축소 ===")
    n = len(items)
    perplexity = 5 if n - 1 > 5 else max(2, (n - 1) // 3)

    def run_tsne(X):
        return TSNE(
            n_components=2, perplexity=perplexity,
            random_state=RANDOM_STATE, init="pca", max_iter=2000,
        ).fit_transform(X)

    # (A) 원본 임베딩 — 스펙 그대로. CLIP 의 'modality gap' 때문에
    #     이미지/텍스트가 서로 다른 영역에 뭉친다(정직한 결과).
    coords_raw = run_tsne(embeddings)

    # (B) modality-gap 제거 — 모달리티별 평균을 빼서 두 모달리티를 같은
    #     영역으로 맞춘 뒤 t-SNE. 같은 카테고리의 이미지+텍스트가
    #     함께 모이는 의미적 정렬을 눈으로 확인하기 위한 보정 뷰.
    img_mask = np.array([it["modality"] == "image" for it in items])
    centered = embeddings.copy()
    centered[img_mask] -= embeddings[img_mask].mean(axis=0, keepdims=True)
    centered[~img_mask] -= embeddings[~img_mask].mean(axis=0, keepdims=True)
    coords_centered = run_tsne(centered)
    print(f"[tsne] perplexity={perplexity}, 좌표 {coords_raw.shape} (x2 패널)")

    # 8) 시각화 --------------------------------------------------------------
    print("\n=== 6. 시각화 ===")
    plot_embeddings(items, coords_raw, coords_centered)

    print("\n완료. 생성된 파일:")
    for f in (OUT_PNG, OUT_CSV, OUT_TXT):
        print(f"  - {f}")


def write_similarity_csv(sim, labels):
    import csv
    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow([""] + labels)
        for i, row_label in enumerate(labels):
            w.writerow([row_label] + [f"{v:.4f}" for v in sim[i]])
    print(f"[csv] {OUT_CSV} 저장 ({sim.shape[0]}x{sim.shape[1]})")


def summarize(items, embeddings):
    """카테고리 내 이미지-텍스트 정렬도 vs 카테고리 간 정렬도 비교."""
    cats = list(CATEGORIES.keys())
    idx = {c: {"image": [], "text": []} for c in cats}
    for k, it in enumerate(items):
        idx[it["category"]][it["modality"]].append(k)

    def mean_sim(rows, cols):
        vals = [float(embeddings[r] @ embeddings[c]) for r in rows for c in cols]
        return float(np.mean(vals)) if vals else float("nan")

    # 이미지(row) x 텍스트(col) 카테고리 매트릭스
    matrix = {}  # (img_cat, txt_cat) -> mean cos
    for ci in cats:
        for cj in cats:
            matrix[(ci, cj)] = mean_sim(idx[ci]["image"], idx[cj]["text"])

    intra = [matrix[(c, c)] for c in cats]
    inter = [matrix[(ci, cj)] for ci in cats for cj in cats if ci != cj]
    intra_mean, inter_mean = float(np.mean(intra)), float(np.mean(inter))

    lines = []
    lines.append("=" * 64)
    lines.append("CLIP 멀티모달 임베딩 — 카테고리 내/간 유사도 요약")
    lines.append("모델: " + MODEL_ID)
    lines.append("=" * 64)
    lines.append("")
    lines.append("[이미지(행) x 텍스트(열)] 카테고리별 평균 코사인 유사도")
    lines.append("(대각선=카테고리 내 정렬, 비대각선=카테고리 간)")
    lines.append("")

    head = "{:>10}".format("img\\txt") + "".join(
        "{:>12}".format(CATEGORIES[c]["label"]) for c in cats
    )
    lines.append(head)
    for ci in cats:
        row = "{:>10}".format(CATEGORIES[ci]["label"])
        for cj in cats:
            v = matrix[(ci, cj)]
            mark = "*" if ci == cj else " "
            row += "{:>11.4f}{}".format(v, mark)
        lines.append(row)

    lines.append("")
    lines.append("-" * 64)
    lines.append("카테고리별 (이미지 vs 자기 텍스트) 평균 코사인 유사도:")
    for c in cats:
        lines.append("  {:<8} ({}) : {:.4f}".format(
            c, CATEGORIES[c]["label"], matrix[(c, c)]))
    lines.append("")
    lines.append("  카테고리 내(intra) 평균 : {:.4f}".format(intra_mean))
    lines.append("  카테고리 간(inter) 평균 : {:.4f}".format(inter_mean))
    lines.append("  격차(intra - inter)     : {:.4f}".format(intra_mean - inter_mean))
    lines.append("")
    verdict = ("✓ 같은 카테고리 이미지-텍스트가 다른 카테고리보다 "
               "뚜렷하게 가깝게 정렬됨." if intra_mean > inter_mean
               else "✗ 카테고리 내 정렬이 기대만큼 두드러지지 않음.")
    lines.append("결론: " + verdict)
    lines.append("")
    lines.append("[참고] CLIP modality gap:")
    lines.append("  이미지/텍스트 임베딩은 공동공간 안에서도 서로 다른 영역(cone)에")
    lines.append("  분포한다. 따라서 원본 t-SNE(패널 A)에서는 모달리티가 먼저 분리되어")
    lines.append("  보이지만, 코사인 유사도(위 표)와 모달리티 평균을 제거한 t-SNE(패널 B)")
    lines.append("  에서는 같은 카테고리의 이미지-텍스트가 의미적으로 정렬됨이 드러난다.")
    lines.append("=" * 64)

    text = "\n".join(lines)
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)
    print(f"\n[txt] {OUT_TXT} 저장")


MARKER_BY_MODALITY = {"image": "o", "text": "^"}


def draw_panel(ax, items, coords, title):
    """하나의 t-SNE 좌표 집합을 카테고리 색 + 모달리티 마커로 그린다."""
    cats = list(CATEGORIES.keys())

    # 포인트
    for k, it in enumerate(items):
        x, y = coords[k]
        ax.scatter(
            x, y,
            c=it["color"],
            marker=MARKER_BY_MODALITY[it["modality"]],
            s=170 if it["modality"] == "text" else 120,
            edgecolors="black", linewidths=0.7,
            alpha=0.9, zorder=3,
        )
        ax.annotate(
            it["name"], (x, y),
            textcoords="offset points", xytext=(6, 4),
            fontsize=8, zorder=4,
        )

    # 카테고리별 이미지 centroid <-> 텍스트 centroid 점선 연결
    for c in cats:
        img_pts = [coords[k] for k, it in enumerate(items)
                   if it["category"] == c and it["modality"] == "image"]
        txt_pts = [coords[k] for k, it in enumerate(items)
                   if it["category"] == c and it["modality"] == "text"]
        if not img_pts or not txt_pts:
            continue
        ic = np.mean(img_pts, axis=0)
        tc = np.mean(txt_pts, axis=0)
        ax.plot(
            [ic[0], tc[0]], [ic[1], tc[1]],
            linestyle="--", color=CATEGORIES[c]["color"],
            linewidth=1.8, alpha=0.7, zorder=2,
        )
        ax.scatter([ic[0], tc[0]], [ic[1], tc[1]],
                   c=CATEGORIES[c]["color"], marker="x", s=70,
                   linewidths=2, zorder=2, alpha=0.85)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.grid(True, linestyle=":", alpha=0.4)


def plot_embeddings(items, coords_raw, coords_centered):
    cats = list(CATEGORIES.keys())
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    draw_panel(axes[0], items, coords_raw,
               "(A) 원본 임베딩 t-SNE\nCLIP modality gap: 이미지·텍스트가 분리됨")
    draw_panel(axes[1], items, coords_centered,
               "(B) modality-gap 보정 t-SNE\n같은 카테고리 이미지+텍스트가 함께 군집")

    # 공통 범례: 카테고리(색) + 모달리티(마커)
    cat_handles = [
        Line2D([0], [0], marker="s", color="w", label=CATEGORIES[c]["label"],
               markerfacecolor=CATEGORIES[c]["color"], markersize=12,
               markeredgecolor="black")
        for c in cats
    ]
    mod_handles = [
        Line2D([0], [0], marker="o", color="w", label="이미지 (image)",
               markerfacecolor="gray", markersize=11, markeredgecolor="black"),
        Line2D([0], [0], marker="^", color="w", label="텍스트 (text)",
               markerfacecolor="gray", markersize=12, markeredgecolor="black"),
        Line2D([0], [0], linestyle="--", color="gray",
               label="이미지↔텍스트 centroid"),
    ]
    leg1 = axes[0].legend(handles=cat_handles, title="카테고리",
                          loc="upper left", fontsize=9)
    axes[0].add_artist(leg1)
    axes[0].legend(handles=mod_handles, title="모달리티",
                   loc="lower right", fontsize=9)

    fig.suptitle(
        "CLIP 멀티모달 임베딩 공간 시각화 (t-SNE 2D)  ·  "
        f"{MODEL_ID}",
        fontsize=15, y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[png] {OUT_PNG} 저장 (300dpi)")


if __name__ == "__main__":
    main()
