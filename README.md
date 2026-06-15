# CLIP 멀티모달 임베딩 공간 시각화 실험

CLIP(`openai/clip-vit-base-patch32`)으로 이미지와 텍스트의 임베딩을 추출하고,
공동 임베딩 공간에서 **의미적으로 유사한 이미지–텍스트 쌍이 가깝게 군집**하는지를
t-SNE 시각화와 코사인 유사도로 검증한다.

## 카테고리

| 카테고리 | 이미지 | 텍스트 프롬프트 | 색상 |
|---|---|---|---|
| 고양이 (cat) | 5장 | `a cat`, `고양이`, `feline animal` | 파랑 |
| 자동차 (car) | 5장 | `a car`, `자동차`, `automobile` | 빨강 |
| 로봇 팔 (robot) | 5장 | `a robot arm`, `로봇 팔`, `robotic manipulator` | 초록 |

이미지는 Wikimedia Commons 검색 API로 카테고리별 실제 사진을 자동 다운로드하며,
다운로드 실패 시 PIL placeholder로 대체한다.

## 파이프라인

1. **임베딩 추출** — CLIP 이미지/텍스트 인코더, L2 정규화 (공동공간 512차원)
2. **코사인 유사도** — 정규화 벡터의 내적
3. **차원 축소** — `sklearn` t-SNE (`random_state=42`)
4. **시각화** — matplotlib (카테고리=색, 이미지=`o`/텍스트=`^`, centroid 점선 연결)

## 실행

### Python

```bash
pip install -r requirements.txt

# 2D 시각화 + 유사도 분석 (embeddings.npz 생성)
python clip_embedding_viz.py

# 3D 시각화 (위 결과 캐시를 재사용)
python clip_embedding_viz_3d.py
```

### MATLAB (3D 시뮬레이션 / 회전 애니메이션)

CLIP 임베딩 계산은 Python이 담당하고, MATLAB은 `.mat` 데이터를 읽어
3D 렌더링과 회전 애니메이션을 그린다 (Statistics & ML Toolbox 불필요 —
t-SNE/PCA 3D 좌표는 Python에서 미리 계산해 저장).

```bash
# 1) 임베딩 캐시 생성 (아직 안 했다면)
python clip_embedding_viz.py
# 2) MATLAB용 데이터 내보내기 -> embeddings.mat
python export_for_matlab.py
```

```matlab
% 3) MATLAB에서 실행
>> clip_embedding_viz_3d
```

## 출력

| 파일 | 설명 |
|---|---|
| `embedding_visualization.png` | 2D t-SNE — (A) 원본 / (B) modality-gap 보정 |
| `embedding_visualization_3d.png` | 3D (Python) — t-SNE 원본/보정 + 단위 구면 벡터 뷰 |
| `embedding_3d_interactive.html` | 회전·확대 가능한 인터랙티브 3D (Plotly) |
| `embedding_visualization_3d_matlab.png` | 3D (MATLAB) — 다크 테마 2-패널, 300dpi |
| `clip_embedding_3d_matlab.gif` | MATLAB 단위 구면 벡터 회전 애니메이션 |
| `similarity_matrix.csv` | 24×24 전체 쌍별 코사인 유사도 |
| `results_summary.txt` | 카테고리 내/간 유사도 평균 요약 |

## 결과 요약

이미지(행) × 텍스트(열) 카테고리별 평균 코사인 유사도 (대각선 = 같은 카테고리):

```
            고양이    자동차   로봇 팔
  고양이   0.2486*  0.2016   0.1896
  자동차   0.1727   0.2359*  0.1862
  로봇 팔   0.1802   0.1936   0.2455*
```

- 카테고리 **내** 평균 `0.2433` vs 카테고리 **간** 평균 `0.1873` (격차 **+0.056**)
- 모든 카테고리에서 자기 텍스트와의 유사도가 행 내 최댓값 → 이미지–텍스트 의미 정렬 성립

### 참고: CLIP modality gap

이미지/텍스트 임베딩은 공동공간 안에서도 서로 다른 영역(cone)에 분포한다.
따라서 원본 t-SNE에서는 모달리티가 먼저 분리되어 보이지만, 코사인 유사도와
모달리티 평균을 제거한 보정 t-SNE에서는 같은 카테고리의 이미지–텍스트가
의미적으로 정렬됨이 드러난다.

코사인 유사도 = 두 단위벡터 사이 각도의 코사인이므로, 3D 단위 구면 벡터 뷰에서는
같은 카테고리 벡터들이 비슷한 방향(작은 각도 = 높은 유사도)을 가리킨다.
