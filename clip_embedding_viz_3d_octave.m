function clip_embedding_viz_3d_octave()
% =========================================================================
% CLIP 멀티모달 임베딩 3D 시각화 — GNU Octave 호환 버전
% =========================================================================
% MATLAB 전용 함수(tiledlayout/exportgraphics/MarkerFaceAlpha)를 피하고
% subplot/print 로 작성. Octave 의 CJK 렌더링 한계 때문에 라벨은 로마자화.
%
% 실행:  octave-cli --eval clip_embedding_viz_3d_octave
% (먼저 Python: clip_embedding_viz.py -> export_for_matlab.py)
%
% 출력:
%   - embedding_visualization_3d_octave.png
%   - clip_embedding_3d_octave.gif   (가능할 때)
% =========================================================================

matfile = 'embeddings.mat';
if ~exist(matfile, 'file')
  error('%s 없음. 먼저 python clip_embedding_viz.py && python export_for_matlab.py', matfile);
end
S = load(matfile);

% qt 툴킷은 오프스크린 렌더링 가능 -> 보이지 않는 figure 로 print/getframe.
% qt 가 없으면 fltk 로 폴백하되, fltk 는 print 에 보이는 figure 가 필요.
if any(strcmp(available_graphics_toolkits(), 'qt'))
  graphics_toolkit('qt');
  vis = 'off';
else
  vis = 'on';
end
printf('[octave] toolkit=%s visible=%s\n', graphics_toolkit(), vis);

BG = [0.08 0.09 0.12];
FG = [0.92 0.93 0.96];
isImg = strcmp(S.modalities(:), 'image');
% Python 에서 만든 ASCII 표시명 사용 (+ 혹시 모를 비ASCII는 바이트 단위로 제거)
src = S.disp_names;
disp_names = cell(numel(src), 1);
for k = 1:numel(src)
  disp_names{k} = ascii_only(src{k});
end

% ---- 정적 2-패널 PNG ----------------------------------------------------
fig = figure('Color', BG, 'Position', [60 60 1480 700], 'Visible', vis);

ax1 = subplot(1, 2, 1);
draw_scatter3(ax1, S.tsne_corr, S, isImg, disp_names, BG, FG, true);
title(ax1, '(A) modality-gap corrected t-SNE 3D', 'Color', FG);

ax2 = subplot(1, 2, 2);
draw_sphere3(ax2, S.sphere, S, isImg, disp_names, BG, FG);
title(ax2, sprintf('(B) PCA unit-sphere vectors  (angle = cosine sim, var %.0f%%)', ...
      100*sum(S.evr)), 'Color', FG);

add_legend(ax2, S, FG, BG);

drawnow;
% fltk 의 print 는 투명 3D 를 GL2PS(벡터)로 처리해 극단적으로 느리다.
% getframe 로 렌더된 래스터를 직접 캡처해 저장(빠르고 GL2PS 회피).
F = getframe(fig);
imwrite(F.cdata, 'embedding_visualization_3d_octave.png');
printf('[png] embedding_visualization_3d_octave.png 저장\n');

% ---- 회전 GIF (단위 구면 뷰) -------------------------------------------
try
  make_spin_gif(S, isImg, disp_names, BG, FG, vis, 'clip_embedding_3d_octave.gif');
  printf('[gif] clip_embedding_3d_octave.gif 저장\n');
catch err
  printf('[gif] 건너뜀 (%s)\n', err.message);
end
printf('완료.\n');

end  % ===== main function 끝 =====


% =========================================================================
function r = ascii_only(s)
  b = double(s);
  b = b(b >= 32 & b < 127);   % 출력 가능한 ASCII 만 유지
  r = char(b);
  if isempty(r), r = '?'; end
end

function col = cat_color(S, ck)
  idx = find(strcmp(S.categories(:), ck), 1);
  col = S.colors(idx, :);
end

function style_axes(ax, BG, FG)
  set(ax, 'Color', BG, 'XColor', FG, 'YColor', FG, 'ZColor', FG, ...
      'GridColor', [0.5 0.5 0.55], 'FontSize', 9);
  grid(ax, 'on');
end

function draw_scatter3(ax, C, S, isImg, names, BG, FG, connect)
  hold(ax, 'on'); style_axes(ax, BG, FG);
  for k = 1:size(C,1)
    if isImg(k), mk = 'o'; ms = 7; else, mk = '^'; ms = 9; end
    % plot3 의 MarkerFaceColor 는 RGB 를 정확히 적용(scatter3 의 colormap 오해 회피)
    plot3(ax, C(k,1), C(k,2), C(k,3), 'LineStyle', 'none', 'Marker', mk, ...
          'MarkerSize', ms, 'MarkerFaceColor', S.colors(k,:), 'MarkerEdgeColor', 'k');
    text(ax, C(k,1), C(k,2), C(k,3), ['  ' names{k}], 'Color', FG, 'FontSize', 7);
  end
  if connect
    for ci = 1:numel(S.cat_keys)
      ck = S.cat_keys{ci};
      sel = strcmp(S.categories(:), ck);
      ii = sel & isImg; tt = sel & ~isImg;
      if ~any(ii) || ~any(tt), continue; end
      ic = mean(C(ii,:),1); tc = mean(C(tt,:),1);
      plot3(ax, [ic(1) tc(1)], [ic(2) tc(2)], [ic(3) tc(3)], '--', ...
            'Color', cat_color(S, ck), 'LineWidth', 1.6);
    end
  end
  view(ax, 40, 22);
  xlabel(ax, 'dim 1'); ylabel(ax, 'dim 2'); zlabel(ax, 'dim 3');
end

function draw_sphere3(ax, C, S, isImg, names, BG, FG)
  hold(ax, 'on'); style_axes(ax, BG, FG);
  [sx, sy, sz] = sphere(20);
  surf(ax, sx, sy, sz, 'FaceColor', [0.6 0.7 0.9], 'FaceAlpha', 0.06, ...
       'EdgeColor', 'none');
  th = linspace(0, 2*pi, 100); g = [0.45 0.47 0.55];
  plot3(ax, cos(th), sin(th), zeros(size(th)), '-', 'Color', g);
  plot3(ax, cos(th), zeros(size(th)), sin(th), '-', 'Color', g);
  for k = 1:size(C,1)
    v = C(k,:);
    plot3(ax, [0 v(1)], [0 v(2)], [0 v(3)], '-', 'Color', S.colors(k,:), ...
          'LineWidth', 1.8);
    if isImg(k), mk = 'o'; ms = 7; else, mk = '^'; ms = 9; end
    plot3(ax, v(1), v(2), v(3), 'LineStyle', 'none', 'Marker', mk, ...
          'MarkerSize', ms, 'MarkerFaceColor', S.colors(k,:), 'MarkerEdgeColor', 'k');
    text(ax, v(1)*1.08, v(2)*1.08, v(3)*1.08, names{k}, 'Color', FG, 'FontSize', 7);
  end
  axis(ax, 'equal'); axis(ax, [-1 1 -1 1 -1 1]);
  view(ax, 40, 18);
  xlabel(ax, 'PC1'); ylabel(ax, 'PC2'); zlabel(ax, 'PC3');
end

function add_legend(ax, S, FG, BG)
  h = []; labels = {};
  for ci = 1:numel(S.cat_keys)
    ck = S.cat_keys{ci};
    h(end+1) = plot3(ax, nan, nan, nan, 'LineStyle', 'none', 'Marker', 's', ...
                     'MarkerSize', 9, 'MarkerFaceColor', cat_color(S, ck), ...
                     'MarkerEdgeColor', 'k');
    labels{end+1} = ck;
  end
  h(end+1) = plot3(ax, nan, nan, nan, 'LineStyle', 'none', 'Marker', 'o', ...
                   'MarkerSize', 8, 'MarkerFaceColor', [0.7 0.7 0.7], 'MarkerEdgeColor', 'k');
  labels{end+1} = 'image';
  h(end+1) = plot3(ax, nan, nan, nan, 'LineStyle', 'none', 'Marker', '^', ...
                   'MarkerSize', 9, 'MarkerFaceColor', [0.7 0.7 0.7], 'MarkerEdgeColor', 'k');
  labels{end+1} = 'text';
  lg = legend(ax, h, labels);
  set(lg, 'TextColor', FG, 'Color', BG);
end

function make_spin_gif(S, isImg, names, BG, FG, vis, fname)
  fig = figure('Color', BG, 'Position', [100 100 560 520], 'Visible', vis);
  ax = axes('Parent', fig);
  draw_sphere3(ax, S.sphere, S, isImg, names, BG, FG);
  title(ax, 'CLIP embeddings: cosine similarity = vector angle', 'Color', FG);

  az = 0:12:348;
  % 고정 6x6x6 web-safe 컬러맵 (모든 프레임 공유 -> rgb2ind(map) 미지원 회피)
  n = 6;  idxlist = (0:n^3-1)';
  map = [floor(idxlist/(n*n)), floor(mod(idxlist, n*n)/n), mod(idxlist, n)] / (n-1);

  A = [];
  for i = 1:numel(az)
    view(ax, az(i), 18); drawnow;
    F = getframe(fig);
    q = websafe_index(F.cdata, n);
    if isempty(A)
      A = zeros([size(q), 1, numel(az)], 'uint8');
    end
    A(:, :, 1, i) = q;
  end
  imwrite(A, map, fname, 'DelayTime', 0.06, 'LoopCount', Inf);
  close(fig);
end

function idx = websafe_index(im, n)
  r = round(double(im(:, :, 1)) / 255 * (n-1));
  g = round(double(im(:, :, 2)) / 255 * (n-1));
  b = round(double(im(:, :, 3)) / 255 * (n-1));
  idx = uint8(r * n * n + g * n + b);   % map 행 정렬과 일치(0-based)
end
