function clip_embedding_viz_3d()
% CLIP 멀티모달 임베딩 3D 시각화 (MATLAB)
% =========================================================================
% Python(clip_embedding_viz.py -> export_for_matlab.py)이 만든
% embeddings.mat 을 읽어 3D로 시각화한다.
%
%   - 패널 (A): modality-gap 보정 t-SNE 3D  (같은 카테고리 군집)
%   - 패널 (B): PCA 단위 구면 벡터          (벡터 각도 = 코사인 유사도)
%
% 코사인 유사도 = 두 단위벡터 사이 각도의 코사인.
% 단위 구면 위에서 같은 카테고리 벡터들은 비슷한 방향(작은 각도)을 가리킨다.
%
% 출력:
%   - embedding_visualization_3d_matlab.png  (300 dpi)
%   - clip_embedding_3d_matlab.gif           (회전 애니메이션)
%
% 실행:  >> clip_embedding_viz_3d
% (먼저 Python 에서  python clip_embedding_viz.py && python export_for_matlab.py)
% =========================================================================

    matfile = 'embeddings.mat';
    if ~isfile(matfile)
        error(['%s 가 없습니다. 먼저 다음을 실행하세요:\n' ...
               '  python clip_embedding_viz.py\n' ...
               '  python export_for_matlab.py'], matfile);
    end
    S = load(matfile);

    KFONT = 'Malgun Gothic';          % 한글 라벨용 폰트
    BG    = [0.08 0.09 0.12];         % 다크 배경
    FG    = [0.92 0.93 0.96];         % 밝은 전경(글자/축)
    N     = size(S.emb, 1);
    isImg = strcmp(S.modalities(:), 'image');

    % ---------------------------------------------------------------------
    % 정적 2-패널 고해상도 figure
    % ---------------------------------------------------------------------
    fig = figure('Color', BG, 'Position', [80 80 1500 720], ...
                 'Name', 'CLIP 3D Embedding', 'InvertHardcopy', 'off');
    tl = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(tl, 'CLIP 멀티모달 임베딩 — 3D 시각화 (MATLAB)', ...
          'Color', FG, 'FontSize', 16, 'FontWeight', 'bold', 'FontName', KFONT);

    ax1 = nexttile(tl);
    drawScatter3(ax1, S.tsne_corr, S, isImg, BG, FG, KFONT, true);
    title(ax1, {'(A) modality-gap 보정 t-SNE 3D', '같은 카테고리 이미지+텍스트 군집'}, ...
          'Color', FG, 'FontName', KFONT, 'FontSize', 12);

    ax2 = nexttile(tl);
    drawSphere3(ax2, S.sphere, S, isImg, BG, FG, KFONT);
    title(ax2, {'(B) PCA 단위 구면 벡터', sprintf('각도 = 코사인 유사도  (3D 설명분산 %.1f%%)', 100*sum(S.evr))}, ...
          'Color', FG, 'FontName', KFONT, 'FontSize', 12);

    addLegend(ax2, S, KFONT, FG, BG);

    drawnow;
    exportgraphics(fig, 'embedding_visualization_3d_matlab.png', ...
                   'Resolution', 300, 'BackgroundColor', BG);
    fprintf('[png] embedding_visualization_3d_matlab.png 저장 (300dpi)\n');

    % ---------------------------------------------------------------------
    % 회전 애니메이션 GIF (단위 구면 뷰)
    % ---------------------------------------------------------------------
    makeSpinGif(S, isImg, BG, FG, KFONT, 'clip_embedding_3d_matlab.gif');
    fprintf('[gif] clip_embedding_3d_matlab.gif 저장\n완료.\n');
end


% =========================================================================
function drawScatter3(ax, C, S, isImg, BG, FG, KFONT, connect)
    hold(ax, 'on');  styleAxes(ax, BG, FG, KFONT);
    for k = 1:size(C,1)
        mk = ternary(isImg(k), 'o', '^');
        sz = ternary(isImg(k), 90, 130);
        scatter3(ax, C(k,1), C(k,2), C(k,3), sz, 'Marker', mk, ...
                 'MarkerFaceColor', S.colors(k,:), 'MarkerEdgeColor', 'k', ...
                 'MarkerFaceAlpha', 0.95, 'LineWidth', 0.6);
        text(ax, C(k,1), C(k,2), C(k,3), ['  ' S.names{k}], ...
             'Color', FG, 'FontSize', 8, 'FontName', KFONT, 'Interpreter', 'none');
    end
    if connect
        connectCentroids(ax, C, S);
    end
    view(ax, 40, 22);  axis(ax, 'vis3d');
    xlabel(ax,'dim 1'); ylabel(ax,'dim 2'); zlabel(ax,'dim 3');
end


% =========================================================================
function drawSphere3(ax, C, S, isImg, BG, FG, KFONT)
    hold(ax, 'on');  styleAxes(ax, BG, FG, KFONT);

    % 반투명 단위 구
    [sx, sy, sz] = sphere(48);
    surf(ax, sx, sy, sz, 'FaceColor', [0.6 0.7 0.9], 'FaceAlpha', 0.06, ...
         'EdgeColor', 'none');
    % 적도/자오선 가이드
    th = linspace(0, 2*pi, 100);
    guide = [0.45 0.47 0.55];
    plot3(ax, cos(th), sin(th), zeros(size(th)), '-', 'Color', guide);
    plot3(ax, cos(th), zeros(size(th)), sin(th), '-', 'Color', guide);

    for k = 1:size(C,1)
        v = C(k,:);
        % 원점에서 뻗는 벡터
        plot3(ax, [0 v(1)], [0 v(2)], [0 v(3)], '-', ...
              'Color', S.colors(k,:), 'LineWidth', 1.8);
        mk = ternary(isImg(k), 'o', '^');
        sz = ternary(isImg(k), 80, 120);
        scatter3(ax, v(1), v(2), v(3), sz, 'Marker', mk, ...
                 'MarkerFaceColor', S.colors(k,:), 'MarkerEdgeColor', 'k', ...
                 'LineWidth', 0.6);
        text(ax, v(1)*1.08, v(2)*1.08, v(3)*1.08, S.names{k}, ...
             'Color', FG, 'FontSize', 8, 'FontName', KFONT, 'Interpreter', 'none');
    end
    axis(ax, 'equal');  axis(ax, [-1 1 -1 1 -1 1]);
    view(ax, 40, 18);   axis(ax, 'vis3d');
    try  % 조명은 장식 — 버전에 따라 실패해도 무시
        axes(ax); camlight('headlight'); lighting(ax, 'gouraud');
    catch
    end
    xlabel(ax,'PC1'); ylabel(ax,'PC2'); zlabel(ax,'PC3');
end


% =========================================================================
function connectCentroids(ax, C, S)
    for ci = 1:numel(S.cat_keys)
        ck = S.cat_keys{ci};
        sel = strcmp(S.categories(:), ck);
        ii = sel & strcmp(S.modalities(:), 'image');
        tt = sel & strcmp(S.modalities(:), 'text');
        if ~any(ii) || ~any(tt), continue; end
        ic = mean(C(ii,:), 1);  tc = mean(C(tt,:), 1);
        col = S.colors(find(sel,1), :);
        plot3(ax, [ic(1) tc(1)], [ic(2) tc(2)], [ic(3) tc(3)], '--', ...
              'Color', col, 'LineWidth', 1.6);
    end
end


% =========================================================================
function styleAxes(ax, BG, FG, KFONT)
    set(ax, 'Color', BG, 'XColor', FG, 'YColor', FG, 'ZColor', FG, ...
        'GridColor', [0.5 0.5 0.55], 'GridAlpha', 0.25, 'FontName', KFONT, ...
        'FontSize', 9);
    grid(ax, 'on');  box(ax, 'off');
end


% =========================================================================
function addLegend(ax, S, KFONT, FG, BG)
    h = gobjects(0);  labels = {};
    for ci = 1:numel(S.cat_keys)
        ck = S.cat_keys{ci};
        col = S.colors(find(strcmp(S.categories(:), ck), 1), :);
        h(end+1) = scatter3(ax, nan, nan, nan, 90, 'Marker', 's', ...
                   'MarkerFaceColor', col, 'MarkerEdgeColor', 'k'); %#ok<AGROW>
        labels{end+1} = S.cat_labels{ci}; %#ok<AGROW>
    end
    h(end+1) = scatter3(ax, nan, nan, nan, 80, 'Marker', 'o', ...
               'MarkerFaceColor', [0.7 0.7 0.7], 'MarkerEdgeColor', 'k');
    labels{end+1} = '이미지';
    h(end+1) = scatter3(ax, nan, nan, nan, 110, 'Marker', '^', ...
               'MarkerFaceColor', [0.7 0.7 0.7], 'MarkerEdgeColor', 'k');
    labels{end+1} = '텍스트';
    lg = legend(ax, h, labels, 'FontName', KFONT, 'TextColor', FG, ...
                'Color', BG, 'EdgeColor', [0.4 0.4 0.45], 'Location', 'northeast');
    lg.FontSize = 9;
end


% =========================================================================
function makeSpinGif(S, isImg, BG, FG, KFONT, fname)
    fig = figure('Color', BG, 'Position', [120 120 760 700], ...
                 'InvertHardcopy', 'off', 'Visible', 'on');
    ax = axes(fig);
    drawSphere3(ax, S.sphere, S, isImg, BG, FG, KFONT);
    title(ax, {'CLIP 임베딩 — 코사인 유사도 (벡터 각도)', ...
               '같은 카테고리 = 비슷한 방향'}, ...
          'Color', FG, 'FontName', KFONT, 'FontSize', 12);

    az = 0:3:357;
    for i = 1:numel(az)
        view(ax, az(i), 18);  drawnow;
        frame = getframe(fig);
        [A, map] = rgb2ind(frame2im(frame), 256);
        if i == 1
            imwrite(A, map, fname, 'gif', 'LoopCount', Inf, 'DelayTime', 0.05);
        else
            imwrite(A, map, fname, 'gif', 'WriteMode', 'append', 'DelayTime', 0.05);
        end
    end
    close(fig);
end


% =========================================================================
function out = ternary(cond, a, b)
    if cond, out = a; else, out = b; end
end
