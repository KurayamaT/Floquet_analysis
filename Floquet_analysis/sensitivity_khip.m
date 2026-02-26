% sensitivity_khip.m
% =====================================================================
% Sensitivity analysis: vary k_hip for the Kuo (2002) walking model
% Uses two-phase step map (corrected).
% =====================================================================

gam = 0; per = 5;

% k_hip values to test (Kuo 2002 used -0.08)
k_vals = [0, -0.04, -0.08, -0.12, -0.16, -0.20];
colors = lines(length(k_vals));

s_range = linspace(0.05, 0.80, 500);
N = length(s_range);

opts_nr   = odeset('RelTol',1e-8,  'AbsTol',1e-10, 'Refine',4, 'Events',@collision_with_guard);
opts_fine = odeset('RelTol',1e-12, 'AbsTol',1e-14, 'Refine',4, 'Events',@collision_with_guard);

g = 9.81; L = 1;

% Storage for all k_hip values
ALL = struct();

fprintf('============================================================\n');
fprintf(' Sensitivity analysis: k_hip sweep\n');
fprintf(' k_hip values: '); fprintf('%.2f  ', k_vals); fprintf('\n');
fprintf(' %d step lengths per k_hip\n', N);
fprintf('============================================================\n\n');

for ki = 1:length(k_vals)
    k_hip = k_vals(ki);
    fprintf('--- k_hip = %.3f ---\n', k_hip);
    tic;

    speeds  = NaN(1, N);
    s_out   = NaN(1, N);
    lam_all = complex(NaN(N, 3));
    valid   = false(1, N);

    for idx = 1:N
        s = s_range(idx);
        alpha = asin(0.5*s);
        omega = -1.04*alpha;
        P = -omega*tan(alpha);
        z0 = [alpha; omega; (1-cos(2*alpha))*omega];

        % Newton-Raphson with loose tolerances
        [z_fp, T_fp, conv] = find_fp(z0, gam, k_hip, P, per, opts_nr);
        if ~conv, continue; end

        % Re-converge with tight tolerances
        [z_fp, T_fp, conv] = find_fp(z_fp, gam, k_hip, P, per, opts_fine);
        if ~conv, continue; end

        % Jacobian with tight tolerances
        delta = 1e-7;
        J = zeros(3);
        ok_all = true;
        for j = 1:3
            zp = z_fp; zp(j) = zp(j) + delta;
            zm = z_fp; zm(j) = zm(j) - delta;
            [Sp,~,o1] = step_map(zp, gam, k_hip, P, per, opts_fine);
            [Sm,~,o2] = step_map(zm, gam, k_hip, P, per, opts_fine);
            if ~o1||~o2, ok_all = false; break; end
            J(:,j) = (Sp - Sm) / (2*delta);
        end
        if ~ok_all, continue; end

        lam = eig(J);
        lam = sort(lam, 'descend', 'ComparisonMethod', 'abs');

        speeds(idx)    = s / T_fp;
        s_out(idx)     = s;
        lam_all(idx,:) = lam.';
        valid(idx)     = true;
    end

    el = toc;
    n_conv = sum(valid);
    fprintf('  Converged: %d / %d  (%.1f sec)\n', n_conv, N, el);

    % Sort by speed
    mask = valid;
    sp = speeds(mask);
    sv = s_out(mask);
    la = lam_all(mask,:);
    [sp, si] = sort(sp);
    sv = sv(si);
    la = la(si,:);

    ALL(ki).k     = k_hip;
    ALL(ki).v     = sp;
    ALL(ki).s     = sv;
    ALL(ki).lam   = la;
    ALL(ki).lam_max = max(abs(la), [], 2)';
    ALL(ki).N_half  = -log(2) ./ log(ALL(ki).lam_max);
end

fprintf('\nAll sweeps complete.\n\n');


% =====================================================================
% FIGURE 1: |lambda_max| vs v for all k_hip
% =====================================================================
figure('Color','w','Position',[100 100 900 600],'Name','Sensitivity: k_hip');

subplot(2,2,1); hold on;
for ki = 1:length(k_vals)
    plot(ALL(ki).v, ALL(ki).lam_max, '-', 'Color', colors(ki,:), ...
        'LineWidth', 1.5, 'DisplayName', sprintf('k=%.2f', ALL(ki).k));
end
yline(1, 'k--', 'LineWidth', 0.5);
xlabel('v = s/T'); ylabel('|\lambda_{max}|');
title('Spectral radius vs speed');
legend('Location','best','FontSize',8); grid on;
ylim([0.4 1.05]);

subplot(2,2,2); hold on;
for ki = 1:length(k_vals)
    nh = ALL(ki).N_half;
    nh(nh > 50) = NaN;  % clip for display
    plot(ALL(ki).v, nh, '-', 'Color', colors(ki,:), 'LineWidth', 1.5);
end
xlabel('v = s/T'); ylabel('N_{1/2} (strides)');
title('Perturbation half-life vs speed');
ylim([0 30]); grid on;

% Dimensional speed
subplot(2,2,3); hold on;
for ki = 1:length(k_vals)
    V_dim = ALL(ki).v * sqrt(g*L);
    plot(V_dim, ALL(ki).lam_max, '-', 'Color', colors(ki,:), ...
        'LineWidth', 1.5, 'DisplayName', sprintf('k=%.2f', ALL(ki).k));
end
yline(1, 'k--', 'LineWidth', 0.5);
xlabel('V (m/s) for L=1m'); ylabel('|\lambda_{max}|');
title('Spectral radius vs dimensional speed');
legend('Location','best','FontSize',8); grid on;
ylim([0.4 1.05]);

% Complex plane trajectories
subplot(2,2,4); hold on;
th = linspace(0, 2*pi, 200);
plot(cos(th), sin(th), 'k--', 'LineWidth', 0.5, 'HandleVisibility','off');
for ki = 1:length(k_vals)
    la = ALL(ki).lam;
    plot(real(la(:,1)), imag(la(:,1)), '.', 'Color', colors(ki,:), ...
        'MarkerSize', 4, 'DisplayName', sprintf('k=%.2f', ALL(ki).k));
    plot(real(la(:,2)), imag(la(:,2)), '.', 'Color', colors(ki,:), ...
        'MarkerSize', 4, 'HandleVisibility','off');
end
xlabel('Re(\lambda)'); ylabel('Im(\lambda)');
title('Eigenvalue loci'); axis equal; grid on;
legend('Location','best','FontSize',8);

sgtitle('Sensitivity to hip spring stiffness k_{hip}');
saveas(gcf, 'sensitivity_khip.png');
fprintf('Figure saved: sensitivity_khip.png\n');


% =====================================================================
% FIGURE 2: Manuscript-style (dual axis) for k=-0.08 vs k=0 vs k=-0.16
% =====================================================================
figure('Color','w','Position',[100 100 800 400],'Name','k_hip comparison');

ki_sel = [find([ALL.k]==0), find([ALL.k]==-0.08), find([ALL.k]==-0.16)];
col_sel = {[0.8 0.2 0.2], [0 0 0], [0.2 0.2 0.8]};
sty_sel = {'--', '-', ':'};

hold on;
for ii = 1:length(ki_sel)
    ki = ki_sel(ii);
    if isempty(ki), continue; end
    V_dim = ALL(ki).v * sqrt(g*L);
    plot(V_dim, ALL(ki).lam_max, sty_sel{ii}, 'Color', col_sel{ii}, ...
        'LineWidth', 2.0, 'DisplayName', sprintf('k_{hip}=%.2f', ALL(ki).k));
end
yline(1, 'k--', 'LineWidth', 0.5, 'HandleVisibility','off');
xlabel('V (m/s) for L=1m', 'FontSize', 12);
ylabel('|\lambda_{max}|', 'FontSize', 12);
title('Effect of hip spring stiffness on gait stability', 'FontSize', 13);
legend('Location','best','FontSize',11); grid on;
ylim([0.4 1.05]); xlim([0 0.8]);
set(gca, 'FontSize', 11);
saveas(gcf, 'sensitivity_khip_ms.png');
fprintf('Figure saved: sensitivity_khip_ms.png\n');


% =====================================================================
% Table: key metrics at v ~ 0.063 and v ~ 0.10 for each k_hip
% =====================================================================
fprintf('\n=== Summary table ===\n');
fprintf('%-8s %-10s %-10s %-10s %-10s %-10s %-10s\n', ...
    'k_hip', 'v_min', '|lam|_min', 'N1/2_min', '|lam|@0.1', 'N1/2@0.1', 'v_elbow');
fprintf('%s\n', repmat('-',1,75));

for ki = 1:length(k_vals)
    v = ALL(ki).v;
    lm = ALL(ki).lam_max;
    nh = ALL(ki).N_half;

    % Minimum speed
    v_min = v(1);
    lm_min = lm(1);
    nh_min = nh(1);

    % Value at v ~ 0.1
    [~, i01] = min(abs(v - 0.10));
    lm_01 = lm(i01);
    nh_01 = nh(i01);

    % Elbow: where |lambda_max| first drops below 0.75
    i_elbow = find(lm < 0.75, 1, 'first');
    if ~isempty(i_elbow)
        v_elbow = v(i_elbow);
    else
        v_elbow = NaN;
    end

    fprintf('%-8.2f %-10.4f %-10.4f %-10.1f %-10.4f %-10.1f %-10.4f\n', ...
        ALL(ki).k, v_min, lm_min, nh_min, lm_01, nh_01, v_elbow);
end
fprintf('\n');


% =========================================================================
% SUBFUNCTIONS
% =========================================================================

function [z_fp, T_stride, conv] = find_fp(z0, gam, k, P, per, opts)
    conv = false; T_stride = []; z_fp = z0;
    dd = 1e-7;
    for it = 1:20
        [Sz, T, ok] = step_map(z_fp, gam, k, P, per, opts);
        if ~ok, return; end
        T_stride = T;
        res = Sz - z_fp;
        if norm(res) < 1e-12, conv = true; return; end
        Jg = zeros(3);
        for j = 1:3
            zp = z_fp; zp(j) = zp(j) + dd;
            zm = z_fp; zm(j) = zm(j) - dd;
            [Sp,~,o1] = step_map(zp, gam, k, P, per, opts);
            [Sm,~,o2] = step_map(zm, gam, k, P, per, opts);
            if ~o1||~o2, return; end
            Jg(:,j) = (Sp - Sm) / (2*dd);
        end
        Jg = Jg - eye(3);
        if rcond(Jg) < 1e-14, return; end
        z_fp = z_fp + 0.8*(-Jg\res);
    end
    [Sz,T,ok] = step_map(z_fp, gam, k, P, per, opts);
    if ok && norm(Sz - z_fp) < 1e-10, conv = true; T_stride = T; end
end

function [z_new, T_stride, ok] = step_map(z, gam, k, P, per, opts)
    z_new = []; T_stride = []; ok = false;
    if abs(z(1)) > pi/3, return; end
    y0 = [z(1); z(2); 2*z(1); z(3)];
    dt = 0.005;
    opts_ne = odeset('RelTol',1e-12, 'AbsTol',1e-14);
    [~, yout1] = ode45(@(t,y)eom(t,y,gam,k), [0 dt], y0, opts_ne);
    y_dep = yout1(end,:).';
    [~, ~, te, ye, ie] = ode45(@(t,y)eom(t,y,gam,k), [dt per], y_dep, opts);
    if isempty(ie), return; end
    idx = find(ie == 1);
    if isempty(idx), return; end
    ic = idx(1);
    T_stride = te(ic);
    yc = ye(ic,:);
    c2 = cos(2*yc(1)); s2p = sin(2*yc(1))*P;
    z_new = [-yc(1); c2*yc(2)+s2p; c2*(1-c2)*yc(2)+(1-c2)*s2p];
    ok = true;
end

function yd = eom(~, y, gam, k)
    F = k*y(3);
    yd = [y(2); sin(y(1)-gam); y(4);
          sin(y(1)-gam)+sin(y(3))*(y(2)^2-cos(y(1)-gam))+F];
end

function [v,ist,dir] = collision_with_guard(~, y)
    v = [y(3)-2*y(1); pi/2-abs(y(1))];
    ist = [1; 1]; dir = [1; 0];
end