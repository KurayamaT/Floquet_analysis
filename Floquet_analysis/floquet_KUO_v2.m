function floquet_KUO_v2(gam, k_hip)
%FLOQUET_KUO_V2  Corrected Floquet multiplier analysis for Kuo (2002)
%
%   BUG FIX: The original floquet_KUO.m used stride_map_reduced which
%   starts integration exactly on the collision surface (phi = 2*theta).
%   ODE45 event detection can behave unpredictably when the event function
%   is exactly zero at t=0 with direction matching the derivative sign.
%   This corrupted the Jacobian and produced a spurious lambda_1 = -1.
%
%   FIX: Two-phase integration.
%     Phase 1: short integration (dt=0.005) WITHOUT events -> depart surface
%     Phase 2: integration WITH event detection -> catch real heelstrike
%
%   Additionally, the fixed point is re-converged with tight ODE tolerances
%   before the final Jacobian computation.
%
%   FLOQUET_KUO_V2()        default: gam=0, k_hip=-0.08
%   FLOQUET_KUO_V2(GAM)     specify slope
%   FLOQUET_KUO_V2(GAM, K)  specify slope and hip spring
%
%   Outputs:
%     floquet_gam{gam}_k{k}.csv   - full results
%     floquet_KUO_v2_fig1.png     - 4-panel diagnostic
%     floquet_KUO_v2_fig_ms.png   - manuscript Figure 1

if nargin < 1, gam = 0; end
if nargin < 2, k_hip = -0.08; end

per = 5;

s_range = [linspace(0.01, 0.05, 500), linspace(0.05, 0.80, 2000)];
N = length(s_range);

speeds_out = NaN(1, N);
s_out      = NaN(1, N);
lam_out    = complex(NaN(N, 3));
z_fp_out   = NaN(N, 3);
T_out      = NaN(1, N);
valid_out  = false(1, N);
trJ_out    = NaN(1, N);

opts_nr   = odeset('RelTol',1e-8, 'AbsTol',1e-10, 'Refine',4, ...
                   'Events',@collision_with_guard);
opts_fine = odeset('RelTol',1e-12,'AbsTol',1e-14, 'Refine',4, ...
                   'Events',@collision_with_guard);

fprintf('============================================================\n');
fprintf(' Floquet Analysis: Kuo (2002) — Corrected (v2)\n');
fprintf(' gamma = %.4f,  k_hip = %.4f\n', gam, k_hip);
fprintf(' %d step lengths in [%.3f, %.3f]\n', N, s_range(1), s_range(end));
fprintf(' Fix: two-phase integration in step map\n');
fprintf('============================================================\n\n');
tic;

n_conv = 0;
pct_prev = 0;
fprintf('Progress: [');

for idx = 1:N
    % Progress bar
    pct = floor(idx/N*50);
    if pct > pct_prev
        fprintf(repmat('#', 1, pct - pct_prev));
        pct_prev = pct;
    end

    s = s_range(idx);
    alpha = asin(0.5*s);
    omega = -1.04*alpha;
    P = -omega*tan(alpha);
    z0 = [alpha; omega; (1-cos(2*alpha))*omega];

    % Newton-Raphson with moderate tolerances
    [z_fp, ~, conv] = find_fixed_point(z0, gam, k_hip, P, per, opts_nr);
    if ~conv, continue; end

    % Re-converge with tight tolerances
    [z_fp, T_stride, conv] = find_fixed_point(z_fp, gam, k_hip, P, per, opts_fine);
    if ~conv, continue; end

    % Verify residual
    [Sz, ~, ok] = step_map(z_fp, gam, k_hip, P, per, opts_fine);
    if ~ok || norm(Sz - z_fp) > 1e-9, continue; end

    % Jacobian with tight tolerances
    delta = 1e-7;
    J = zeros(3, 3);
    ok_all = true;
    for j = 1:3
        zp = z_fp; zp(j) = zp(j) + delta;
        zm = z_fp; zm(j) = zm(j) - delta;
        [Sp, ~, ok1] = step_map(zp, gam, k_hip, P, per, opts_fine);
        [Sm, ~, ok2] = step_map(zm, gam, k_hip, P, per, opts_fine);
        if ~ok1 || ~ok2, ok_all = false; break; end
        J(:,j) = (Sp - Sm) / (2*delta);
    end
    if ~ok_all, continue; end

    lambda = eig(J);
    [~, si] = sort(abs(lambda), 'descend');
    lambda = lambda(si);

    speeds_out(idx) = s / T_stride;
    s_out(idx)      = s;
    lam_out(idx,:)  = lambda.';
    z_fp_out(idx,:) = z_fp.';
    T_out(idx)      = T_stride;
    valid_out(idx)  = true;
    trJ_out(idx)    = real(trace(J));
    n_conv = n_conv + 1;
end

fprintf('] Done\n');
elapsed = toc;
fprintf('Elapsed: %.1f sec.  Converged: %d / %d\n\n', elapsed, n_conv, N);

% Filter & sort
mask    = valid_out;
speeds  = speeds_out(mask);
s_vals  = s_out(mask);
lam_all = lam_out(mask, :);
T_vals  = T_out(mask);
trJ_vals = trJ_out(mask);
n_found = sum(mask);

if n_found == 0, error('No periodic orbits found.'); end

[speeds, si] = sort(speeds);
s_vals   = s_vals(si);
lam_all  = lam_all(si, :);
T_vals   = T_vals(si);
trJ_vals = trJ_vals(si);

% Spectral radius (= max |lambda_i| over all three eigenvalues)
lam_max = max(abs(lam_all), [], 2);
N_half  = -log(2) ./ log(lam_max);

% Stability: all |lambda| < 1
stable = lam_max < 1;

% =====================================================================
% CONSOLE SUMMARY
% =====================================================================
fprintf('%s\n', repmat('=',1,100));
fprintf('%-8s %-8s %-8s %-30s %-30s %-8s %-8s\n', ...
    's', 'v', 'T', 'lambda_1', 'lambda_2', '|lam_max|', 'N_1/2');
fprintf('%s\n', repmat('-',1,100));
step_print = max(1, floor(n_found/20));
for i = 1:step_print:n_found
    la = lam_all(i,:);
    fprintf('%-8.3f %-8.4f %-8.3f %-30s %-30s %-8.4f %-8.1f\n', ...
        s_vals(i), speeds(i), T_vals(i), fmt(la(1)), fmt(la(2)), ...
        lam_max(i), N_half(i));
end
fprintf('%s\n\n', repmat('=',1,100));

% Table 2 output
fprintf('=== Table 2 (manuscript) ===\n');
v_targets = [0.063, 0.080, 0.100, 0.120, 0.140, 0.160, 0.180, 0.200, 0.220];
fprintf('%-6s %-8s %-10s %-10s %-10s %-10s\n', ...
    'v', 'V(m/s)', '|lam_max|', 'N_1/2', 'Re(lam1)', 'Im(lam1)');
fprintf('%s\n', repmat('-',1,60));
for i = 1:length(v_targets)
    [~, ki] = min(abs(speeds - v_targets(i)));
    la = lam_all(ki,:);
    V_dim = speeds(ki) * sqrt(9.81);
    fprintf('%-6.3f %-8.2f %-10.4f %-10.1f %-10.4f %-10.4f\n', ...
        speeds(ki), V_dim, lam_max(ki), N_half(ki), ...
        real(la(1)), imag(la(1)));
end
fprintf('%s\n\n', repmat('-',1,60));

% =====================================================================
% CSV EXPORT
% =====================================================================
csv_name = sprintf('floquet_gam%.3f_k%.3f.csv', gam, k_hip);
fid = fopen(csv_name, 'w');
fprintf(fid, ['s,v,T_stride,' ...
    'lam1_re,lam1_im,lam2_re,lam2_im,lam3_re,lam3_im,' ...
    'abs_lam1,abs_lam2,abs_lam3,lam_max,N_half,trace_J,stable\n']);
for i = 1:n_found
    la = lam_all(i,:);
    fprintf(fid, ['%.6f,%.6f,%.6f,' ...
        '%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,' ...
        '%.8f,%.8f,%.8f,%.8f,%.4f,%.6f,%d\n'], ...
        s_vals(i), speeds(i), T_vals(i), ...
        real(la(1)),imag(la(1)), real(la(2)),imag(la(2)), ...
        real(la(3)),imag(la(3)), ...
        abs(la(1)), abs(la(2)), abs(la(3)), ...
        lam_max(i), N_half(i), trJ_vals(i), stable(i));
end
fclose(fid);
fprintf('Exported %d orbits to %s\n\n', n_found, csv_name);

% =====================================================================
% FIGURE 1: Diagnostic 4-panel
% =====================================================================
figure('Color','w','Position',[100 100 1200 800],'Name','Diagnostic');

subplot(2,2,1); hold on;
plot(speeds, abs(lam_all(:,1)), 'b.', 'MarkerSize',3);
plot(speeds, abs(lam_all(:,2)), 'r.', 'MarkerSize',3);
plot(speeds, abs(lam_all(:,3)), 'g.', 'MarkerSize',3);
yline(1,'k--');
xlabel('v = s/T'); ylabel('|\lambda|'); title('Eigenvalue magnitudes');
legend('|\lambda_1|','|\lambda_2|','|\lambda_3|','Location','best');
grid on;

subplot(2,2,2); hold on;
th = linspace(0,2*pi,200);
plot(cos(th),sin(th),'k--','LineWidth',0.5);
cmap = parula(n_found);
for i = 1:n_found
    plot(real(lam_all(i,:)), imag(lam_all(i,:)), '.', ...
        'Color',cmap(i,:), 'MarkerSize',6);
end
xlabel('Re(\lambda)'); ylabel('Im(\lambda)');
title('Complex plane'); axis equal; grid on;
xlim([-1.5 1.5]); ylim([-1 1]);

subplot(2,2,3); hold on;
plot(speeds, N_half, 'b.-', 'MarkerSize',3);
xlabel('v = s/T'); ylabel('N_{1/2} (steps)');
title('Perturbation half-life'); grid on; ylim([0 30]);

subplot(2,2,4); hold on;
plot(speeds, trJ_vals, 'k.-', 'MarkerSize',3);
yline(0,'k:');
xlabel('v = s/T'); ylabel('trace(J)');
title('Jacobian trace (= \Sigma\lambda_i)'); grid on;

sgtitle(sprintf('Floquet Analysis (corrected) — \\gamma=%.3f, k=%.3f', gam, k_hip));
saveas(gcf, 'floquet_KUO_v2_fig1.png');
fprintf('Diagnostic figure saved.\n');

% =====================================================================
% FIGURE 2: Manuscript Figure 1
%   Left axis:  |lambda_max| vs v (black solid)
%   Right axis: N_1/2 vs v (grey dashed)
%   Top axis:   V (m/s) for L=1m
% =====================================================================
L = 1;  g = 9.81;
V_dim = speeds * sqrt(g*L);

fig = figure('Color','w','Position',[100 100 800 500],'Name','Manuscript Figure 1');

% Only use converged stable orbits for clean plot
idx_stable = find(stable(:)' & speeds > 0.02);

v_plot  = speeds(idx_stable);
V_plot  = V_dim(idx_stable);
lm_plot = lam_max(idx_stable);
nh_plot = N_half(idx_stable);

% Left axis: |lambda_max|
ax1 = axes('Position',[0.12 0.14 0.74 0.72]);
hold(ax1, 'on');
plot(ax1, v_plot, lm_plot, 'k-', 'LineWidth', 2.0);
ylabel(ax1, 'Floquet multiplier  |\lambda_{max}|', 'FontSize', 12);
xlabel(ax1, 'Dimensionless walking speed  v', 'FontSize', 12);
set(ax1, 'YLim', [0.55 1.00], 'XLim', [0.02 0.24]);
set(ax1, 'FontSize', 11, 'Box', 'off');
grid(ax1, 'on');

% Right axis: N_1/2
ax2 = axes('Position', get(ax1,'Position'), 'YAxisLocation','right', ...
    'Color','none', 'XTick',[], 'Box','off');
hold(ax2, 'on');
plot(ax2, v_plot, nh_plot, '--', 'Color',[0.5 0.5 0.5], 'LineWidth', 1.8);
ylabel(ax2, 'Perturbation half-life  N_{1/2}  (strides)', 'FontSize', 12);
set(ax2, 'YLim', [0 30], 'XLim', [0.02 0.24]);
set(ax2, 'FontSize', 11);
linkaxes([ax1 ax2], 'x');

% Top axis: dimensional speed
ax3 = axes('Position', get(ax1,'Position'), 'XAxisLocation','top', ...
    'Color','none', 'YTick',[], 'Box','off');
v_ticks = [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.24];
V_ticks = v_ticks * sqrt(g*L);
set(ax3, 'XLim', [0.02 0.24] * sqrt(g*L), 'XTick', V_ticks);
set(ax3, 'XTickLabel', arrayfun(@(x) sprintf('%.1f',x), V_ticks, 'Uni',0));
xlabel(ax3, 'Dimensional walking speed  V  (m/s) for L = 1 m', 'FontSize', 11);
set(ax3, 'FontSize', 10);

% Vertical dashed line at elbow (v ~ 0.1)
[~, elbow_idx] = min(abs(v_plot - 0.10));
xline(ax1, v_plot(elbow_idx), '--', 'Color', [0.6 0.6 0.6], 'LineWidth', 1.0);
text(ax1, v_plot(elbow_idx)+0.003, 0.57, ...
    sprintf('V \\approx %.2f m/s', v_plot(elbow_idx)*sqrt(g*L)), ...
    'FontSize', 9, 'Color', [0.4 0.4 0.4], 'FontAngle', 'italic');

% Arrow indicating "Stable" direction
annotation(fig, 'textarrow', [0.22 0.22], [0.55 0.38], ...
    'String','Stable', 'FontSize',11, 'FontWeight','bold', ...
    'HeadStyle','vback2', 'HeadWidth',8, 'HeadLength',8);

% Legend
legend(ax1, '|\lambda_{max}|', 'Location','northeast', 'FontSize',10);
legend(ax2, 'N_{1/2}', 'Location','east', 'FontSize',10);

saveas(gcf, 'floquet_KUO_v2_fig_ms.png');
fprintf('Manuscript figure saved.\n');

end  % main function


% =========================================================================
% SUBFUNCTIONS
% =========================================================================

%% STEP MAP: two-phase integration
function [z_new, T_stride, ok] = step_map(z, gam, k, P, per, opts)
    z_new = []; T_stride = []; ok = false;
    if abs(z(1)) > pi/3, return; end

    y0 = [z(1); z(2); 2*z(1); z(3)];

    % Phase 1: depart collision surface (no events)
    dt = 0.005;
    opts_noevent = odeset('RelTol',1e-12, 'AbsTol',1e-14);
    [~, yout1] = ode45(@(t,y)eom(t,y,gam,k), [0 dt], y0, opts_noevent);
    y_dep = yout1(end,:).';

    % Phase 2: integrate with collision event
    [~, ~, te, ye, ie] = ode45(@(t,y)eom(t,y,gam,k), ...
        [dt, per], y_dep, opts);

    if isempty(ie), return; end
    idx_coll = find(ie == 1);
    if isempty(idx_coll), return; end
    ic = idx_coll(1);

    T_stride = te(ic);
    yc = ye(ic,:);

    % Collision + push-off
    c2  = cos(2*yc(1));
    s2P = sin(2*yc(1)) * P;
    z_new = [-yc(1);
             c2*yc(2) + s2P;
             c2*(1-c2)*yc(2) + (1-c2)*s2P];
    ok = true;
end


%% EQUATIONS OF MOTION
function ydot = eom(~, y, gam, k)
    F = k * y(3);
    ydot = [y(2);
            sin(y(1) - gam);
            y(4);
            sin(y(1)-gam) + sin(y(3))*(y(2)^2 - cos(y(1)-gam)) + F];
end


%% COLLISION EVENT
function [val, ist, dir] = collision_with_guard(~, y) %#ok<INUSL>
    val = [y(3) - 2*y(1);       % collision: phi - 2*theta = 0
           pi/2 - abs(y(1))];   % divergence guard
    ist = [1; 1];
    dir = [1; 0];
end


%% NEWTON-RAPHSON
function [z_fp, T_stride, converged] = find_fixed_point(z0, gam, k, P, per, opts)
    converged = false; T_stride = []; z_fp = z0;
    delta_nr = 1e-7; max_iter = 20; tol = 1e-12;

    [~, ~, ok] = step_map(z_fp, gam, k, P, per, opts);
    if ~ok, return; end

    for iter = 1:max_iter
        [Sz, T, ok] = step_map(z_fp, gam, k, P, per, opts);
        if ~ok, return; end
        T_stride = T;
        res = Sz - z_fp;
        if norm(res) < tol, converged = true; return; end

        Jg = zeros(3);
        for j = 1:3
            zp = z_fp; zp(j) = zp(j) + delta_nr;
            zm = z_fp; zm(j) = zm(j) - delta_nr;
            [Sp,~,ok1] = step_map(zp, gam, k, P, per, opts);
            [Sm,~,ok2] = step_map(zm, gam, k, P, per, opts);
            if ~ok1 || ~ok2, return; end
            Jg(:,j) = (Sp - Sm) / (2*delta_nr);
        end
        Jg = Jg - eye(3);
        if rcond(Jg) < 1e-14, return; end
        z_fp = z_fp + 0.8 * (-Jg \ res);
    end

    [Sz, T, ok] = step_map(z_fp, gam, k, P, per, opts);
    if ok && norm(Sz - z_fp) < 1e-9
        converged = true; T_stride = T;
    end
end


%% FORMAT EIGENVALUE FOR DISPLAY
function s = fmt(lam)
    if abs(imag(lam)) < 1e-8
        s = sprintf('%.6f (|%.4f|)', real(lam), abs(lam));
    else
        s = sprintf('%.4f%+.4fi (|%.4f|)', real(lam), imag(lam), abs(lam));
    end
end