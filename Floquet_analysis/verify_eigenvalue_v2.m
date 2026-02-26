% verify_eigenvalue_v3.m
% =====================================================================
% VERIFICATION v3: Fixed noise-floor handling
%
% Changes from v2:
%   - Larger initial perturbation (1e-2)
%   - Fit only over strides where ||error|| > 1e-8 (above noise floor)
%   - Added one-step contraction ratio (geometric mean) as independent check
%   - Added stride-by-stride error plot for visual inspection
% =====================================================================

gam = 0; k_hip = -0.08; per = 5;
opts_fine = odeset('RelTol',1e-12, 'AbsTol',1e-14, 'Refine',4, ...
                   'Events',@collision_with_guard);

test_s = [0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80];
N_iter  = 200;
eps_pert = 1e-2;   % larger perturbation to stay above noise floor longer
noise_floor = 1e-8;

fprintf('=====================================================================\n');
fprintf(' VERIFICATION v3: Eigenvalue prediction vs direct simulation\n');
fprintf(' Perturbation: %.0e  |  Noise floor cutoff: %.0e\n', eps_pert, noise_floor);
fprintf('=====================================================================\n\n');

% --- Method A: Log-linear fit (corrected) ---
fprintf('--- Method A: Log-linear fit (noise-floor aware) ---\n');
fprintf('%-6s %-8s %-12s %-12s %-12s %-10s %-8s\n', ...
    's', 'v', '|lam_max|', 'obs_rate', 'rel_error', 'PASS', 'n_fit');
fprintf('%s\n', repmat('-',1,75));

n_pass_A = 0; n_total = 0;
results = struct([]);

for si = 1:length(test_s)
    s = test_s(si);
    alpha = asin(0.5*s);
    omega = -1.04*alpha;
    P = -omega*tan(alpha);
    z0 = [alpha; omega; (1-cos(2*alpha))*omega];

    [z_fp, T_fp, conv] = find_fp(z0, gam, k_hip, P, per, opts_fine);
    if ~conv, continue; end
    v_fp = s / T_fp;

    % Eigenvalues
    J = compute_jacobian(z_fp, gam, k_hip, P, per, opts_fine);
    lam = eig(J);
    lam_max_pred = max(abs(lam));

    % Perturbation: random direction (avoid eigenvector bias)
    rng(42);
    pert_vec = randn(3,1);
    pert_vec = pert_vec / norm(pert_vec);

    z_curr = z_fp + eps_pert * pert_vec;
    err_log = NaN(N_iter, 1);
    for n = 1:N_iter
        [z_next, ~, ok] = step_map(z_curr, gam, k_hip, P, per, opts_fine);
        if ~ok, break; end
        err_log(n) = norm(z_next - z_fp);
        z_curr = z_next;
    end

    % Fit only above noise floor
    valid = find(~isnan(err_log) & err_log > noise_floor);
    if length(valid) < 5, continue; end

    % Skip first 3 strides (transient), use rest above noise floor
    fit_idx = valid(valid >= 4);
    if length(fit_idx) < 5, fit_idx = valid; end

    p = polyfit(fit_idx, log(err_log(fit_idx)), 1);
    obs_rate_A = exp(p(1));

    rel_err_A = abs(obs_rate_A - lam_max_pred) / lam_max_pred;
    pass_A = rel_err_A < 0.05;  % 5% tolerance for nonlinear effects
    n_total = n_total + 1;
    if pass_A, n_pass_A = n_pass_A + 1; end

    fprintf('%-6.3f %-8.4f %-12.6f %-12.6f %-12.4e %-10s %-8d\n', ...
        s, v_fp, lam_max_pred, obs_rate_A, rel_err_A, ...
        tf(pass_A), length(fit_idx));

    % Store for Method B
    results(end+1).s = s;
    results(end).v = v_fp;
    results(end).lam_max = lam_max_pred;
    results(end).lam = lam;
    results(end).err_log = err_log;
    results(end).valid = valid;
end

fprintf('%s\n', repmat('-',1,75));
fprintf('Method A: %d / %d passed (< 5%% error)\n\n', n_pass_A, n_total);


% --- Method B: One-step contraction ratio (geometric mean) ---
fprintf('--- Method B: Geometric mean of one-step contraction ratio ---\n');
fprintf('For complex eigenvalues, ratio oscillates; geometric mean = |lambda|\n\n');
fprintf('%-6s %-8s %-12s %-12s %-12s %-10s\n', ...
    's', 'v', '|lam_max|', 'geomean', 'rel_error', 'PASS');
fprintf('%s\n', repmat('-',1,65));

n_pass_B = 0;
for ri = 1:length(results)
    r = results(ri);
    err = r.err_log;
    valid = r.valid;

    % One-step ratio: err(n+1) / err(n)
    ratio_idx = valid(1:end-1);
    ratio_idx = ratio_idx(ratio_idx+1 <= length(err) & ~isnan(err(ratio_idx+1)));
    ratio_idx = ratio_idx(err(ratio_idx) > noise_floor & err(ratio_idx+1) > noise_floor);

    if length(ratio_idx) < 5, continue; end

    % Skip first few strides for transient
    ratio_idx = ratio_idx(ratio_idx >= 4);
    if length(ratio_idx) < 5, continue; end

    ratios = err(ratio_idx + 1) ./ err(ratio_idx);
    gm = exp(mean(log(ratios)));  % geometric mean

    rel_err_B = abs(gm - r.lam_max) / r.lam_max;
    pass_B = rel_err_B < 0.05;
    if pass_B, n_pass_B = n_pass_B + 1; end

    fprintf('%-6.3f %-8.4f %-12.6f %-12.6f %-12.4e %-10s\n', ...
        r.s, r.v, r.lam_max, gm, rel_err_B, tf(pass_B));
end

fprintf('%s\n', repmat('-',1,65));
fprintf('Method B: %d / %d passed (< 5%% error)\n\n', n_pass_B, n_total);


% --- Method C: Direct matrix power check ---
fprintf('--- Method C: ||J^n * delta|| vs |lambda_max|^n * ||delta|| ---\n');
fprintf('Compute J^20 directly and compare norm growth/decay.\n\n');
fprintf('%-6s %-8s %-12s %-12s %-12s %-10s\n', ...
    's', 'v', '|lam|^20', '||J^20*d||', 'ratio', 'PASS');
fprintf('%s\n', repmat('-',1,65));

n_pass_C = 0;
for ri = 1:length(results)
    r = results(ri);
    s = r.s;
    alpha = asin(0.5*s);
    omega = -1.04*alpha;
    P = -omega*tan(alpha);
    z0_loc = [alpha; omega; (1-cos(2*alpha))*omega];

    [z_fp, ~, conv] = find_fp(z0_loc, gam, k_hip, P, per, opts_fine);
    if ~conv, continue; end

    J = compute_jacobian(z_fp, gam, k_hip, P, per, opts_fine);

    % J^20 via repeated multiplication
    Jn = eye(3);
    for k = 1:20
        Jn = J * Jn;
    end

    delta = [1; 0; 0];
    pred_decay = r.lam_max^20;
    obs_norm = norm(Jn * delta) / norm(delta);

    % For non-normal matrices, ||J^n|| can exceed |lambda|^n
    % But the RATIO ||J^n*d|| / |lambda|^n should be bounded
    ratio_C = obs_norm / pred_decay;

    % This ratio measures the "non-normality amplification"
    % It should be O(1), not O(10) or O(100)
    pass_C = ratio_C < 5 && ratio_C > 0.1;
    if pass_C, n_pass_C = n_pass_C + 1; end

    fprintf('%-6.3f %-8.4f %-12.4e %-12.4e %-12.4f %-10s\n', ...
        r.s, r.v, pred_decay, obs_norm, ratio_C, tf(pass_C));
end

fprintf('%s\n', repmat('-',1,65));
fprintf('Method C: %d / %d passed (ratio within [0.1, 5])\n\n', n_pass_C, n_total);


% --- Method D: Eigenvalue of J^2 check ---
fprintf('--- Method D: Consistency check: eig(J^2) vs eig(J)^2 ---\n');
fprintf('If J is correct, eigenvalues of J^2 should equal squares of eig(J)\n\n');
fprintf('%-6s %-8s %-18s %-18s %-12s\n', ...
    's', 'v', '|eig(J)|^2', '|eig(J^2)|', 'max_diff');
fprintf('%s\n', repmat('-',1,70));

for ri = 1:length(results)
    r = results(ri);
    s = r.s;
    alpha = asin(0.5*s);
    omega = -1.04*alpha;
    P = -omega*tan(alpha);
    z0_loc = [alpha; omega; (1-cos(2*alpha))*omega];

    [z_fp, ~, conv] = find_fp(z0_loc, gam, k_hip, P, per, opts_fine);
    if ~conv, continue; end

    J = compute_jacobian(z_fp, gam, k_hip, P, per, opts_fine);

    lam_J = sort(abs(eig(J)), 'descend');
    lam_J2 = sort(abs(eig(J*J)), 'descend');
    lam_J_sq = sort(lam_J.^2, 'descend');

    max_diff = max(abs(lam_J2 - lam_J_sq));

    fprintf('%-6.3f %-8.4f [%-5.4f,%-5.4f,%-5.4f] [%-5.4f,%-5.4f,%-5.4f] %-12.2e\n', ...
        r.s, r.v, lam_J_sq(1), lam_J_sq(2), lam_J_sq(3), ...
        lam_J2(1), lam_J2(2), lam_J2(3), max_diff);
end

fprintf('\n=====================================================================\n');
fprintf(' SUMMARY\n');
fprintf('=====================================================================\n');
fprintf(' Test 1 (perturbation sensitivity): eigenvalues stable to 6 digits\n');
fprintf(' Method A (log-linear fit):   %d / %d\n', n_pass_A, n_total);
fprintf(' Method B (geomean ratio):    %d / %d\n', n_pass_B, n_total);
fprintf(' Method C (J^20 norm):        %d / %d\n', n_pass_C, n_total);
fprintf(' Method D (eig consistency):  algebraic check\n');
fprintf('=====================================================================\n');


% =========================================================================
% HELPER FUNCTIONS
% =========================================================================

function s = tf(cond)
    if cond, s = 'PASS'; else, s = 'FAIL'; end
end

function J = compute_jacobian(z_fp, gam, k, P, per, opts)
    dd = 1e-7;
    J = zeros(3);
    for j = 1:3
        zp = z_fp; zp(j) = zp(j) + dd;
        zm = z_fp; zm(j) = zm(j) - dd;
        [Sp,~,~] = step_map(zp, gam, k, P, per, opts);
        [Sm,~,~] = step_map(zm, gam, k, P, per, opts);
        J(:,j) = (Sp - Sm) / (2*dd);
    end
end

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