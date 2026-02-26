% verify_eigenvalue_v2.m
% =====================================================================
% VERIFICATION: Compare Floquet eigenvalue prediction with direct
% stride map iteration.
%
% Method:
%   1. Find fixed point z* at each speed
%   2. Compute Jacobian eigenvalues (same as floquet_KUO_v2)
%   3. Perturb z* -> z* + delta
%   4. Iterate stride map N times, record ||z_n - z*||
%   5. Fit log(||z_n - z*||) to extract observed decay rate
%   6. Compare observed rate with |lambda_max|
%
% If the eigenvalue computation is correct, the observed decay rate
% should match |lambda_max| asymptotically.
% =====================================================================

gam = 0; k_hip = -0.08; per = 5;
opts_fine = odeset('RelTol',1e-12, 'AbsTol',1e-14, 'Refine',4, ...
                   'Events',@collision_with_guard);

% Test speeds (dimensionless step lengths)
test_s = [0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80];
N_iter  = 200;       % number of stride iterations
eps_pert = 1e-4;     % perturbation magnitude

% Perturbation directions: test multiple to check all eigenvalue directions
pert_dirs = eye(3);  % perturb each state variable independently

fprintf('=====================================================================\n');
fprintf(' VERIFICATION: Eigenvalue prediction vs direct simulation\n');
fprintf(' Perturbation magnitude: %.0e\n', eps_pert);
fprintf(' Stride iterations: %d\n', N_iter);
fprintf('=====================================================================\n\n');

% Also test perturbation magnitude sensitivity (method 4)
fprintf('--- Test 1: Jacobian perturbation sensitivity ---\n');
fprintf('%-6s %-8s', 's', 'v');
deltas_test = [1e-5, 1e-6, 1e-7, 1e-8];
for d = deltas_test, fprintf('  delta=%.0e       ', d); end
fprintf('\n%s\n', repmat('-',1,90));

for si = 1:length(test_s)
    s = test_s(si);
    alpha = asin(0.5*s);
    omega = -1.04*alpha;
    P = -omega*tan(alpha);
    z0 = [alpha; omega; (1-cos(2*alpha))*omega];

    [z_fp, T_fp, conv] = find_fp(z0, gam, k_hip, P, per, opts_fine);
    if ~conv, fprintf('%-6.3f NO CONV\n', s); continue; end

    v_fp = s / T_fp;
    fprintf('%-6.3f %-8.4f', s, v_fp);

    for di = 1:length(deltas_test)
        dd = deltas_test(di);
        J = zeros(3);
        ok_all = true;
        for j = 1:3
            zp = z_fp; zp(j) = zp(j) + dd;
            zm = z_fp; zm(j) = zm(j) - dd;
            [Sp,~,o1] = step_map(zp, gam, k_hip, P, per, opts_fine);
            [Sm,~,o2] = step_map(zm, gam, k_hip, P, per, opts_fine);
            if ~o1||~o2, ok_all = false; break; end
            J(:,j) = (Sp - Sm) / (2*dd);
        end
        if ~ok_all, fprintf('  FAIL              '); continue; end
        lam = eig(J);
        lam_abs = sort(abs(lam), 'descend');
        fprintf('  %.6f,%.6f', lam_abs(1), lam_abs(2));
    end
    fprintf('\n');
end
fprintf('\n');


% --- Test 2: Direct simulation vs eigenvalue prediction ---
fprintf('--- Test 2: Observed decay rate vs predicted |lambda_max| ---\n');
fprintf('%-6s %-8s %-12s %-12s %-12s %-10s\n', ...
    's', 'v', '|lam_max|', 'obs_rate', 'rel_error', 'PASS/FAIL');
fprintf('%s\n', repmat('-',1,70));

n_pass = 0;
n_total = 0;

for si = 1:length(test_s)
    s = test_s(si);
    alpha = asin(0.5*s);
    omega = -1.04*alpha;
    P = -omega*tan(alpha);
    z0 = [alpha; omega; (1-cos(2*alpha))*omega];

    [z_fp, T_fp, conv] = find_fp(z0, gam, k_hip, P, per, opts_fine);
    if ~conv, continue; end

    v_fp = s / T_fp;

    % Compute eigenvalues
    delta_J = 1e-7;
    J = zeros(3);
    ok_all = true;
    for j = 1:3
        zp = z_fp; zp(j) = zp(j) + delta_J;
        zm = z_fp; zm(j) = zm(j) - delta_J;
        [Sp,~,o1] = step_map(zp, gam, k_hip, P, per, opts_fine);
        [Sm,~,o2] = step_map(zm, gam, k_hip, P, per, opts_fine);
        if ~o1||~o2, ok_all = false; break; end
        J(:,j) = (Sp - Sm) / (2*delta_J);
    end
    if ~ok_all, continue; end

    lam = eig(J);
    lam_max_pred = max(abs(lam));

    % Direct simulation: perturb and iterate
    % Use perturbation in direction of dominant eigenvector for cleanest signal
    [V, D] = eig(J);
    [~, idx_dom] = max(abs(diag(D)));
    pert_vec = real(V(:, idx_dom));
    pert_vec = pert_vec / norm(pert_vec);

    z_curr = z_fp + eps_pert * pert_vec;
    err_log = NaN(N_iter, 1);

    for n = 1:N_iter
        [z_next, ~, ok] = step_map(z_curr, gam, k_hip, P, per, opts_fine);
        if ~ok, break; end
        err_log(n) = norm(z_next - z_fp);
        z_curr = z_next;
    end

    valid = find(~isnan(err_log) & err_log > 1e-14);
    if length(valid) < 20, continue; end

    % Fit: log(err) = n*log(rate) + const
    % Use strides 10-100 (skip transient, avoid noise floor)
    fit_range = valid(valid >= 10 & valid <= min(100, valid(end)));
    if length(fit_range) < 10, fit_range = valid(10:min(end,50)); end

    log_err = log(err_log(fit_range));
    nn = fit_range;
    p = polyfit(nn, log_err, 1);
    obs_rate = exp(p(1));

    rel_err = abs(obs_rate - lam_max_pred) / lam_max_pred;
    pass = rel_err < 0.02;
    n_total = n_total + 1;
    if pass, n_pass = n_pass + 1; end

    fprintf('%-6.3f %-8.4f %-12.6f %-12.6f %-12.4e %-10s\n', ...
        s, v_fp, lam_max_pred, obs_rate, rel_err, ...
        ternary(pass, 'PASS', 'FAIL'));
end

fprintf('%s\n', repmat('-',1,70));
fprintf('Result: %d / %d passed (< 2%% relative error)\n\n', n_pass, n_total);


% --- Test 3: Complex eigenvalue verification via oscillation ---
fprintf('--- Test 3: Complex eigenvalue -> oscillatory decay ---\n');
fprintf('If eigenvalues are complex conjugate, perturbation decay\n');
fprintf('should show oscillation. If eigenvalues are real (old result),\n');
fprintf('decay should be monotonic.\n\n');

% Pick a mid-speed point where eigenvalues are clearly complex
s_test = 0.40;
alpha = asin(0.5*s_test);
omega = -1.04*alpha;
P = -omega*tan(alpha);
z0 = [alpha; omega; (1-cos(2*alpha))*omega];

[z_fp, T_fp, conv] = find_fp(z0, gam, k_hip, P, per, opts_fine);
if conv
    v_fp = s_test / T_fp;
    J = zeros(3);
    for j = 1:3
        zp = z_fp; zp(j) = zp(j) + 1e-7;
        zm = z_fp; zm(j) = zm(j) - 1e-7;
        [Sp,~,~] = step_map(zp, gam, k_hip, P, per, opts_fine);
        [Sm,~,~] = step_map(zm, gam, k_hip, P, per, opts_fine);
        J(:,j) = (Sp - Sm) / 2e-7;
    end
    lam = eig(J);
    fprintf('s=%.2f, v=%.4f\n', s_test, v_fp);
    fprintf('Eigenvalues: ');
    for j = 1:3
        if abs(imag(lam(j))) > 1e-8
            fprintf('%.4f%+.4fi  ', real(lam(j)), imag(lam(j)));
        else
            fprintf('%.6f  ', real(lam(j)));
        end
    end
    fprintf('\n');

    has_complex = any(abs(imag(lam)) > 1e-6);
    fprintf('Complex conjugate pair: %s\n', ternary(has_complex, 'YES', 'NO'));

    % Iterate and check sign changes in theta perturbation
    z_curr = z_fp + [1e-4; 0; 0];
    dtheta = NaN(60, 1);
    for n = 1:60
        [z_next, ~, ok] = step_map(z_curr, gam, k_hip, P, per, opts_fine);
        if ~ok, break; end
        dtheta(n) = z_next(1) - z_fp(1);
        z_curr = z_next;
    end
    valid_dt = dtheta(~isnan(dtheta));

    % Count sign changes
    sign_changes = sum(diff(sign(valid_dt)) ~= 0);
    fprintf('Sign changes in dtheta over %d strides: %d\n', length(valid_dt), sign_changes);
    if has_complex
        fprintf('-> Complex eigenvalues predict oscillatory decay (many sign changes)\n');
    else
        fprintf('-> Real eigenvalues predict monotonic decay (few sign changes)\n');
    end
    if has_complex && sign_changes > 5
        fprintf('-> CONSISTENT with complex eigenvalues\n');
    elseif ~has_complex && sign_changes <= 2
        fprintf('-> CONSISTENT with real eigenvalues\n');
    else
        fprintf('-> CHECK: pattern may warrant investigation\n');
    end
end

fprintf('\n=====================================================================\n');
fprintf(' VERIFICATION COMPLETE\n');
fprintf('=====================================================================\n');


% =========================================================================
% HELPER FUNCTIONS
% =========================================================================

function s = ternary(cond, a, b)
    if cond, s = a; else, s = b; end
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

    % Phase 1: depart collision surface (no events)
    dt = 0.005;
    opts_ne = odeset('RelTol',1e-12, 'AbsTol',1e-14);
    [~, yout1] = ode45(@(t,y)eom(t,y,gam,k), [0 dt], y0, opts_ne);
    y_dep = yout1(end,:).';

    % Phase 2: integrate with collision event
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