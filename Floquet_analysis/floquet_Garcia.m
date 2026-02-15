function floquet_Garcia(beta_mass)
%FLOQUET_GARCIA  Floquet multiplier analysis for Garcia et al. (1998) simplest walking model
%
%   Sweeps slope angle gamma using continuation from known reference
%   solutions. Finds long-step (stable) and short-step (unstable)
%   periodic orbits and computes 2x2 Floquet multipliers.
%
%   The Poincare section is the heelstrike surface theta2 = pi - 2*theta1.
%   Post-collision state reduces to z = [theta1, theta1dot] since
%   theta2dot = -theta1dot*(1 - cos(2*theta1)) from the collision law.
%   Hence the stride map is R^2 -> R^2 with 2 multipliers.
%
%   FLOQUET_GARCIA()      beta = 0 (simplest model: point feet, hip mass only)
%   FLOQUET_GARCIA(BETA)  specifies foot-to-hip mass ratio
%
%   Reference:
%     Garcia, M., Chatterjee, A., Ruina, A., & Coleman, M. (1998).
%     "The Simplest Walking Model: Stability, Complexity, and Scaling."
%     ASME J. Biomech. Eng., 120(2), 281-288.

if nargin < 1, beta_mass = 0; end
g_val = 1; l_val = 1;
per = 30;

%% ====================================================================
%  Configuration
%  ====================================================================
N_gam   = 300;
gam_min = 0.0005;
gam_max = 0.019;
gam_range = linspace(gam_min, gam_max, N_gam);

% Reference fixed points at gam = 0.009 (Garcia 1998, Table 1)
%   Long step:  loc = -0.191 +/- 0.556i  (|lam| ~ 0.588, stable)
%   Short step: loc = 4.000, 0.459       (unstable)
gam_ref = 0.009;
z_ref = { [0.20031090049483; -0.19983247291764], ...   % long step
          [0.19393736960624; -0.20386692731962] };      % short step
branch_names  = {'Long step','Short step'};
branch_colors = {[0.00 0.45 0.74], [0.85 0.33 0.10]};

% ODE options: loose for Newton-Raphson, tight for final Jacobian
opts_nr   = odeset('RelTol',1e-8, 'AbsTol',1e-10, 'Refine',4, ...
                   'Events',@heelstrike_event);
opts_fine = odeset('RelTol',1e-12,'AbsTol',1e-14, 'Refine',4, ...
                   'Events',@heelstrike_event);

%% ====================================================================
%  Pre-allocate storage
%  ====================================================================
for b = 2:-1:1
    R(b).gam   = gam_range;
    R(b).s     = NaN(1, N_gam);
    R(b).v     = NaN(1, N_gam);
    R(b).T     = NaN(1, N_gam);
    R(b).lam   = complex(NaN(N_gam, 2));
    R(b).z_fp  = NaN(N_gam, 2);
    R(b).valid = false(1, N_gam);
end

%% ====================================================================
%  Main computation (continuation from reference gamma)
%  ====================================================================
fprintf('============================================================\n');
fprintf(' Floquet Analysis: Garcia (1998) Simplest Walking Model\n');
fprintf(' beta = %.4f,  g = %.1f,  l = %.1f\n', beta_mass, g_val, l_val);
fprintf(' %d slopes: gamma in [%.4f, %.4f] rad\n', ...
    N_gam, gam_min, gam_max);
fprintf('============================================================\n\n');
tic;

for b = 1:2
    fprintf('--- %s branch ---\n', branch_names{b});

    % Index nearest to reference gamma
    [~, i_ref] = min(abs(gam_range - gam_ref));

    % Converge reference solution
    [z_fp, ~, conv] = find_fp_garcia(z_ref{b}, gam_range(i_ref), ...
        beta_mass, g_val, l_val, per, opts_nr);
    if ~conv
        fprintf('  FAILED at reference gamma = %.4f\n', gam_range(i_ref));
        fprintf('  Running diagnostic...\n');
        stride_map_garcia(z_ref{b}, gam_range(i_ref), ...
            beta_mass, g_val, l_val, per, opts_nr, true);
        continue;
    end
    fprintf('  Reference converged: theta1* = %.8f, theta1dot* = %.8f\n', ...
        z_fp(1), z_fp(2));
    R(b) = store_one(R(b), i_ref, z_fp, gam_range(i_ref), ...
        beta_mass, g_val, l_val, per, opts_fine);

    % --- Forward continuation (increasing gamma) ---
    z_guess = z_fp;
    n_fail  = 0;
    for i = (i_ref+1):N_gam
        [z_fp_i, ~, conv] = find_fp_garcia(z_guess, gam_range(i), ...
            beta_mass, g_val, l_val, per, opts_nr);
        if ~conv
            n_fail = n_fail + 1;
            if n_fail > 8, break; end
            continue;
        end
        n_fail = 0;
        R(b) = store_one(R(b), i, z_fp_i, gam_range(i), ...
            beta_mass, g_val, l_val, per, opts_fine);
        z_guess = z_fp_i;
    end

    % --- Backward continuation (decreasing gamma) ---
    z_guess = R(b).z_fp(i_ref,:).';
    n_fail  = 0;
    for i = (i_ref-1):-1:1
        [z_fp_i, ~, conv] = find_fp_garcia(z_guess, gam_range(i), ...
            beta_mass, g_val, l_val, per, opts_nr);
        if ~conv
            n_fail = n_fail + 1;
            if n_fail > 8, break; end
            % Scaling-based fallback guess: theta1* ~ gamma^(1/3)
            gc = gam_range(i)^(1/3);
            if b == 1
                z_fb = [0.963*gc; -0.961*gc];
            else
                z_fb = [0.932*gc; -0.980*gc];
            end
            [z_fp_i, ~, conv] = find_fp_garcia(z_fb, gam_range(i), ...
                beta_mass, g_val, l_val, per, opts_nr);
            if ~conv, continue; end
        end
        n_fail = 0;
        R(b) = store_one(R(b), i, z_fp_i, gam_range(i), ...
            beta_mass, g_val, l_val, per, opts_fine);
        z_guess = z_fp_i;
    end

    fprintf('  Found: %d / %d periodic orbits\n', sum(R(b).valid), N_gam);
end

elapsed = toc;
fprintf('\nDone in %.1f sec.\n\n', elapsed);

%% ====================================================================
%  Output
%  ====================================================================
plot_floquet_garcia(R, branch_names, branch_colors, beta_mass);
print_summary(R, branch_names);
export_csv_garcia(R, branch_names, beta_mass);

end % floquet_Garcia


%% =====================================================================
%  STRIDE MAP:  z = [theta1, theta1dot] -> z' after one full stride
%
%  KEY FIX: Two-phase integration.
%  Initial state lies exactly on collision surface theta2 = pi - 2*theta1,
%  causing ode45 event detection to trigger at t=0.
%  Phase 1: short integration WITHOUT events to depart collision surface
%  Phase 2: integration WITH events until next heelstrike
%  =====================================================================
function [z_new, T_stride, ok] = stride_map_garcia(z, gam, beta_mass, g, l, per, opts, varargin)
    z_new = []; T_stride = []; ok = false;
    debug = ~isempty(varargin) && varargin{1};

    if abs(z(1)) > pi/3 || z(1) <= 0
        if debug, fprintf('    Rejected: z(1) = %.6f out of range\n', z(1)); end
        return;
    end

    % Reconstruct full state on Poincare section
    theta1    = z(1);
    theta1dot = z(2);
    theta2    = pi - 2*theta1;
    theta2dot = -theta1dot * (1 - cos(2*theta1));

    y0  = [theta1; theta2; theta1dot; theta2dot];
    bv  = [beta_mass, gam, g, l];

    % --- Phase 1: depart collision surface (no events) ---
    dt_depart = 0.005;
    opts_noevent = odeset('RelTol',1e-10, 'AbsTol',1e-12);
    [~, yout1] = ode45(@(t,y) eom_garcia(y, bv), [0, dt_depart], y0, opts_noevent);
    y_depart = yout1(end,:).';

    if debug
        coll_val = y_depart(2) + 2*y_depart(1) - pi;
        fprintf('    After phase 1 (dt=%.4f): coll_val = %.6e\n', dt_depart, coll_val);
    end

    % --- Phase 2: integrate with heelstrike event ---
    [~, ~, te, ye, ie] = ode45(@(t,y) eom_garcia(y, bv), ...
        [dt_depart, per], y_depart, opts);

    if isempty(ie)
        if debug, fprintf('    No events detected\n'); end
        return;
    end

    idx_coll = find(ie == 1);
    if isempty(idx_coll)
        if debug, fprintf('    No heelstrike events (ie = [%s])\n', num2str(ie')); end
        return;
    end
    ic = idx_coll(1);

    T_stride = te(ic);   % total stride time includes phase 1
    yc = ye(ic,:);

    if debug
        fprintf('    Heelstrike at T = %.6f\n', T_stride);
        fprintf('    State at collision: [%.6f, %.6f, %.6f, %.6f]\n', yc);
        fprintf('    Collision residual: %.6e\n', yc(2) + 2*yc(1) - pi);
    end

    % Collision law (angular momentum balance about strike point)
    % At heelstrike, theta1 must be negative (stance leg past vertical)
    if yc(1) > 0
        if debug, fprintf('    Rejected: theta1 = %.6f > 0 at collision (not real heelstrike)\n', yc(1)); end
        return;
    end

    c2t = cos(2*yc(1));
    s2t = sin(2*yc(1));
    theta1dot_new = yc(3) * c2t / (1 + beta_mass * s2t^2);
    theta1_new    = -yc(1);   % always positive since yc(1) < 0

    z_new = [theta1_new; theta1dot_new];
    ok = true;
end


%% =====================================================================
%  EQUATIONS OF MOTION  (Garcia 1998)
%
%  State: y = [theta1, theta2, theta1dot, theta2dot]
%  theta1 = stance leg angle from vertical
%  theta2 = exterior angle between legs (= pi - phi in Garcia's paper)
%
%  M(q)*qdd = -V(q,qd) - G(q)
%  =====================================================================
function ydot = eom_garcia(y, bv)
    beta = bv(1); gam = bv(2); g_val = bv(3); l_val = bv(4);

    c2   = cos(y(2));    s2   = sin(y(2));
    s1g  = sin(y(1) - gam);
    s12g = sin(y(1) + y(2) - gam);

    M = [1 + 2*beta*(1+c2),  beta*(1+c2);
         1 + c2,             1           ];

    % RHS = -V - G  (matching Garcia's yderivs_doubpend.m exactly)
    rhs = [ beta*s2*y(4)*(2*y(3)+y(4)) + (g_val/l_val)*(beta*s12g + s1g*(1+beta));
           -s2*y(3)^2                   + (g_val/l_val)*s12g                       ];

    qdd = M \ rhs;
    ydot = [y(3); y(4); qdd(1); qdd(2)];
end


%% =====================================================================
%  HEELSTRIKE EVENT DETECTION
%  =====================================================================
function [val, ist, dir] = heelstrike_event(~, y)
%   Event 1: theta2 + 2*theta1 - pi = 0  (heelstrike)
%     Collision function sequence during one stride:
%       coll = 0 (start) -> negative (legs open) -> 0 (swing catches up,
%       theta1 still > 0, NOT real collision) -> positive (swing ahead)
%       -> 0 (actual heelstrike, theta1 < 0)
%     Therefore direction = -1 (positive-to-negative crossing)
%   Event 2: |theta1| > pi/2  (divergence guard)
    val = [y(2) + 2*y(1) - pi;
           pi/2 - abs(y(1))      ];
    ist = [1; 1];
    dir = [-1; 0];
end


%% =====================================================================
%  NEWTON-RAPHSON FIXED-POINT FINDER
%  =====================================================================
function [z_fp, T_stride, converged] = find_fp_garcia(z0, gam, beta, g, l, per, opts)
    converged = false;  T_stride = [];  z_fp = z0;
    delta_nr = 1e-7;  max_iter = 20;  tol_fp = 1e-10;

    [~, ~, ok] = stride_map_garcia(z_fp, gam, beta, g, l, per, opts);
    if ~ok, return; end

    for iter = 1:max_iter
        [Sz, T_stride, ok] = stride_map_garcia(z_fp, gam, beta, g, l, per, opts);
        if ~ok, return; end
        res = Sz - z_fp;
        if norm(res) < tol_fp, converged = true; return; end

        J = zeros(2);
        for j = 1:2
            zp = z_fp;  zp(j) = zp(j) + delta_nr;
            zm = z_fp;  zm(j) = zm(j) - delta_nr;
            [Sp,~,ok1] = stride_map_garcia(zp, gam, beta, g, l, per, opts);
            [Sm,~,ok2] = stride_map_garcia(zm, gam, beta, g, l, per, opts);
            if ~ok1 || ~ok2, return; end
            J(:,j) = (Sp - Sm) / (2*delta_nr);
        end

        Jg = J - eye(2);
        if rcond(Jg) < 1e-14, return; end
        z_fp = z_fp + 0.8 * (-Jg \ res);
    end

    [Sz, T_stride, ok] = stride_map_garcia(z_fp, gam, beta, g, l, per, opts);
    if ok && norm(Sz - z_fp) < 1e-8, converged = true; end
end


%% =====================================================================
%  NUMERICAL JACOBIAN (tight tolerances for Floquet multipliers)
%  =====================================================================
function [J, ok] = compute_jacobian_garcia(z_fp, gam, beta, g, l, per, opts)
    delta = 1e-7;
    J = zeros(2);
    ok = true;
    for j = 1:2
        zp = z_fp;  zp(j) = zp(j) + delta;
        zm = z_fp;  zm(j) = zm(j) - delta;
        [Sp,~,ok1] = stride_map_garcia(zp, gam, beta, g, l, per, opts);
        [Sm,~,ok2] = stride_map_garcia(zm, gam, beta, g, l, per, opts);
        if ~ok1 || ~ok2, ok = false; return; end
        J(:,j) = (Sp - Sm) / (2*delta);
    end
end


%% =====================================================================
%  STORE ONE RESULT
%  =====================================================================
function S = store_one(S, idx, z_fp, gam, beta, g, l, per, opts)
    [J, ok] = compute_jacobian_garcia(z_fp, gam, beta, g, l, per, opts);
    if ~ok, return; end
    [~, T, ok2] = stride_map_garcia(z_fp, gam, beta, g, l, per, opts);
    if ~ok2, return; end

    lam = eig(J);
    lam = sort(lam, 'descend', 'ComparisonMethod', 'abs');
    s_len = 2*l*sin(z_fp(1));

    S.s(idx)      = s_len;
    S.v(idx)      = s_len / T;
    S.T(idx)      = T;
    S.lam(idx,:)  = lam.';
    S.z_fp(idx,:) = z_fp.';
    S.valid(idx)  = true;
end


%% =====================================================================
%  PLOTTING
%  =====================================================================
function plot_floquet_garcia(R, names, colors, beta)

figure('Color','w','Name','Floquet Multipliers - Garcia (1998)');

for b = 1:2
    m = R(b).valid;
    if ~any(m), continue; end
    gams = R(b).gam(m); vels = R(b).v(m); la = R(b).lam(m,:);

    subplot(2,2,1), hold on
    plot(gams, abs(la(:,1)), 'o-', 'Color',colors{b}, ...
        'MarkerSize',2, 'LineWidth',1.2, 'MarkerFaceColor',colors{b}, ...
        'DisplayName',[names{b} ' |\lambda_1|'])
    plot(gams, abs(la(:,2)), 's--','Color',colors{b}, ...
        'MarkerSize',1.5,'LineWidth',0.8, 'HandleVisibility','off')

    subplot(2,2,2), hold on
    plot(vels, abs(la(:,1)), 'o-', 'Color',colors{b}, ...
        'MarkerSize',2, 'LineWidth',1.2, 'MarkerFaceColor',colors{b}, ...
        'DisplayName',names{b})
    plot(vels, abs(la(:,2)), 's--','Color',colors{b}, ...
        'MarkerSize',1.5,'LineWidth',0.8, 'HandleVisibility','off')

    subplot(2,2,3), hold on
    plot(vels, real(la(:,1)), 'o-', 'Color',colors{b}, ...
        'MarkerSize',2, 'LineWidth',1.2, 'MarkerFaceColor',colors{b}, ...
        'DisplayName',names{b})
    plot(vels, real(la(:,2)), 's--','Color',colors{b}, ...
        'MarkerSize',1.5,'LineWidth',0.8, 'HandleVisibility','off')

    subplot(2,2,4), hold on
    plot(real(la), imag(la), '.', ...
        'Color',colors{b}, 'MarkerSize',6, 'DisplayName',names{b})
end

subplot(2,2,1)
yline(1,'k--'), xlabel('\gamma [rad]'), ylabel('|\lambda|')
title('Magnitudes vs Slope'), legend('Location','best'), grid on

subplot(2,2,2)
yline(1,'k--'), xlabel('v = s/T'), ylabel('|\lambda|')
title('Magnitudes vs Speed'), legend('Location','best'), grid on

subplot(2,2,3)
yline(1,'k--','LineWidth',0.5), yline(-1,'k--','LineWidth',0.5)
yline(0,'k:','LineWidth',0.5)
xlabel('v = s/T'), ylabel('Re(\lambda)'), title('Real Parts')
legend('Location','best'), grid on

subplot(2,2,4)
th = linspace(0,2*pi,200);
plot(cos(th),sin(th),'k--','LineWidth',0.5,'HandleVisibility','off')
xlabel('Re(\lambda)'), ylabel('Im(\lambda)')
title('Complex Plane'), axis equal, legend('Location','best'), grid on

sgtitle(sprintf('Garcia (1998)  \\beta = %.3f', beta));


figure('Color','w','Name','Spectral Radius - Garcia');
hold on
legend_entries = {};
for b = 1:2
    m = R(b).valid;
    if ~any(m), continue; end
    max_lam = max(abs(R(b).lam(m,:)), [], 2);
    plot(R(b).v(m), max_lam, 'o-', 'Color',colors{b}, ...
        'MarkerSize',3, 'LineWidth',1.2, 'MarkerFaceColor',colors{b})
    legend_entries{end+1} = names{b}; %#ok
end
if ~isempty(legend_entries)
    yline(1,'r--','LineWidth',1.5)
    xlabel('v = s/T'), ylabel('max|\lambda_i|')
    title(sprintf('Spectral Radius  (\\beta = %.3f)', beta))
    legend_entries{end+1} = '|\lambda|=1';
    legend(legend_entries{:}, 'Location','best'), grid on
end


figure('Color','w','Name','Bifurcation Diagram - Garcia');
subplot(2,1,1), hold on
for b = 1:2
    m = find(R(b).valid);
    if isempty(m), continue; end
    max_lam = max(abs(R(b).lam(m,:)), [], 2);
    stable = max_lam < 1;
    idx_st = m(stable);  idx_us = m(~stable);
    if ~isempty(idx_st)
        scatter(R(b).gam(idx_st), R(b).z_fp(idx_st,1)*180/pi, ...
            12, colors{b}, 'filled', 'DisplayName',[names{b} ' (stable)'])
    end
    if ~isempty(idx_us)
        scatter(R(b).gam(idx_us), R(b).z_fp(idx_us,1)*180/pi, ...
            12, colors{b}, 'DisplayName',[names{b} ' (unstable)'])
    end
end
xlabel('\gamma [rad]'), ylabel('\theta_1^* [deg]')
title('Fixed-point stance angle'), legend('Location','best'), grid on

subplot(2,1,2), hold on
for b = 1:2
    m = R(b).valid;
    if ~any(m), continue; end
    plot(R(b).gam(m), R(b).s(m), 'o-', 'Color',colors{b}, ...
        'MarkerSize',3, 'MarkerFaceColor',colors{b}, 'DisplayName',names{b})
end
xlabel('\gamma [rad]'), ylabel('Step length s = 2l sin\theta_1')
title('Step length vs slope'), legend('Location','best'), grid on
sgtitle(sprintf('Garcia (1998)  \\beta = %.3f', beta));


figure('Color','w','Name','Low-Speed Regime');
hold on
has_data = false;
for b = 1:2
    m = find(R(b).valid);
    if isempty(m), continue; end
    gams = R(b).gam(m);
    low  = gams < 0.02;
    if ~any(low), continue; end
    ml = m(low);
    plot(R(b).gam(ml), abs(R(b).lam(ml,1)), 'o-', 'Color',colors{b}, ...
        'MarkerSize',4, 'LineWidth',1.5, 'MarkerFaceColor',colors{b}, ...
        'DisplayName',[names{b} ' |\lambda_1|'])
    plot(R(b).gam(ml), abs(R(b).lam(ml,2)), 's--','Color',colors{b}, ...
        'MarkerSize',3, 'LineWidth',1.0, ...
        'DisplayName',[names{b} ' |\lambda_2|'])
    has_data = true;
end
if has_data
    yline(1,'k--','LineWidth',1)
    xlabel('\gamma [rad]'), ylabel('|\lambda|')
    title(sprintf('Low-speed regime  (\\beta = %.3f)', beta))
    legend('Location','best'), grid on
end

end


%% =====================================================================
%  CONSOLE SUMMARY
%  =====================================================================
function print_summary(R, names)
fprintf('\n%s\n', repmat('=',1,95));
for b = 1:2
    m = find(R(b).valid);
    if isempty(m), continue; end
    fprintf('\n--- %s --- (%d orbits)\n', names{b}, length(m));
    fprintf('%-8s %-8s %-8s %-8s %-10s %-35s %-6s\n', ...
        'gamma','s','v','T','theta1*','lambda','Stable');
    fprintf('%s\n', repmat('-',1,95));
    step = max(1, floor(length(m)/20));
    for k = 1:step:length(m)
        i = m(k);
        lam = R(b).lam(i,:);
        mag_str = '';
        for j = 1:2
            if abs(imag(lam(j))) > 1e-6
                mag_str = [mag_str sprintf('%.4f(%.3f%+.3fi) ', ...
                    abs(lam(j)), real(lam(j)), imag(lam(j)))]; %#ok
            else
                mag_str = [mag_str sprintf('%.5f ', real(lam(j)))]; %#ok
            end
        end
        stable = all(abs(lam) < 1);
        fprintf('%-8.4f %-8.4f %-8.4f %-8.3f %-10.6f %-35s %-6s\n', ...
            R(b).gam(i), R(b).s(i), R(b).v(i), R(b).T(i), ...
            R(b).z_fp(i,1), strtrim(mag_str), string(stable));
    end
end
fprintf('%s\n', repmat('=',1,95));
end



%% =====================================================================
%  CSV EXPORT
%  =====================================================================
function export_csv_garcia(R, names, beta)
csv_name = sprintf('floquet_garcia_beta%.3f.csv', beta);
fid = fopen(csv_name, 'w');
fprintf(fid, ['branch,branch_name,gamma,s,v,T_stride,theta1_fp,theta1dot_fp,' ...
    'lam1_re,lam1_im,lam2_re,lam2_im,abs_lam1,abs_lam2,stable\n']);
for b = 1:2
    m = find(R(b).valid);
    for k = 1:length(m)
        i = m(k);
        lam = R(b).lam(i,:);
        fprintf(fid, '%d,%s,%.8f,%.8f,%.8f,%.8f,%.12f,%.12f,', ...
            b, names{b}, R(b).gam(i), R(b).s(i), R(b).v(i), R(b).T(i), ...
            R(b).z_fp(i,1), R(b).z_fp(i,2));
        fprintf(fid, '%.12f,%.12f,%.12f,%.12f,%.12f,%.12f,%d\n', ...
            real(lam(1)), imag(lam(1)), real(lam(2)), imag(lam(2)), ...
            abs(lam(1)), abs(lam(2)), all(abs(lam) < 1));
    end
end
fclose(fid);
fprintf('\nExported to %s\n', csv_name);

% =====================================================================
% DIAGNOSTIC: |lambda_1 + 1| precision check
% =====================================================================
fprintf('\n=== DIAGNOSTIC: |lambda_1 + 1| ===\n');
fprintf('ODE: RelTol=1e-10, AbsTol=1e-12\n\n');

diag_speeds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50];
deltas = [1e-5, 1e-6, 1e-7, 1e-8];

fprintf('%-8s', 's');
for d = deltas, fprintf('  delta=%.0e   ', d); end
fprintf('\n%s\n', repmat('-',1,72));

for si = 1:length(diag_speeds)
    s = diag_speeds(si);
    alpha = asin(0.5*s);
    omega = -1.04*alpha;
    P = -omega*tan(alpha);
    z0 = [alpha; omega; (1-cos(2*alpha))*omega];

    [z_fp, ~, conv] = find_fixed_point(z0, gam, a, tau, k_hip, P, per, opts_fine);
    if ~conv, continue; end

    fprintf('%-8.3f', s);
    for di = 1:length(deltas)
        delta_d = deltas(di);
        J = zeros(3,3);
        ok_all = true;
        for j = 1:3
            zp = z_fp; zp(j) = zp(j) + delta_d;
            zm = z_fp; zm(j) = zm(j) - delta_d;
            [Sp,~,ok1] = stride_map_reduced(zp, gam, a, tau, k_hip, P, per, opts_fine);
            [Sm,~,ok2] = stride_map_reduced(zm, gam, a, tau, k_hip, P, per, opts_fine);
            if ~ok1||~ok2, ok_all = false; break; end
            J(:,j) = (Sp - Sm) / (2*delta_d);
        end
        if ~ok_all
            fprintf('  FAIL            ');
        else
            lam = eig(J);
            [~, idx_m1] = min(abs(lam + 1));
            dev = abs(lam(idx_m1) + 1);
            fprintf('  %.4e (%+.6f)', dev, real(lam(idx_m1)));
        end
    end
    fprintf('\n');
end
fprintf('%s\n', repmat('=',1,72));



end