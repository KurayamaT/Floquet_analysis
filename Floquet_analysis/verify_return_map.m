function verify_return_map
%VERIFY_RETURN_MAP  Independent verification of Floquet multipliers
%
%   This script is INDEPENDENT of floquet_KUO_v2.m.
%   It reimplements the stride map from scratch and verifies:
%     1. Periodic orbit (fixed point) is correctly computed
%     2. Floquet multipliers match the main analysis
%     3. Perturbation decay rate follows |lambda_max|^n
%
%   Two representative speeds:
%     Low-speed regime:  s = 0.08 (v ~ 0.023)
%     Stability plateau: s = 0.40 (v ~ 0.116)
%
%   Three perturbation levels (0.1%, 0.3%, 1.0%) are tested
%   to confirm linearity.

clear; close all; clc;

khip = -0.16;
gam  = 0;
per  = 5;

opts = odeset('RelTol',1e-12,'AbsTol',1e-14,'Refine',4, ...
              'Events',@heelstrike_event);

%% Test cases
test = struct( ...
    's',      {0.08,    0.40}, ...
    'N_iter', {60,      20}, ...
    'label',  {'Low-speed (s=0.08)', 'Plateau (s=0.40)'});

pert_pct = [0.001, 0.003, 0.01];  % 0.1%, 0.3%, 1.0%

figure('Color','w','Position',[50 50 1400 900]);

for ic = 1:2
    s  = test(ic).s;
    Ni = test(ic).N_iter;

    fprintf('\n==============================================\n');
    fprintf(' %s\n', test(ic).label);
    fprintf('==============================================\n');

    alpha = asin(s/2);

    %% 1. Find periodic orbit (Newton-Raphson)
    th0   = alpha;
    thd0  = -1.04*alpha;
    P_imp = -thd0 * tan(alpha);
    phid0 = (1 - cos(2*th0)) * thd0;
    z = [th0; thd0; phid0];

    for iter = 1:30
        [zf, T] = stride(z, gam, khip, P_imp, per, opts);
        res = zf - z;
        if norm(res) < 1e-10
            fprintf('  Fixed point converged at iter %d\n', iter);
            break
        end
        J_nr = jacobian_cd(z, gam, khip, P_imp, per, opts);
        z = z - (J_nr - eye(3)) \ res;
    end
    z_fp = z;
    T_fp = T;
    v_fp = s / T_fp;

    %% 2. Compute Floquet multipliers
    J = jacobian_cd(z_fp, gam, khip, P_imp, per, opts);
    lam = eig(J);
    [~, ord] = sort(abs(lam),'descend');
    lam = lam(ord);
    lam_max = max(abs(lam));

    fprintf('  v = %.6f,  T = %.6f\n', v_fp, T_fp);
    fprintf('  z* = [%.8f, %.8f, %.8f]\n', z_fp);
    fprintf('  lambda = ');
    for k = 1:3
        if abs(imag(lam(k))) < 1e-8
            fprintf('%.6f  ', real(lam(k)));
        else
            fprintf('%.4f%+.4fi  ', real(lam(k)), imag(lam(k)));
        end
    end
    fprintf('\n  |lambda_max| = %.6f\n', lam_max);
    fprintf('  N_1/2 = %.1f strides\n', -log(2)/log(lam_max));

    %% 3. Perturbation sweep (0.1%, 0.3%, 1.0%)
    fprintf('\n  Perturbation linearity check:\n');
    fprintf('  %8s  %12s  %10s\n', 'delta%', '|lam_fit|', 'rel_err%');
    fprintf('  %s\n', repmat('-',1,34));

    fitted_lam = zeros(1,3);
    for ip = 1:3
        delta = pert_pct(ip) * abs(z_fp(1));
        z_pert = z_fp;
        z_pert(1) = z_pert(1) + delta;

        err = zeros(1, Ni);
        for n = 1:Ni
            z_pert = stride(z_pert, gam, khip, P_imp, per, opts);
            err(n) = norm(z_pert - z_fp);
        end

        % Fit |lambda_max| from log-linear regression
        nn = (1:Ni)';
        above = err > 1e-12;
        if sum(above) > 5
            p = polyfit(nn(above), log(err(above))', 1);
            fitted_lam(ip) = exp(p(1));
        end
        rel_err = abs(fitted_lam(ip) - lam_max) / lam_max * 100;
        fprintf('  %7.1f%%  %12.6f  %9.2f%%\n', ...
            pert_pct(ip)*100, fitted_lam(ip), rel_err);
    end

    %% 4. Plots (using 0.3% perturbation)
    delta = 0.003 * abs(z_fp(1));
    z_pert = z_fp;
    z_pert(1) = z_pert(1) + delta;

    th_hist = zeros(1, Ni);
    thd_hist = zeros(1, Ni);
    err_hist = zeros(1, Ni);

    for n = 1:Ni
        z_pert = stride(z_pert, gam, khip, P_imp, per, opts);
        th_hist(n)  = z_pert(1);
        thd_hist(n) = z_pert(2);
        err_hist(n) = norm(z_pert - z_fp);
    end

    row = (ic-1)*3;

    % (a) Cobweb diagram
    subplot(2,3,row+1); hold on;
    th_all = [z_fp(1)+delta, th_hist];
    mn = min(th_all); mx = max(th_all); pad = 0.1*(mx-mn);
    fplot(@(x) x, [mn-pad mx+pad], 'k-', 'LineWidth',0.5);
    % stride map curve (sample around fixed point)
    th_sample = linspace(mn-pad, mx+pad, 80);
    th_next = zeros(size(th_sample));
    for j = 1:length(th_sample)
        zt = z_fp; zt(1) = th_sample(j);
        zt = stride(zt, gam, khip, P_imp, per, opts);
        th_next(j) = zt(1);
    end
    plot(th_sample, th_next, 'b-', 'LineWidth', 1.5);
    % cobweb lines
    for n = 1:length(th_hist)-1
        plot([th_all(n) th_all(n)], [th_all(n) th_all(n+1)], 'r-', 'LineWidth',0.8);
        plot([th_all(n) th_all(n+1)], [th_all(n+1) th_all(n+1)], 'r-', 'LineWidth',0.8);
    end
    plot(z_fp(1), z_fp(1), 'kx', 'MarkerSize',10, 'LineWidth',2);
    xlabel('\theta_n'); ylabel('\theta_{n+1}');
    title(sprintf('(%s) Cobweb: %s', char('a'+row), test(ic).label));
    grid on;

    % (b) Phase portrait
    subplot(2,3,row+2); hold on;
    dth  = th_hist - z_fp(1);
    dthd = thd_hist - z_fp(2);
    plot(dth, dthd, 'b.-', 'MarkerSize',8);
    plot(dth(1), dthd(1), 'ro', 'MarkerSize',8, 'LineWidth',2);
    plot(0, 0, 'kx', 'MarkerSize',10, 'LineWidth',2);
    xlabel('\delta\theta'); ylabel('\delta\theta-dot');
    title(sprintf('(%s) Phase portrait', char('b'+row)));
    grid on; axis equal;

    % (c) Perturbation decay
    subplot(2,3,row+3); hold on;
    nn = 1:Ni;
    semilogy(nn, err_hist/err_hist(1), 'bo-', 'MarkerSize',4, 'LineWidth',1.2);
    semilogy(nn, lam_max.^nn, 'k--', 'LineWidth',1);
    yline(0.5, ':r', 'LineWidth',1);
    xlabel('Stride number'); ylabel('||{\delta}x_n|| / ||{\delta}x_0||');
    title(sprintf('(%s) Decay: |\\lambda_{max}|=%.3f, N_{1/2}=%.1f', ...
        char('c'+row), lam_max, -log(2)/log(lam_max)));
    legend('Simulation','|\lambda_{max}|^n','Half-life','Location','northeast');
    grid on;
end

sgtitle('Independent Verification: Return Map Iteration (k_{hip} = -0.16)', ...
    'FontSize', 14, 'FontWeight', 'bold');
saveas(gcf, 'verify_return_map.png');
fprintf('\nFigure saved: verify_return_map.png\n');
fprintf('Done.\n');

end


%% ========================================================================
% SUBFUNCTIONS (independent reimplementation)
%% ========================================================================

function [z_new, T] = stride(z, gam, khip, P, per, opts)
% One stride: swing phase + push-off + collision

    y0 = [z(1); z(2); 2*z(1); z(3)];

    % Phase 1: depart collision surface
    dt = 0.005;
    opts_noevent = odeset('RelTol',1e-12,'AbsTol',1e-14);
    [~, yout1] = ode45(@(t,y) eom(t,y,gam,khip), [0 dt], y0, opts_noevent);
    y_dep = yout1(end,:).';

    % Phase 2: integrate to next heelstrike
    [~, ~, te, ye, ie] = ode45(@(t,y) eom(t,y,gam,khip), [dt per], y_dep, opts);

    idx_c = find(ie == 1, 1);
    if isempty(idx_c), error('No heelstrike detected'); end

    T  = te(idx_c);
    yc = ye(idx_c,:);

    % Push-off + collision
    c2 = cos(2*yc(1));
    sP = sin(2*yc(1)) * P;
    z_new = [-yc(1);
              c2*yc(2) + sP;
              c2*(1-c2)*yc(2) + (1-c2)*sP];
end


function J = jacobian_cd(z, gam, khip, P, per, opts)
% Central difference Jacobian
    eps = 1e-7;
    J = zeros(3);
    for j = 1:3
        zp = z; zp(j) = zp(j) + eps;
        zm = z; zm(j) = zm(j) - eps;
        fp = stride(zp, gam, khip, P, per, opts);
        fm = stride(zm, gam, khip, P, per, opts);
        J(:,j) = (fp - fm) / (2*eps);
    end
end


function ydot = eom(~, y, gam, khip)
    ydot = [y(2);
            sin(y(1) - gam);
            y(4);
            sin(y(1)-gam) + sin(y(3))*(y(2)^2 - cos(y(1)-gam)) + khip*y(3)];
end


function [val, ist, dir] = heelstrike_event(~, y)
    val = [y(3) - 2*y(1);
           pi/2 - abs(y(1))];
    ist = [1; 1];
    dir = [1; 0];
end