% perturbation_sensitivity.m
% Sensitivity check: verify that normalised perturbation decay is
% insensitive to perturbation magnitude (0.1%, 0.3%, 1.0%).
%
% Uses the same subfunctions as return_map_nonlinear.m

clear; close all; clc;

k_hip = -0.16;
T_max = 5;

eps_levels = [0.001, 0.003, 0.010];
eps_labels = {'0.1%', '0.3%', '1.0%'};
colors     = {[0 0.45 0.74], [0.85 0.33 0.10], [0.47 0.67 0.19]};

cases = struct( ...
    's',     {0.080,  0.400}, ...
    'N',     {60,     20}, ...
    'label', {'s = 0.08 (low speed)', 's = 0.40 (plateau)'} ...
);

figure('Position', [100 100 1100 450]);

for ci = 1:length(cases)
    s = cases(ci).s;
    N = cases(ci).N;

    [z_fp, P, T_fp, conv] = find_fixed_point_kuo(s, k_hip, T_max);
    if ~conv, fprintf('FAILED s=%.3f\n', s); continue; end
    v_fp = s / T_fp;

    delta = 1e-7;
    J = zeros(3);
    for j = 1:3
        zp = z_fp; zp(j) = zp(j) + delta;
        zm = z_fp; zm(j) = zm(j) - delta;
        [fp,~,o1] = stride_map_twophase(zp, k_hip, P, T_max);
        [fm,~,o2] = stride_map_twophase(zm, k_hip, P, T_max);
        if ~o1 || ~o2, error('Jacobian failed'); end
        J(:,j) = (fp - fm) / (2*delta);
    end
    lam = eig(J);
    lam_max = max(abs(lam));

    subplot(1, 2, ci);
    hold on; grid on; box on;
    strides = (0:N)';

    for ei = 1:length(eps_levels)
        eps_pert = eps_levels(ei) * abs(z_fp(1));
        z_curr = z_fp;
        z_curr(1) = z_curr(1) + eps_pert;

        theta = NaN(N+1, 1);
        theta(1) = z_curr(1);
        for n = 1:N
            [z_next, ~, ok] = stride_map_twophase(z_curr, k_hip, P, T_max);
            if ~ok, break; end
            theta(n+1) = z_next(1);
            z_curr = z_next;
        end
        theta = theta(~isnan(theta));
        M = length(theta);

        dev = abs(theta - z_fp(1));
        dev_norm = dev / dev(1);

        semilogy((0:M-1)', dev_norm, '.-', ...
            'Color', colors{ei}, 'MarkerSize', 6, 'LineWidth', 1.2, ...
            'DisplayName', ['\epsilon = ' eps_labels{ei}]);
    end

    semilogy(strides, lam_max.^strides, 'k--', 'LineWidth', 1.0, ...
        'DisplayName', sprintf('|\\lambda_{max}|^n = %.3f^n', lam_max));
    yline(0.5, ':', 'Color', [0.5 0.5 0.5], 'LineWidth', 0.8, ...
        'HandleVisibility', 'off');

    xlabel('Stride number');
    ylabel('|\theta_n - \theta^*| / |\theta_0 - \theta^*|');
    title(sprintf('%s\nv = %.3f,  |\\lambda_{max}| = %.4f', ...
        cases(ci).label, v_fp, lam_max));
    legend('Location', 'southwest');
    ylim([1e-5, 3]);
    set(gca, 'YScale', 'log');
end

sgtitle('Perturbation magnitude sensitivity  (k_{hip} = -0.16)', ...
    'FontSize', 13, 'FontWeight', 'bold');


% =========================================================================
%  SUBFUNCTIONS (identical to return_map_nonlinear.m)
% =========================================================================

function [z_fp, P, T_fp, converged] = find_fixed_point_kuo(s, k_hip, T_max)
    converged = false; T_fp = []; P = [];
    alpha = asin(0.5*s);
    omega = -1.04*alpha;
    P = -omega*tan(alpha);
    z_fp = [alpha; omega; (1-cos(2*alpha))*omega];
    delta = 1e-7;
    for iter = 1:20
        [Sz, T, ok] = stride_map_twophase(z_fp, k_hip, P, T_max);
        if ~ok, return; end
        T_fp = T;
        res = Sz - z_fp;
        if norm(res) < 1e-10, converged = true; return; end
        Jg = zeros(3);
        for j = 1:3
            zp = z_fp; zp(j) = zp(j)+delta;
            zm = z_fp; zm(j) = zm(j)-delta;
            [Sp,~,o1] = stride_map_twophase(zp,k_hip,P,T_max);
            [Sm,~,o2] = stride_map_twophase(zm,k_hip,P,T_max);
            if ~o1||~o2, return; end
            Jg(:,j) = (Sp-Sm)/(2*delta);
        end
        Jg = Jg - eye(3);
        if rcond(Jg) < 1e-14, return; end
        z_fp = z_fp + 0.8*(-Jg\res);
    end
    [Sz,T,ok] = stride_map_twophase(z_fp,k_hip,P,T_max);
    if ok && norm(Sz-z_fp)<1e-8, converged=true; T_fp=T; end
end

function [z_new, T_stride, ok] = stride_map_twophase(z, k_hip, P, T_max)
    z_new = []; T_stride = []; ok = false;
    if abs(z(1)) > pi/3, return; end
    y0 = [z(1); z(2); 2*z(1); z(3)];
    dt = 0.005;
    opts1 = odeset('RelTol',1e-12,'AbsTol',1e-14);
    [~, Y1] = ode45(@(t,y) eom_kuo(t,y,k_hip), [0, dt], y0, opts1);
    y_dep = Y1(end,:).';
    opts2 = odeset('RelTol',1e-12,'AbsTol',1e-14,'Events',@heelstrike_event);
    [~, ~, te, ye, ie] = ode45(@(t,y) eom_kuo(t,y,k_hip), [dt, T_max], y_dep, opts2);
    if isempty(ie), return; end
    idx = find(ie == 1, 1);
    if isempty(idx), return; end
    T_stride = te(idx);
    yc = ye(idx,:);
    c2 = cos(2*yc(1));
    s2P = sin(2*yc(1)) * P;
    z_new = [-yc(1); c2*yc(2) + s2P; c2*(1-c2)*yc(2) + (1-c2)*s2P];
    ok = true;
end

function ydot = eom_kuo(~, y, k_hip)
    th  = y(1); thd = y(2);
    ph  = y(3); phd = y(4);
    thdd = sin(th);
    phdd = sin(th) + sin(ph)*(thd^2 - cos(th)) + k_hip*ph;
    ydot = [thd; thdd; phd; phdd];
end

function [val, ist, dir] = heelstrike_event(~, y)
    val = y(3) - 2*y(1);
    ist = 1;
    dir = 1;
end