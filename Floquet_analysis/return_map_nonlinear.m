% return_map_nonlinear.m
% Nonlinear stride map simulation with cobweb diagram
% Kuo (2002) simplest walking model, k_hip = -0.16
%
% Reproduces the Python verification: perturb from fixed point,
% iterate the full stride map, plot cobweb + deviation decay.

clear; close all; clc;

k_hip = -0.16;
T_max = 5;

cases = struct( ...
    's',       {0.080,  0.130,  0.320,  0.580}, ...
    'N',       {80,     50,     30,     25}, ...
    'label',   {'v~0.023 low','v~0.038 trans','v~0.093 plateau','v~0.167 high'} ...
);

figure('Position',[50 50 1600 800]);

for ci = 1:length(cases)
    s = cases(ci).s;
    N = cases(ci).N;

    % --- Find fixed point ---
    [z_fp, P, T_fp, conv] = find_fixed_point_kuo(s, k_hip, T_max);
    if ~conv
        fprintf('FAILED s=%.3f\n', s);
        continue;
    end
    v_fp = s / T_fp;

    % --- Eigenvalues via Jacobian ---
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
    is_cc = any(abs(imag(lam)) > 1e-6);
    nhalf = -log(2)/log(lam_max);

    fprintf('s=%.3f  v=%.4f  |lam_max|=%.4f  N1/2=%.1f  CC=%d\n', ...
        s, v_fp, lam_max, nhalf, is_cc);

    % --- Perturb and iterate ---
    eps_pert = 0.002 * abs(z_fp(1));
    z_curr = z_fp;
    z_curr(1) = z_curr(1) + eps_pert;

    theta = NaN(N+1, 1);
    theta(1) = z_curr(1);
    for n = 1:N
        [z_next, ~, ok] = stride_map_twophase(z_curr, k_hip, P, T_max);
        if ~ok
            fprintf('  stride %d failed\n', n);
            break;
        end
        theta(n+1) = z_next(1);
        z_curr = z_next;
    end
    theta = theta(~isnan(theta));
    M = length(theta);

    % =====================================================================
    % Plot: cobweb (top) + deviation (bottom)
    % =====================================================================

    % --- Cobweb ---
    subplot(2,4,ci);
    hold on; grid on; box on;
    th_n  = theta(1:end-1);
    th_n1 = theta(2:end);

    % Cobweb lines
    Nweb = min(length(th_n)-1, 50);
    for i = 1:Nweb
        plot([th_n(i), th_n(i)],   [th_n(i), th_n1(i)], 'b-', 'LineWidth', 0.4, 'Color', [0 0 1 0.3]);
        plot([th_n(i), th_n1(i)],  [th_n1(i), th_n1(i)],'b-', 'LineWidth', 0.4, 'Color', [0 0 1 0.3]);
    end
    plot(th_n, th_n1, 'b.', 'MarkerSize', 6);
    margin = max(max(abs(theta - z_fp(1)))*1.5, 0.001);
    xl = [z_fp(1)-margin, z_fp(1)+margin];
    plot(xl, xl, 'k--', 'LineWidth', 0.8);
    plot(z_fp(1), z_fp(1), 'r+', 'MarkerSize', 14, 'LineWidth', 2.5);
    plot(th_n(1), th_n1(1), 'go', 'MarkerSize', 7, 'MarkerFaceColor', 'g');
    xlim(xl); ylim(xl);
    axis square;
    xlabel('\theta_n (rad)');
    ylabel('\theta_{n+1} (rad)');
    if is_cc, tstr = 'CC'; else, tstr = 'Real'; end
    title(sprintf('v = %.3f  (%s)\n|\\lambda_{max}| = %.3f,  N_{1/2} = %.1f', ...
        v_fp, tstr, lam_max, nhalf));

    % --- Deviation ---
    subplot(2,4,4+ci);
    hold on; grid on; box on;
    dev = abs(theta - z_fp(1));
    dev_norm = dev / dev(1);
    strides = (0:M-1)';

    semilogy(strides, dev_norm, 'b.-', 'MarkerSize', 4, 'LineWidth', 0.8);
    yline(0.5, 'r--', 'LineWidth', 0.8);
    semilogy(strides, lam_max.^strides, ':', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.2);
    xlabel('Stride number');
    ylabel('|\theta_n - \theta^*| / |\theta_0 - \theta^*|');
    ylim([1e-4, 3]);
    xlim([0, M-1]);
    set(gca, 'YScale', 'log');
    if ci == 1
        legend('simulation', 'half-life', '|\lambda_{max}|^n', 'Location', 'southwest');
    end
end

sgtitle('Nonlinear stride map simulation  (k_{hip} = -0.16)', 'FontSize', 13, 'FontWeight', 'bold');


% =========================================================================
%  SUBFUNCTIONS
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

    % Phase 1: depart collision surface (no event detection)
    dt = 0.005;
    opts1 = odeset('RelTol',1e-12,'AbsTol',1e-14);
    [~, Y1] = ode45(@(t,y) eom_kuo(t,y,k_hip), [0, dt], y0, opts1);
    y_dep = Y1(end,:).';

    % Phase 2: detect next heel-strike
    opts2 = odeset('RelTol',1e-12,'AbsTol',1e-14,'Events',@heelstrike_event);
    [~, ~, te, ye, ie] = ode45(@(t,y) eom_kuo(t,y,k_hip), [dt, T_max], y_dep, opts2);
    if isempty(ie), return; end
    idx = find(ie == 1, 1);
    if isempty(idx), return; end

    T_stride = te(idx);
    yc = ye(idx,:);

    % Push-off + collision map
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
    val = y(3) - 2*y(1);   % phi - 2*theta = 0
    ist = 1;
    dir = 1;
end