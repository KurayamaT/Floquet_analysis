% return_map_data_v6.m
% Perturb along the lambda_2 eigenvector to isolate lambda_2 decay.
% This eliminates the lambda_1 = -1 oscillation from the time series.

gam = 0; a = 0; tau = 3.84; k_hip = -0.08; per = 5;

opts_nr   = odeset('RelTol',1e-2, 'AbsTol',1e-6, 'Refine',4,'Events',@collision_with_guard);
opts_fine = odeset('RelTol',1e-12,'AbsTol',1e-14,'Refine',4,'Events',@collision_with_guard);

cases = struct( ...
    'label', {'low', 'high'}, ...
    's',     {0.230,  0.700}, ...
    'N',     {100,     30}     ...
);

delta_jac = 1e-7;  % perturbation for Jacobian

for ci = 1:length(cases)
    s = cases(ci).s;
    N = cases(ci).N;
    label = cases(ci).label;

    alpha = asin(0.5*s);
    omega = -1.04*alpha;
    P = -omega*tan(alpha);
    z0 = [alpha; omega; (1-cos(2*alpha))*omega];

    % Find fixed point
    [z_fp, T_fp, conv] = find_fixed_point(z0, gam, a, tau, k_hip, P, per, opts_nr);
    if ~conv
        fprintf('FAILED to converge for s=%.3f (%s)\n', s, label);
        continue;
    end
    v_fp = s / T_fp;

    % Compute Jacobian at fixed point (tight tolerances, SAME map as NR)
    J = zeros(3);
    ok_all = true;
    for j = 1:3
        zp = z_fp; zp(j) = zp(j) + delta_jac;
        zm = z_fp; zm(j) = zm(j) - delta_jac;
        [Sp,~,o1] = stride_map_reduced(zp, gam, a, tau, k_hip, P, per, opts_fine);
        [Sm,~,o2] = stride_map_reduced(zm, gam, a, tau, k_hip, P, per, opts_fine);
        if ~o1||~o2, ok_all = false; break; end
        J(:,j) = (Sp - Sm) / (2*delta_jac);
    end
    if ~ok_all
        fprintf('  Jacobian failed for %s\n', label);
        continue;
    end

    [V, D] = eig(J);
    lam = diag(D);
    fprintf('%s: eigenvalues = [%.6f, %.6f, %.6f]\n', label, lam(1), lam(2), lam(3));

    % Find lambda_2: the eigenvalue that is NOT near -1 and NOT near 0
    [~, idx_sort] = sort(abs(lam), 'descend');
    lam_sorted = lam(idx_sort);
    V_sorted = V(:, idx_sort);
    % idx_sort(1) should be lambda_1 ~ -1
    % idx_sort(2) should be lambda_2
    % idx_sort(3) should be lambda_3 ~ 0
    lam2_val = lam_sorted(2);
    v2 = real(V_sorted(:, 2));  % eigenvector for lambda_2
    v2 = v2 / norm(v2);        % normalize

    fprintf('  lambda_2 = %.6f, eigenvector = [%.6f, %.6f, %.6f]\n', ...
        lam2_val, v2(1), v2(2), v2(3));

    % Perturb along lambda_2 eigenvector
    eps_pert = 0.001 * alpha;  % small enough to stay in linear regime
    z_pert = z_fp + eps_pert * v2;

    fprintf('  z_fp   = [%.8f, %.8f, %.8f]\n', z_fp);
    fprintf('  z_pert = [%.8f, %.8f, %.8f]\n', z_pert);

    % Iterate stride map
    theta_n = NaN(N+1, 1);
    thetadot_n = NaN(N+1, 1);
    theta_n(1) = z_pert(1);
    thetadot_n(1) = z_pert(2);
    z_curr = z_pert;
    for n = 1:N
        [z_next, ~, ok] = stride_map_twophase(z_curr, gam, a, tau, k_hip, P, per, opts_fine);
        if ~ok
            fprintf('  %s: stride map failed at stride %d\n', label, n);
            break;
        end
        theta_n(n+1) = z_next(1);
        thetadot_n(n+1) = z_next(2);
        z_curr = z_next;
    end

    % Trim NaN
    valid = find(~isnan(theta_n), 1, 'last');
    theta_n = theta_n(1:valid);
    thetadot_n = thetadot_n(1:valid);

    % Save CSV
    M = length(theta_n) - 1;
    fname = sprintf('return_map_%s.csv', label);
    fid = fopen(fname, 'w');
    fprintf(fid, 'stride,theta_n,theta_n1,thetadot_n,thetadot_n1,theta_fp,thetadot_fp,v,lambda2\n');
    for i = 1:M
        fprintf(fid, '%d,%.12f,%.12f,%.12f,%.12f,%.12f,%.12f,%.6f,%.6f\n', ...
            i-1, theta_n(i), theta_n(i+1), ...
            thetadot_n(i), thetadot_n(i+1), ...
            z_fp(1), z_fp(2), v_fp, abs(lam2_val));
    end
    fclose(fid);
    fprintf('  Saved %s (%d strides)\n\n', fname, M);
end


% =========================================================================
% TWO-PHASE STRIDE MAP
% =========================================================================
function [z_new, T_stride, ok] = stride_map_twophase(z, gam, a, tau, k, P, per, opts)
    z_new = []; T_stride = []; ok = false;
    if abs(z(1)) > pi/3, return; end
    y0 = [z(1); z(2); 2*z(1); z(3)];
    dt_depart = 0.005;
    opts_noevent = odeset('RelTol',1e-10, 'AbsTol',1e-12);
    [~, yout1] = ode45(@(t,y)f(t,y,gam,a,tau,k), [0, dt_depart], y0, opts_noevent);
    y_depart = yout1(end,:).';
    [~, ~, te, ye, ie] = ode45(@(t,y)f(t,y,gam,a,tau,k), ...
        [dt_depart, per], y_depart, opts);
    if isempty(ie), return; end
    idx_coll = find(ie == 1);
    if isempty(idx_coll), return; end
    ic = idx_coll(1);
    T_stride = te(ic);
    yc = ye(ic, :);
    c2 = cos(2*yc(1)); s2p = sin(2*yc(1))*P;
    z_new = [-yc(1); c2*yc(2)+s2p; c2*(1-c2)*yc(2)+(1-c2)*s2p];
    ok = true;
end


% =========================================================================
% ORIGINAL SUBFUNCTIONS (for Newton-Raphson)
% =========================================================================
function [z_fp, T_stride, converged] = find_fixed_point(z0, gam, a, tau, k, P, per, opts)
    converged = false; T_stride = []; z_fp = z0;
    delta_nr = 1e-7; max_iter = 12; tol = 1e-10;
    [~, ~, ok] = stride_map_reduced(z_fp, gam, a, tau, k, P, per, opts);
    if ~ok, return; end
    for iter = 1:max_iter
        [Sz, T, ok] = stride_map_reduced(z_fp, gam, a, tau, k, P, per, opts);
        if ~ok, return; end
        T_stride = T;
        res = Sz - z_fp;
        if norm(res) < tol, converged = true; return; end
        Jg = zeros(3);
        for j = 1:3
            zp = z_fp; zp(j)=zp(j)+delta_nr;
            zm = z_fp; zm(j)=zm(j)-delta_nr;
            [Sp,~,ok1] = stride_map_reduced(zp,gam,a,tau,k,P,per,opts);
            [Sm,~,ok2] = stride_map_reduced(zm,gam,a,tau,k,P,per,opts);
            if ~ok1||~ok2, return; end
            Jg(:,j) = (Sp-Sm)/(2*delta_nr);
        end
        Jg = Jg - eye(3);
        if rcond(Jg) < 1e-14, return; end
        z_fp = z_fp + 0.8*(-Jg\res);
    end
    [Sz,T,ok] = stride_map_reduced(z_fp,gam,a,tau,k,P,per,opts);
    if ok && norm(Sz-z_fp)<1e-8, converged=true; T_stride=T; end
end

function [z_new, T_stride, ok] = stride_map_reduced(z, gam, a, tau, k, P, per, opts)
    z_new = []; T_stride = []; ok = false;
    if abs(z(1)) > pi/3, return; end
    y0 = [z(1); z(2); 2*z(1); z(3)];
    [tout, yout, te, ~, ie] = ode45(@(t,y)f(t,y,gam,a,tau,k), [0 per], y0, opts);
    if isempty(te), return; end
    if isempty(ie) || ie(end) ~= 1, return; end
    T_stride = tout(end);
    ye = yout(end,:);
    c2 = cos(2*ye(1)); s2p = sin(2*ye(1))*P;
    z_new = [-ye(1); c2*ye(2)+s2p; c2*(1-c2)*ye(2)+(1-c2)*s2p];
    ok = true;
end

function ydot = f(t,y,gam,a,tau,k)
    F = a*sin(2*pi/tau*t) + k*y(3);
    ydot = [y(2); sin(y(1)-gam); y(4);
            sin(y(1)-gam)+sin(y(3))*(y(2)^2-cos(y(1)-gam))+F];
end

function [val,ist,dir] = collision_with_guard(t,y) %#ok<INUSL>
    val = [y(3)-2*y(1); pi/2 - abs(y(1))];
    ist = [1; 1];
    dir = [1; 0];
end