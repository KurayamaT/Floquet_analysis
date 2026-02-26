% return_map_data.m
% Generate stride-to-stride return map data for supplementary figure.
% Requires: stride_map_reduced, find_fixed_point, f, collision_with_guard
%           from floquet_KUO.m (paste them at the bottom or ensure they are on the path)
%
% Outputs: return_map_low.csv  (low speed case)
%          return_map_high.csv (high speed case)

gam = 0; a = 0; tau = 3.84; k_hip = -0.08; per = 5;
opts_nr   = odeset('RelTol',1e-6, 'AbsTol',1e-8, 'Refine',4,'Events',@collision_with_guard);
opts_fine = odeset('RelTol',1e-12,'AbsTol',1e-14,'Refine',4,'Events',@collision_with_guard);

% Two cases: low speed and high speed
cases = struct( ...
    'label', {'low', 'high'}, ...
    's',     {0.291,  0.700}, ...  % step lengths (from CSV: v≈0.080, v≈0.200)
    'N',     {60,     30}     ...  % number of strides to iterate
);

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
        fprintf('FAILED to converge for s=%.3f\n', s);
        continue;
    end
    v_fp = s / T_fp;
    fprintf('s=%.3f: v=%.4f, T=%.4f, z_fp=[%.8f, %.8f, %.8f]\n', ...
        s, v_fp, T_fp, z_fp);

    % Compute eigenvalues at fixed point
    delta = 1e-7;
    J = zeros(3);
    for j = 1:3
        zp = z_fp; zp(j) = zp(j) + delta;
        zm = z_fp; zm(j) = zm(j) - delta;
        [Sp,~,ok1] = stride_map_reduced(zp, gam, a, tau, k_hip, P, per, opts_fine);
        [Sm,~,ok2] = stride_map_reduced(zm, gam, a, tau, k_hip, P, per, opts_fine);
        if ~ok1||~ok2, fprintf('  Jacobian failed\n'); break; end
        J(:,j) = (Sp - Sm) / (2*delta);
    end
    lam = eig(J);
    fprintf('  eigenvalues: '); fprintf('%.4f ', abs(lam)); fprintf('\n');

    % Perturbation: add small offset to theta
    delta_theta = 0.005 * alpha;  % 0.5% of alpha
    z_pert = z_fp + [delta_theta; 0; 0];

    % Iterate stride map and record theta_n
    theta_n = NaN(N+1, 1);
    theta_n(1) = z_pert(1);
    z_curr = z_pert;
    for n = 1:N
        [z_next, ~, ok] = stride_map_reduced(z_curr, gam, a, tau, k_hip, P, per, opts_fine);
        if ~ok
            fprintf('  Stride map failed at stride %d\n', n);
            break;
        end
        theta_n(n+1) = z_next(1);
        z_curr = z_next;
    end

    % Trim NaN
    valid = find(~isnan(theta_n), 1, 'last');
    theta_n = theta_n(1:valid);

    % Save CSV: columns = [stride_index, theta_n, theta_n+1, theta_fp]
    M = length(theta_n) - 1;
    out = [(0:M-1)', theta_n(1:M), theta_n(2:M+1), repmat(z_fp(1), M, 1)];
    fname = sprintf('return_map_%s.csv', label);
    fid = fopen(fname, 'w');
    fprintf(fid, 'stride,theta_n,theta_n1,theta_fp\n');
    for i = 1:M
        fprintf(fid, '%d,%.12f,%.12f,%.12f\n', out(i,1), out(i,2), out(i,3), out(i,4));
    end
    fclose(fid);
    fprintf('  Saved %s (%d strides)\n', fname, M);
end


% =========================================================================
% SUBFUNCTIONS (from floquet_KUO.m)
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