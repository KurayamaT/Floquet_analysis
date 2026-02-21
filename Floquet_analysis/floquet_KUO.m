function floquet_KUO(gam, k_hip)
%FLOQUET_KUO  Floquet multiplier analysis for Kuo (2002) walking model
%   Uses parfor for parallel computation across step lengths.
%
%   FLOQUET_KUO() runs with default parameters (gam=0, k=-0.08)
%   FLOQUET_KUO(GAM) specifies slope angle in radians
%   FLOQUET_KUO(GAM, K) specifies slope and hip spring constant

if nargin < 1, gam = 0; end
if nargin < 2, k_hip = -0.08; end

a = 0;
tau = 3.84;
per = 5;        % Max integration time per stride (stride period ~ pi)

s_range = linspace(0.05, 1.5, 500);
N = length(s_range);

% Pre-allocate sliced output arrays
speeds_out = NaN(1, N);
s_out      = NaN(1, N);
lam_out    = complex(NaN(N, 3));
z_fp_out   = NaN(N, 3);
T_out      = NaN(1, N);
valid_out  = false(1, N);

% Two-tier tolerances:
%   opts_nr   - loose, for Newton-Raphson iterations (speed)
%   opts_fine - tight, for final Jacobian only (accuracy)
opts_nr   = odeset('RelTol',1e-2, 'AbsTol',1e-6, 'Refine',4, ...
                   'Events',@collision_with_guard);
opts_fine = odeset('RelTol',1e-12,'AbsTol',1e-14,'Refine',4, ...
                   'Events',@collision_with_guard);



% Progress reporting
dq = parallel.pool.DataQueue;
progress_tracker('init', N);
afterEach(dq, @(~) progress_tracker('update', 0));

for idx = 1:N
    s = s_range(idx);
    alpha = asin(0.5*s);
    omega = -1.04*alpha;
    P = -omega*tan(alpha);
    z0 = [alpha; omega; (1-cos(2*alpha))*omega];

    % Newton-Raphson with LOOSE tolerances
    [z_fp, T_stride, converged] = find_fixed_point( ...
        z0, gam, a, tau, k_hip, P, per, opts_nr);
    if ~converged, send(dq, idx); continue; end

    % Final Jacobian with TIGHT tolerances
    delta = 1e-7;
    J = zeros(3, 3);
    ok_all = true;
    for j = 1:3
        zp = z_fp; zp(j) = zp(j) + delta;
        zm = z_fp; zm(j) = zm(j) - delta;
        [Sp, ~, ok1] = stride_map_reduced(zp, gam, a, tau, k_hip, P, per, opts_fine);
        [Sm, ~, ok2] = stride_map_reduced(zm, gam, a, tau, k_hip, P, per, opts_fine);
        if ~ok1 || ~ok2, ok_all = false; break; end
        J(:,j) = (Sp - Sm) / (2*delta);
    end
    if ~ok_all, send(dq, idx); continue; end

    lambda = eig(J);
    lambda = sort(lambda, 'descend', 'ComparisonMethod', 'abs');

    speeds_out(idx) = s / T_stride;
    s_out(idx)      = s;
    lam_out(idx,:)  = lambda.';
    z_fp_out(idx,:) = z_fp.';
    T_out(idx)      = T_stride;
    valid_out(idx)  = true;
    send(dq, idx);
end

elapsed = toc;
fprintf('Done in %.2f sec. Found %d / %d periodic orbits.\n', elapsed, sum(valid_out), N);

% Filter & sort
mask    = valid_out;
speeds  = speeds_out(mask);
s_vals  = s_out(mask);
lam_all = lam_out(mask, :);
T_vals  = T_out(mask);
n_found = sum(mask);

if n_found == 0, error('No periodic orbits found.'); end

[speeds, si] = sort(speeds);
s_vals  = s_vals(si);
lam_all = lam_all(si, :);
T_vals  = T_vals(si);

% =====================================================================
% PLOTS
% =====================================================================
markers = {'o-','s-','d-'};
colors  = [0.85 0.33 0.10; 0.00 0.45 0.74; 0.47 0.67 0.19];

figure('Color','w','Name','Floquet Multipliers vs Speed')

subplot(2,2,1), hold on
for j = 1:3
    plot(speeds, abs(lam_all(:,j)), markers{j}, ...
        'Color',colors(j,:),'MarkerSize',3,'LineWidth',1.2,'MarkerFaceColor',colors(j,:))
end
yline(1,'k--'), xlabel('v = s/T'), ylabel('|\lambda_i|')
title('Magnitudes'), legend('\lambda_1','\lambda_2','\lambda_3','Location','best'), grid on

subplot(2,2,2), hold on
for j = 1:3
    plot(s_vals, abs(lam_all(:,j)), markers{j}, ...
        'Color',colors(j,:),'MarkerSize',3,'LineWidth',1.2,'MarkerFaceColor',colors(j,:))
end
yline(1,'k--'), xlabel('Step length s'), ylabel('|\lambda_i|')
title('Magnitudes vs Step Length'), grid on

subplot(2,2,3), hold on
for j = 1:3
    plot(speeds, real(lam_all(:,j)), markers{j}, ...
        'Color',colors(j,:),'MarkerSize',3,'LineWidth',1.2,'MarkerFaceColor',colors(j,:))
end
yline(1,'k--','LineWidth',0.5), yline(-1,'k--','LineWidth',0.5), yline(0,'k:','LineWidth',0.5)
xlabel('v = s/T'), ylabel('Re(\lambda_i)'), title('Real Parts'), grid on

subplot(2,2,4), hold on
th = linspace(0,2*pi,200);
plot(cos(th),sin(th),'k--','LineWidth',0.5)
cmap = parula(n_found);
for i = 1:n_found
    plot(real(lam_all(i,:)), imag(lam_all(i,:)), '.', 'Color',cmap(i,:),'MarkerSize',8)
end
colormap(parula), colorbar, xlabel('Re(\lambda)'), ylabel('Im(\lambda)')
title('Complex Plane'), axis equal, grid on

% --- Separate marginal (|lam|~1) from non-trivial multipliers ---
tol_marginal = 0.01;   % only treat as marginal if |1 - |lambda|| < tol
lam_nontrivial = NaN(n_found, 2);
lam_marginal   = NaN(n_found, 1);
max_nontrivial = NaN(n_found, 1);
for i = 1:n_found
    la = abs(lam_all(i,:));
    near1 = abs(la - 1) < tol_marginal;
    if any(near1)
        % Pick one marginal, rest are non-trivial
        idx_near1 = find(near1, 1, 'first');
        idx_rest  = setdiff(1:3, idx_near1);
        lam_marginal(i)     = la(idx_near1);
        lam_nontrivial(i,:) = sort(la(idx_rest), 'descend');
    else
        % No marginal: all are non-trivial, use largest as "marginal" placeholder
        [~, ix] = sort(la, 'descend');
        lam_marginal(i)     = la(ix(1));
        lam_nontrivial(i,:) = la(ix(2:3));
    end
    max_nontrivial(i) = max(lam_nontrivial(i,:));
    % Override: if ANY |lambda| > 1+tol, orbit is unstable regardless
    if any(la > 1 + tol_marginal)
        max_nontrivial(i) = max(la);
    end
end

% Figure 2: Full spectral radius
figure('Color','w','Name','Spectral Radius vs Speed')
max_lam = max(abs(lam_all),[],2);
hold on
plot(speeds, max_lam, 'k.-','MarkerSize',8,'LineWidth',1.2)
yline(1,'r--','LineWidth',1.5)
xlabel('v = s/T'), ylabel('max|\lambda_i|')
title(sprintf('Full Spectral Radius (\\gamma=%.3f, k=%.3f)', gam, k_hip))
grid on

% Figure 3: Non-trivial multiplier stability
figure('Color','w','Name','Non-trivial Floquet Multipliers')
subplot(2,1,1), hold on
plot(speeds, lam_nontrivial(:,1), 'bo-','MarkerSize',4,'LineWidth',1.2,'MarkerFaceColor','b')
plot(speeds, lam_nontrivial(:,2), 'rs-','MarkerSize',3,'LineWidth',1.0,'MarkerFaceColor','r')
yline(1,'k--','LineWidth',1)
xlabel('v = s/T'), ylabel('|\lambda|')
title('Non-trivial Multipliers (marginal \lambda\approx1 excluded)')
legend('|\lambda_2|','|\lambda_3|','Boundary','Location','best'), grid on

subplot(2,1,2), hold on
plot(speeds, lam_marginal, 'k.-','MarkerSize',8,'LineWidth',1.2)
yline(1,'r--','LineWidth',1)
xlabel('v = s/T'), ylabel('|\lambda_{marginal}|')
title('Marginal Multiplier (energy-neutral direction)')
ylim([0.999 1.001]), grid on

% Console summary
fprintf('\n%s\n', repmat('=',1,80));
fprintf('%-8s %-8s %-8s %-40s %-6s\n','s','v','T','|lambda|','Stable*');
fprintf('%s\n', repmat('-',1,80));
step_print = max(1,floor(n_found/15));
for i = 1:step_print:n_found
    lam = lam_all(i,:);
    mag_str = '';
    for j = 1:3
        if imag(lam(j))~=0
            mag_str = [mag_str sprintf('%.4f(%.2f%+.2fi) ',abs(lam(j)),real(lam(j)),imag(lam(j)))];
        else
            mag_str = [mag_str sprintf('%.4f ',abs(lam(j)))];
        end
    end
    fprintf('%-8.3f %-8.4f %-8.3f %-40s %-6s\n', ...
        s_vals(i), speeds(i), T_vals(i), strtrim(mag_str), ...
        string(max_nontrivial(i) < 1));
end
fprintf('%s\n', repmat('=',1,80));
fprintf('* Stable = max|lambda_nontrivial| < 1 (marginal lambda~1 excluded)\n');

% =====================================================================
% CSV Export
% =====================================================================
csv_name = sprintf('floquet_gam%.3f_k%.3f.csv', gam, k_hip);
fid = fopen(csv_name, 'w');
fprintf(fid, 's,v,T_stride,lam1_re,lam1_im,lam2_re,lam2_im,lam3_re,lam3_im,abs_lam1,abs_lam2,abs_lam3,stable\n');
for i = 1:n_found
    lam = lam_all(i,:);
    fprintf(fid, '%.6f,%.6f,%.6f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%d\n', ...
        s_vals(i), speeds(i), T_vals(i), ...
        real(lam(1)), imag(lam(1)), real(lam(2)), imag(lam(2)), real(lam(3)), imag(lam(3)), ...
        abs(lam(1)), abs(lam(2)), abs(lam(3)), max_nontrivial(i) < 1);
end
fclose(fid);
fprintf('Exported %d orbits to %s\n', n_found, csv_name);
end


% =========================================================================
% SUBFUNCTIONS
% =========================================================================

function [z_fp, T_stride, converged] = find_fixed_point(z0, gam, a, tau, k, P, per, opts)
    converged = false; T_stride = []; z_fp = z0;
    delta_nr = 1e-7; max_iter = 12; tol = 1e-10;

    % Quick pre-check: does the initial guess even produce a stride?
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

        % Check conditioning before solving
        if rcond(Jg) < 1e-14, return; end

        z_fp = z_fp + 0.8*(-Jg\res);
    end
    [Sz,T,ok] = stride_map_reduced(z_fp,gam,a,tau,k,P,per,opts);
    if ok && norm(Sz-z_fp)<1e-8, converged=true; T_stride=T; end
end


function [z_new, T_stride, ok] = stride_map_reduced(z, gam, a, tau, k, P, per, opts)
    z_new = []; T_stride = []; ok = false;

    % Reject obviously bad ICs
    if abs(z(1)) > pi/3, return; end

    y0 = [z(1); z(2); 2*z(1); z(3)];
    [tout, yout, te, ~, ie] = ode45(@(t,y)f(t,y,gam,a,tau,k), [0 per], y0, opts);

    % Check that termination was by collision (ie==1), not divergence (ie==2)
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
%COLLISION_WITH_GUARD  Heelstrike detection + divergence guard
%   Event 1: phi - 2*theta = 0 (heelstrike, direction +)
%   Event 2: |theta| > pi/2   (divergence, terminate immediately)
    val = [y(3)-2*y(1);            % collision
           pi/2 - abs(y(1))];      % divergence guard
    ist = [1; 1];                   % stop on either event
    dir = [1; 0];                   % collision: + direction only; guard: any
end


function progress_tracker(action, N)
    persistent count total
    if strcmp(action, 'init')
        count = 0;
        total = N;
        return;
    end
    count = count + 1;
    pct = count / total * 100;
    n_bar = round(pct / 2);
    bar_str = [repmat('#', 1, n_bar), repmat('.', 1, 50 - n_bar)];
    fprintf('\r  [%s] %3.0f%%  (%d/%d)', bar_str, pct, count, total);
    if count == total, fprintf('\n'); end
end