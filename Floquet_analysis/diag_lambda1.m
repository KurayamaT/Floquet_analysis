function diagnose_lambda1()
%DIAGNOSE_LAMBDA1  Isolate the cause of lambda_1 = -1
%
%   Tests 4 combinations at a single speed (v ~ 0.08):
%     A) Original stride map + loose NR  (= floquet_KUO.m)
%     B) Original stride map + tight NR
%     C) Two-phase step map  + loose NR
%     D) Two-phase step map  + tight NR  (= verify_floquet_KUO.m)

gam = 0;  k_hip = -0.08;  per = 5;
s = 0.291;

alpha = asin(0.5*s);
omega = -1.04*alpha;
P = -omega*tan(alpha);
z0 = [alpha; omega; (1-cos(2*alpha))*omega];

opts_loose = odeset('RelTol',1e-2, 'AbsTol',1e-6, 'Refine',4, ...
                    'Events',@collision_with_guard);
opts_tight = odeset('RelTol',1e-12,'AbsTol',1e-14,'Refine',4, ...
                    'Events',@collision_with_guard);

delta = 1e-7;

fprintf('============================================================\n');
fprintf(' Diagnostic: What causes lambda_1 = -1?\n');
fprintf(' s = %.3f (v ~ 0.08)\n', s);
fprintf('============================================================\n\n');

% =====================================================================
% TEST A: Original map + loose NR (= floquet_KUO.m)
% =====================================================================
fprintf('A) Orig map + loose NR  (= floquet_KUO.m)\n');
nr_map_A = @(z) stride_map_orig(z, gam, k_hip, P, per, opts_loose);
jac_map_A = @(z) stride_map_orig(z, gam, k_hip, P, per, opts_tight);
[z_A, ~] = find_fp_generic(z0, nr_map_A);
run_test(z_A, jac_map_A, delta, 'A');

% Check: what does orig map return at z_A with TIGHT tolerance?
[Sz_tight, T_tight, ok_tight] = stride_map_orig(z_A, gam, k_hip, P, per, opts_tight);
[Sz_loose, T_loose, ok_loose] = stride_map_orig(z_A, gam, k_hip, P, per, opts_loose);
fprintf('  Orig map at z_A with LOOSE tol: T=%.6f, ok=%d\n', T_loose, ok_loose);
fprintf('  Orig map at z_A with TIGHT tol: T=%.6f, ok=%d\n', T_tight, ok_tight);
if ok_tight && T_tight < 0.01
    fprintf('  *** T ~ 0 with tight tol! False event at t=0 confirmed. ***\n');
end
fprintf('\n');

% =====================================================================
% TEST B: Original map + tight NR (re-converge)
% =====================================================================
fprintf('B) Orig map + tight NR (re-converge from A)\n');
[z_B, ~] = find_fp_generic(z_A, jac_map_A);
[~, T_B, ok_B] = stride_map_orig(z_B, gam, k_hip, P, per, opts_tight);
fprintf('  Orig map at z_B with TIGHT tol: T=%.6f, ok=%d\n', T_B, ok_B);
if ok_B && T_B > 0.1
    run_test(z_B, jac_map_A, delta, 'B');
else
    fprintf('  Cannot re-converge with orig map + tight tol (T~0 problem)\n\n');
end

% =====================================================================
% TEST C: Two-phase map + loose NR
% =====================================================================
fprintf('C) TwoPhase map + loose NR\n');
nr_map_C = @(z) step_map_twophase(z, gam, k_hip, P, per, opts_loose);
jac_map_C = @(z) step_map_twophase(z, gam, k_hip, P, per, opts_tight);
[z_C, ~] = find_fp_generic(z0, nr_map_C);
run_test(z_C, jac_map_C, delta, 'C');

% =====================================================================
% TEST D: Two-phase map + tight NR (= verify_floquet_KUO.m)
% =====================================================================
fprintf('D) TwoPhase map + tight NR (re-converge from C)\n');
[z_D, ~] = find_fp_generic(z_C, jac_map_C);
run_test(z_D, jac_map_C, delta, 'D');

% =====================================================================
% Stride map output comparison at perturbed states
% =====================================================================
fprintf('============================================================\n');
fprintf(' Stride map comparison: Orig vs TwoPhase at z_D +/- delta\n');
fprintf('============================================================\n');
fprintf('  z_D = [%.12f, %.12f, %.12f]\n\n', z_D);

fprintf('%-4s %-6s %-14s %-14s %-14s %-14s %-10s\n', ...
    'j', 'sign', 'Orig_T', 'TwoPhase_T', 'Orig_z1', 'TwoPhase_z1', 'T_match?');
fprintf('%s\n', repmat('-', 1, 80));

for j = 1:3
    for sgn = [+1, -1]
        zp = z_D;
        zp(j) = zp(j) + sgn*delta;

        [So, To, ok_o] = stride_map_orig(zp, gam, k_hip, P, per, opts_tight);
        [St, Tt, ok_t] = step_map_twophase(zp, gam, k_hip, P, per, opts_tight);

        if ok_o && ok_t
            tmatch = 'OK';
            if abs(To - Tt) > 0.01
                tmatch = sprintf('*** DIFF %.2e ***', abs(To-Tt));
            end
            fprintf('%-4d %-6s %-14.8f %-14.8f %-14.10f %-14.10f %-s\n', ...
                j, sprintf('%+d',sgn), To, Tt, So(1), St(1), tmatch);
        else
            fprintf('%-4d %-6s ', j, sprintf('%+d',sgn));
            if ~ok_o, fprintf('ORIG:FAIL     '); else, fprintf('ORIG:T=%.4f  ',To); end
            if ~ok_t, fprintf('TP:FAIL'); else, fprintf('TP:T=%.4f',Tt); end
            fprintf('\n');
        end
    end
end

fprintf('\nIf Orig_T ~ 0 for some rows → t=0 false event is the cause.\n');

end  % main


%% =====================================================================
function run_test(z_fp, mapfun, delta, label)
    [Sz, T, ok] = mapfun(z_fp);
    if ~ok || isempty(Sz)
        fprintf('  z_fp = [%.10f, %.10f, %.10f]\n', z_fp);
        fprintf('  Map returned FAIL or empty at fixed point\n\n');
        return;
    end
    res = norm(Sz(:) - z_fp(:));

    J = zeros(3);
    ok_all = true;
    for j = 1:3
        zp = z_fp; zp(j) = zp(j) + delta;
        zm = z_fp; zm(j) = zm(j) - delta;
        [Sp, ~, ok1] = mapfun(zp);
        [Sm, ~, ok2] = mapfun(zm);
        if ~ok1 || ~ok2 || isempty(Sp) || isempty(Sm)
            ok_all = false; break;
        end
        J(:,j) = (Sp(:) - Sm(:)) / (2*delta);
    end

    fprintf('  z_fp = [%.10f, %.10f, %.10f]\n', z_fp);
    fprintf('  T = %.6f, |S(z_fp)-z_fp| = %.2e\n', T, res);

    if ~ok_all
        fprintf('  Jacobian computation FAILED\n\n');
        return;
    end

    lam = eig(J);
    [~, si] = sort(abs(lam), 'descend');
    lam = lam(si);

    fprintf('  trace(J) = %.6f\n', real(trace(J)));
    for i = 1:3
        if abs(imag(lam(i))) < 1e-8
            fprintf('    lam_%d = %+.8f          (|lam| = %.6f)\n', ...
                i, real(lam(i)), abs(lam(i)));
        else
            fprintf('    lam_%d = %+.6f %+.6fi  (|lam| = %.6f)\n', ...
                i, real(lam(i)), imag(lam(i)), abs(lam(i)));
        end
    end
    fprintf('\n');
end


%% ORIGINAL STRIDE MAP (= floquet_KUO.m stride_map_reduced)
function [z_new, T, ok] = stride_map_orig(z, gam, k, P, per, opts)
    z_new = []; T = []; ok = false;
    if abs(z(1)) > pi/3, return; end
    y0 = [z(1); z(2); 2*z(1); z(3)];
    [tout, yout, te, ~, ie] = ode45(@(t,y)eom(t,y,gam,k), [0 per], y0, opts);
    if isempty(te), return; end
    if isempty(ie) || ie(end) ~= 1, return; end
    T = tout(end);
    ye = yout(end,:);
    c2 = cos(2*ye(1)); s2p = sin(2*ye(1))*P;
    z_new = [-ye(1); c2*ye(2)+s2p; c2*(1-c2)*ye(2)+(1-c2)*s2p];
    ok = true;
end


%% TWO-PHASE STEP MAP (corrected)
function [z_new, T, ok] = step_map_twophase(z, gam, k, P, per, opts)
    z_new = []; T = []; ok = false;
    if abs(z(1)) > pi/3, return; end
    y0 = [z(1); z(2); 2*z(1); z(3)];
    dt = 0.005;
    opts1 = odeset('RelTol',1e-12,'AbsTol',1e-14);
    [~, yout1] = ode45(@(t,y)eom(t,y,gam,k), [0 dt], y0, opts1);
    y_dep = yout1(end,:).';
    [~, ~, te, ye, ie] = ode45(@(t,y)eom(t,y,gam,k), [dt per], y_dep, opts);
    if isempty(ie), return; end
    idx_c = find(ie == 1);
    if isempty(idx_c), return; end
    ic = idx_c(1);
    T = te(ic); yc = ye(ic,:);
    c2 = cos(2*yc(1)); s2P = sin(2*yc(1))*P;
    z_new = [-yc(1); c2*yc(2)+s2P; c2*(1-c2)*yc(2)+(1-c2)*s2P];
    ok = true;
end


%% EQUATIONS OF MOTION
function ydot = eom(~, y, gam, k)
    F = k*y(3);
    ydot = [y(2); sin(y(1)-gam); y(4);
            sin(y(1)-gam)+sin(y(3))*(y(2)^2-cos(y(1)-gam))+F];
end


%% COLLISION EVENT
function [val,ist,dir] = collision_with_guard(~,y)
    val = [y(3)-2*y(1); pi/2-abs(y(1))];
    ist = [1;1]; dir = [1;0];
end


%% GENERIC NEWTON-RAPHSON
function [z_fp, T] = find_fp_generic(z0, mapfun)
    z_fp = z0; T = []; delta = 1e-7;
    for iter = 1:20
        [Sz, T, ok] = mapfun(z_fp);
        if ~ok || isempty(Sz), return; end
        res = Sz(:) - z_fp(:);
        if norm(res) < 1e-12, return; end
        Jg = zeros(3);
        for j = 1:3
            zp = z_fp; zp(j) = zp(j)+delta;
            zm = z_fp; zm(j) = zm(j)-delta;
            [Sp,~,o1] = mapfun(zp);
            [Sm,~,o2] = mapfun(zm);
            if ~o1||~o2||isempty(Sp)||isempty(Sm), return; end
            Jg(:,j) = (Sp(:)-Sm(:))/(2*delta);
        end
        Jg = Jg - eye(3);
        if rcond(Jg) < 1e-14, return; end
        z_fp = z_fp(:) + 0.8*(-Jg\res);
    end
end