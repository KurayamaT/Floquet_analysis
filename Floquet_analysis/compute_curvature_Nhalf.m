function compute_curvature_Nhalf(csv_file)
%COMPUTE_CURVATURE_NHALF  Find the transition speed of N_1/2(v)
%
%   compute_curvature_Nhalf('floquet_gam0.000_k-0.080.csv')
%
%   Uses three methods to identify the transition speed:
%     1. Parametric curvature kappa(v)
%     2. Elbow method (max perpendicular distance from chord)
%     3. Log-domain elbow (same method on ln N_1/2)
%
%   No toolboxes required (base MATLAB only).

if nargin < 1
    csv_file = 'floquet_gam0.000_k-0.080.csv';
end

%% 1. Load data
T = readtable(csv_file);
v = T.v;
s = T.s;
abs_lam2 = T.abs_lam2;

s_max = 0.80;
mask = (abs_lam2 > 0) & (abs_lam2 < 1) & (T.stable == 1) & (s <= s_max);
v = v(mask);
abs_lam2 = abs_lam2(mask);
[v, si] = sort(v);
abs_lam2 = abs_lam2(si);

fprintf('Using %d data points in range v = [%.3f, %.3f]\n', length(v), min(v), max(v));

%% 2. Compute perturbation half-life
N_half = -log(2) ./ log(abs_lam2);

%% 3. Fine grid
v_fine = linspace(min(v), max(v), 2000)';
N_fine = spline(v, N_half, v_fine);

%% 4. Numerical derivatives
dv = v_fine(2) - v_fine(1);
dN  = gradient(N_fine, dv);
d2N = gradient(dN, dv);

%% 5. Method 1: Parametric curvature
kappa = abs(d2N) ./ (1 + dN.^2).^(3/2);
margin = round(0.10 * length(v_fine));
kappa_int = kappa;
kappa_int(1:margin) = 0;
kappa_int(end-margin+1:end) = 0;
[kappa_max, idx_k] = max(kappa_int);
v_kappa = v_fine(idx_k);

%% 6. Method 2: Elbow method (perpendicular distance from chord)
%  Normalize v and N_half to [0,1] for unbiased distance
v_norm = (v_fine - v_fine(1)) / (v_fine(end) - v_fine(1));
N_norm = (N_fine - N_fine(end)) / (N_fine(1) - N_fine(end));

% Chord from (0,1) to (1,0) in normalized space: line x + y = 1
% Perpendicular distance = |x + y - 1| / sqrt(2)
dist = abs(v_norm + N_norm - 1) / sqrt(2);
[~, idx_e] = max(dist);
v_elbow = v_fine(idx_e);
N_elbow = N_fine(idx_e);

%% 7. Method 3: Log-domain elbow
lnN = log(N_fine);
lnN_norm = (lnN - lnN(end)) / (lnN(1) - lnN(end));
dist_log = abs(v_norm + lnN_norm - 1) / sqrt(2);
[~, idx_l] = max(dist_log);
v_log = v_fine(idx_l);
N_log = N_fine(idx_l);

%% 8. Dimensional speeds
g = 9.81; L = 1.0;

fprintf('\n============================================================\n');
fprintf('  Transition speed estimates\n');
fprintf('============================================================\n');
fprintf('  Method              v*       V* [m/s]   N_1/2 [strides]\n');
fprintf('  ----------------------------------------------------------\n');
fprintf('  Parametric kappa    %.4f   %.2f        %.1f\n', v_kappa, v_kappa*sqrt(g*L), N_fine(idx_k));
fprintf('  Elbow method        %.4f   %.2f        %.1f\n', v_elbow, v_elbow*sqrt(g*L), N_elbow);
fprintf('  Log elbow           %.4f   %.2f        %.1f\n', v_log, v_log*sqrt(g*L), N_log);
fprintf('============================================================\n\n');

%% 9. Plot
figure('Color','w','Position',[100 100 1000 800],'Name','Transition Speed Analysis')

% Panel 1: N_1/2(v) with all transition points
subplot(2,2,[1 2]), hold on
plot(v, N_half, 'b.', 'MarkerSize', 4)
plot(v_fine, N_fine, 'b-', 'LineWidth', 1.5)
% Chord line (unnormalized)
plot([v_fine(1) v_fine(end)], [N_fine(1) N_fine(end)], 'k--', 'LineWidth', 0.8)
plot(v_elbow, N_elbow, 'rs', 'MarkerSize', 12, 'LineWidth', 2, 'MarkerFaceColor','r')
plot(v_log, N_log, 'gd', 'MarkerSize', 12, 'LineWidth', 2, 'MarkerFaceColor','g')
plot(v_kappa, N_fine(idx_k), 'mo', 'MarkerSize', 12, 'LineWidth', 2, 'MarkerFaceColor','m')
xline(v_elbow, 'r--', 'LineWidth', 1)
legend('Data', 'Spline', 'Chord', ...
    sprintf('Elbow (v=%.3f)', v_elbow), ...
    sprintf('Log elbow (v=%.3f)', v_log), ...
    sprintf('\\kappa_{max} (v=%.3f)', v_kappa), ...
    'Location','northeast')
xlabel('Dimensionless speed v', 'FontSize', 12)
ylabel('N_{1/2} [strides]', 'FontSize', 12)
title('Perturbation half-life with transition speed estimates', 'FontSize', 13)
grid on, box on
set(gca, 'FontSize', 11)

% Panel 2: Elbow distance
subplot(2,2,3), hold on
plot(v_fine, dist, 'r-', 'LineWidth', 1.5)
plot(v_elbow, dist(idx_e), 'rs', 'MarkerSize', 10, 'LineWidth', 2, 'MarkerFaceColor','r')
xline(v_elbow, 'r--', 'LineWidth', 1)
xlabel('v', 'FontSize', 12), ylabel('Distance from chord', 'FontSize', 12)
title('Elbow method (linear)', 'FontSize', 13)
grid on, box on, set(gca, 'FontSize', 11)

% Panel 3: Curvature
subplot(2,2,4), hold on
plot(v_fine, kappa, 'k-', 'LineWidth', 1.5)
plot(v_kappa, kappa_max, 'mo', 'MarkerSize', 10, 'LineWidth', 2, 'MarkerFaceColor','m')
xline(v_kappa, 'm--', 'LineWidth', 1)
xlabel('v', 'FontSize', 12), ylabel('\kappa(v)', 'FontSize', 12)
title('Parametric curvature', 'FontSize', 13)
grid on, box on, set(gca, 'FontSize', 11)

sgtitle('Transition speed analysis of N_{1/2}(v)', 'FontSize', 14, 'FontWeight','bold')

end