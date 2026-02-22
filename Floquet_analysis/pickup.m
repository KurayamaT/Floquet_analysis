% extract_table2.m
% CSV from floquet_KUO.m を読み込み、等間隔v値に最近傍のデータを抽出

data = readtable('floquet_gam0.000_k-0.080.csv');

v_targets = [0.04 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22];
L = 1.0;
g = 9.81;

fprintf('\nTable 2: Floquet multipliers at equally spaced speeds\n');
fprintf('%6s %8s %8s %10s %8s %12s\n', 'v', 'V(m/s)', 'lam1', 'lam2', 'lam3', 'N1/2');
fprintf('%s\n', repmat('-', 1, 60));

for i = 1:length(v_targets)
    vt = v_targets(i);
    [~, idx] = min(abs(data.v - vt));
    row = data(idx, :);
    
    lam2_abs = row.abs_lam2;
    N_half = -log(2) / log(lam2_abs);
    V_dim = row.v * sqrt(g * L);
    
    fprintf('%6.3f %8.2f %8.3f %10.4f %8.1f %12.1f\n', ...
        row.v, V_dim, -1, lam2_abs, 0, N_half);
end
