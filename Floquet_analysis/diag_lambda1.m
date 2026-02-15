% diag_lambda1.m — standalone diagnostic (fixed)
gam = 0; a = 0; tau = 3.84; k_hip = -0.08; per = 5;
opts_nr   = odeset('RelTol',1e-6, 'AbsTol',1e-8, 'Refine',4,'Events',@collision_with_guard);
opts_fine = odeset('RelTol',1e-10,'AbsTol',1e-12,'Refine',4,'Events',@collision_with_guard);

diag_s = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80];
deltas = [1e-5, 1e-6, 1e-7, 1e-8];

fprintf('%-8s','s');
for d = deltas, fprintf('  delta=%.0e       ', d); end
fprintf('\n%s\n', repmat('-',1,80));

for si = 1:length(diag_s)
    s = diag_s(si);
    alpha = asin(0.5*s);
    omega = -1.04*alpha;
    P = -omega*tan(alpha);
    z0 = [alpha; omega; (1-cos(2*alpha))*omega];
    [z_fp, ~, conv] = find_fp(z0, gam, a, tau, k_hip, P, per, opts_nr);
    if ~conv, fprintf('%-8.3f  NO CONVERGENCE\n', s); continue; end

    fprintf('%-8.3f', s);
    for di = 1:length(deltas)
        dd = deltas(di);
        J = zeros(3);
        ok = true;
        for j = 1:3
            zp = z_fp; zp(j)=zp(j)+dd;
            zm = z_fp; zm(j)=zm(j)-dd;
            [Sp,~,o1] = smap(zp, gam, a, tau, k_hip, P, per, opts_fine);
            [Sm,~,o2] = smap(zm, gam, a, tau, k_hip, P, per, opts_fine);
            if ~o1||~o2, ok=false; break; end
            J(:,j) = (Sp-Sm)/(2*dd);
        end
        if ~ok, fprintf('  FAIL              '); continue; end
        lam = eig(J);
        [~,ix] = min(abs(lam+1));
        fprintf('  %.4e(%+.8f)', abs(lam(ix)+1), real(lam(ix)));
    end
    fprintf('\n');
end

function [z_fp,T,conv] = find_fp(z0,gam,a,tau,k,P,per,opts)
    conv=false; T=[]; z_fp=z0; dd=1e-7;
    for it=1:15
        [Sz,T,ok]=smap(z_fp,gam,a,tau,k,P,per,opts);
        if ~ok, return; end
        res=Sz-z_fp;
        if norm(res)<1e-10, conv=true; return; end
        Jg=zeros(3);
        for j=1:3
            zp=z_fp;zp(j)=zp(j)+dd; zm=z_fp;zm(j)=zm(j)-dd;
            [Sp,~,o1]=smap(zp,gam,a,tau,k,P,per,opts);
            [Sm,~,o2]=smap(zm,gam,a,tau,k,P,per,opts);
            if ~o1||~o2, return; end
            Jg(:,j)=(Sp-Sm)/(2*dd);
        end
        Jg=Jg-eye(3);
        if rcond(Jg)<1e-14, return; end
        z_fp=z_fp+0.8*(-Jg\res);
    end
    [Sz,T,ok]=smap(z_fp,gam,a,tau,k,P,per,opts);
    if ok&&norm(Sz-z_fp)<1e-8, conv=true; end
end

function [zn,T,ok] = smap(z,gam,a,tau,k,P,per,opts)
    zn=[]; T=[]; ok=false;
    if abs(z(1))>pi/3, return; end
    y0=[z(1);z(2);2*z(1);z(3)];
    [t,y,te,~,ie]=ode45(@(t,y)dyn(t,y,gam,a,tau,k),[0 per],y0,opts);
    if isempty(te)||isempty(ie)||ie(end)~=1, return; end
    T=t(end); ye=y(end,:);
    c2=cos(2*ye(1)); s2p=sin(2*ye(1))*P;
    zn=[-ye(1); c2*ye(2)+s2p; c2*(1-c2)*ye(2)+(1-c2)*s2p];
    ok=true;
end

function yd=dyn(t,y,gam,a,tau,k)
    F=a*sin(2*pi/tau*t)+k*y(3);
    yd=[y(2);sin(y(1)-gam);y(4);sin(y(1)-gam)+sin(y(3))*(y(2)^2-cos(y(1)-gam))+F];
end

function [v,ist,dir]=collision_with_guard(t,y)
    v=[y(3)-2*y(1); pi/2-abs(y(1))];
    ist=[1;1]; dir=[1;0];
end