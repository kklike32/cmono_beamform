maxiters = 100;
alpha = 0.01;
beta = 0.5;
nttol= 1e-8;
p=30;
A = 
nu = zeros(p,1);
for i = 1:maxiters
    val = b' * nu + sum(exp((-A' * nu -1)));
    grad = b -A*exp(-A'*nu-1);
    hess = A*diag(exp(-a'*nu-1))*A;
    v = -heff\grad;
    fprime = grad'*v;
    if (abs(fprime)<nttol), break; 
    end
    t = 1;
   while (b'*(nu+t*v) + sum(exp(-A'*(nu+t*v)-1)) > val + t*ALPHA*fprime), t = BETA*t; 
  end;
 nu = nu + t*v;
 end;
