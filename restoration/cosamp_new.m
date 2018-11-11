% cosamp.m
% Performs CS reconstruction using the CoSaMP algorithm
% (D. Needell and J. Tropp, "CoSaMP: iterative signal recovery from 
%  incomplete and inaccurate measurements" , ACHA, 2009)
% 
% INPUTS
% yy : measurement (M x 1)
% Phi: measurement matrix (M x N)
% K  : signal sparsity
% Its: number of iterations
% 
% OUTPUTS
% xhat   : Signal estimate (N x 1) 
% xcosamp: Matrix with N rows and at most Its columns; 
%          columns represent intermediate signal estimates   
% 
%
% CITE: Richard Baraniuk, Volkan Cevher, Marco Duarte, Chinmay Hegde
%       "Model-based compressive sensing", submitted to IEEE IT, 2008.
% Created: Aug 2009.
% email: chinmay@rice.edu

function [xhat,xcosamp] = cosamp_new(yy, Phi, K, Its);

yy = yy(:); % 
[M,N] = size(Phi);

xcosamp = zeros(N,Its);
kk=1; 
maxiter= 1000;
verbose= 0;
tol= 1e-3;
s_cosamp = zeros(N,1);

while le(kk,Its),
    
    %-----Backprojection---%
    rcosamp = yy - Phi*(s_cosamp);
    proxy_cosamp = Phi'*(rcosamp);
    [trash,ww]= sort(abs(proxy_cosamp),'descend');
    tt_cosamp= union(find(ne(s_cosamp,0)),ww(1:(2*K)));
    
    %------Estimate------%
    [w_cosamp, res, iter] = cgsolve(Phi(:,tt_cosamp)'*Phi(:,tt_cosamp), Phi(:,tt_cosamp)'*yy,...
                                        tol,maxiter, verbose);
  
    bb2= zeros(N,1);
    bb2(tt_cosamp)= w_cosamp;
    
    %---Prune----%
    kk = kk+1;   
    [trash,ww2]= sort(abs(bb2),'descend'); s_cosamp=0*bb2;
    s_cosamp(ww2(1:K))= bb2(ww2(1:K));


    xcosamp(:,kk) = s_cosamp; % current signal estimate
    if (norm(xcosamp(:,kk)-xcosamp(:,kk-1)) < 1e-3*norm(xcosamp(:,kk)))
       break;
    end
    
    if(mod(kk,10) == 0)
        status = ['CosaMP Iteration ',num2str(kk)];
        disp(status);
    end
    
end
xcosamp(:,kk+1:end)=[];
xhat = xcosamp(:,end);
