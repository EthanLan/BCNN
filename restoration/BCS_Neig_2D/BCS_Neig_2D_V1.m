%--------------------------------------------------------------------------
% References:
% Yu Lei, "Bayesion compressive sensing for clustered sparse signals" 
%
% Created: Nov. 25, 2008
% Xinjie Lan, ECE, University of Delaware;
%
% Version:
% BCS_Neig_2D_V1: show PI and beta's two parmeters
%--------------------------------------------------------------------------
function [img_re,PSNRr,SSIMr, PI]=BCS_Neig_2D_V1(img_cl,Phi, v, Ms,S0,WaveletName)

% ---------------------
% check input arguments
% ---------------------

MCMCpara.nBurnin=100;
MCMCpara.nCollect=50;
MCMCpara.thinFactor=1;


% Number of coefficients
M=size(Phi,2);          % all coefficients      
Mscaling=M-sum(Ms,2);   % scaling coefficients
Mwavelet=M-Mscaling;    % wavelet coefficients
% Number of decomposition level
L=length(Ms);
% hyperparameters
hyperpara.a=1e-6;
hyperpara.b=1e-6;
hyperpara.c=1e-6;
hyperpara.d=1e-6;
hyperpara.es=(1-eps)*Mscaling;
hyperpara.fs=eps*Mscaling;
hyperpara.er=0.9*Ms(1);
hyperpara.fr=0.1*Ms(1);
hyperpara.e0=[0, ones(1, L-1)/M].*Ms;
hyperpara.f0=[0, 1-ones(1, L-1)/M].*Ms;
hyperpara.e1=[0, ones(1, L-1)*0.5].*Ms;
hyperpara.f1=[0, ones(1, L-1)*0.5].*Ms;
hyperpara.e2=[0, 1-ones(1, L-1)/M].*Ms;
hyperpara.f2=[0, ones(1, L-1)/M].*Ms;

MCMCpara.nBurnin=100;
MCMCpara.nCollect=50;
MCMCpara.thinFactor=1;

a=hyperpara.a;
b=hyperpara.b;
c=hyperpara.c;
d=hyperpara.d;
es=hyperpara.es;
fs=hyperpara.fs;
er=hyperpara.er;
fr=hyperpara.fr;
e0=hyperpara.e0;
f0=hyperpara.f0;
e1=hyperpara.e1;
f1=hyperpara.f1;
e2=hyperpara.e2;
f2=hyperpara.f2;

std_v=std(v);
v=v/std_v;

% -------------------
% Data specifications
% -------------------

% Number of CS measurements
N=length(v);       
% Number of coefficients
M=size(Phi,2);          % all coefficients      
Mscaling=M-sum(Ms,2);   % scaling coefficients
% Mwavelet=M-Mscaling;    % wavelet coefficients
% Number of decomposizion level
L=length(Ms);
% start and end index for each level
idxLevelEnd=Mscaling+cumsum(Ms,2);
idxLevelStart=[Mscaling+1,idxLevelEnd(1:L-1)+1];

% --------------
% Initialization
% --------------

% theta: M x 1, estimated scaling and wavelet coefficients
theta=zeros(M,1);

% alpha: 1 x L, coefficient precision, coefficients at each wavelet level share one common alpha
alpha=ones(1,L);
% alpha_scaling, 1x1, precision shared by all scaling coefficients
alpha_scaling=1;

% PI: M x 1, mixing weight for each coefficient
PI=es/(es+fs)*ones(M,1);
PI(idxLevelStart(1):idxLevelEnd(1))=er/(er+fr)*ones(Ms(1),1);
for s=2:L
    PI(idxLevelStart(s):idxLevelEnd(s))=e0(s)/(e0(s)+f0(s))*ones(Ms(s),1);
end

% alpha0: scalar, noise precision
alpha0=1/(std(v)^2/1e2);

%---------------
% precomputation
%---------------

% Phi_{i}^{T}Phi_{i} for all i, M x 1 vector
PhiTPhi=sum(Phi.*Phi,1)';

% indicator of level for each coefficient (zero means scaling)
indiLevel=zeros(M,1);
for s=1:L
    indiLevel(idxLevelStart(s):idxLevelEnd(s))=s;
end

% --------------
% Gibbs Sampling
% --------------

for iter=1:(MCMCpara.nBurnin+MCMCpara.nCollect)

    % (1) theta -- sequentially drawn
    % \tilde{alpha}_{i}, M x 1
    alpha_tilde=[alpha_scaling*ones(Mscaling,1);alpha(indiLevel(Mscaling+1:M))']+alpha0*PhiTPhi;
    % i=1
    A=Phi*theta;
    v_tilde=v-A+Phi(:,1)*theta(1);
    mu_tilde=(alpha0*Phi(:,1)'*v_tilde)/alpha_tilde(1);
    ratio=sqrt(alpha_scaling/alpha_tilde(1))*exp(0.5*alpha_tilde(1)*mu_tilde*mu_tilde)*PI(1)/(1-PI(1));
    theta_old=theta(1);
    if isinf(ratio)
        pi_tilde(1)=1;
        theta(1)=normrnd(mu_tilde, sqrt(1/alpha_tilde(1)));
    else
        pi_tilde(1)=ratio/(ratio+1);
        if rand<pi_tilde(1)
            theta(1)=normrnd(mu_tilde, sqrt(1/alpha_tilde(1)));
        else
            theta(1)=0;
        end
    end
    for i=2:M
        % update A
        A=A+Phi(:,i-1)*(theta(i-1)-theta_old);
        % \tilde{mu}_{i}, M x 1
        v_tilde=v-A+Phi(:,i)*theta(i);
        mu_tilde=(alpha0*Phi(:,i)'*v_tilde)/alpha_tilde(i);
        % \tilde{pi}_{i}, M x 1
        if i<=Mscaling
            ratio=sqrt(alpha_scaling/alpha_tilde(i))*exp(0.5*alpha_tilde(i)*mu_tilde*mu_tilde)*PI(i)/(1-PI(i));
        else
            ratio=sqrt(alpha(indiLevel(i))/alpha_tilde(i))*exp(0.5*alpha_tilde(i)*mu_tilde*mu_tilde)*PI(i)/(1-PI(i));
        end
        % sample theta
        theta_old=theta(i);
        if isinf(ratio)
            pi_tilde(i)=1;
            theta(i)=normrnd(mu_tilde, sqrt(1/alpha_tilde(i)));
        else
            pi_tilde(i)=ratio/(ratio+1);
            if rand<pi_tilde(i)
                theta(i)=normrnd(mu_tilde, sqrt(1/alpha_tilde(i)));
            else
                theta(i)=0;
            end
        end
    end
    z=find(theta~=0);
    
    % (2) alpha
    % for scaling level
    idx=find(z>=1 & z<=Mscaling);
    if ~isempty(idx)
        alpha_scaling=gamrnd(c+0.5*length(idx), 1/(d+0.5*sum(theta(z(idx)).^2,1)));
    end
    % for wavelet level
    for s=1:L
        idx=find(z>=idxLevelStart(s) & z<=idxLevelEnd(s));
        if ~isempty(idx)
            alpha(s)=gamrnd(c+0.5*length(idx), 1/(d+0.5*sum(theta(z(idx)).^2,1)));
        end
    end
    
    neig_size = 4;
    % (3) PI
    indi=zeros(M,1); indi(z)=1;     % nonzero coefficient indicator
    % for scaling level
    idx=1:Mscaling;
    Nnon0=sum(indi(idx),1);
    beta_sample=betarnd(es+Nnon0,fs+length(idx)-Nnon0);
    PI(idx)=beta_sample;
    % for root level
    idx=idxLevelStart(1):idxLevelEnd(1);
    Nnon0=sum(indi(idx),1);
    beta_sample=betarnd(er+Nnon0,fr+length(idx)-Nnon0);
    PI(idx)=beta_sample;
    
%     idx=idxLevelStart(1):idxLevelEnd(1);
%     Indi_Neig = NeighborRelation2D(indi(idx),neig_size);
% 	% for neighbor=0
% 	idx0N=find(Indi_Neig==0);
% 	Nnon0=sum(indi(idx(idx0N)),1);
% 	beta_sample=betarnd(er+Nnon0, fr+length(idx0N)-Nnon0);
% 	PI(idx(idx0N))=beta_sample;
% 	% for neighbor=2
% 	idx1N=find(Indi_Neig==2);
% 	Nnon1=sum(indi(idx(idx1N)),1);
% 	beta_sample=betarnd(er+Nnon1, fr+length(idx1N)-Nnon1);
% 	PI(idx(idx1N))=beta_sample;
% 	% for neighbor is not 0 or 1
% 	idx_unclear = setdiff(idx-idxLevelStart(1)+1,union(idx0N,idx1N));
% 	Nnon_unclear=sum(indi(idx(idx_unclear)),1);
% 	beta_sample=betarnd(er+Nnon_unclear, fr+length(idx_unclear)-Nnon_unclear);
% 	PI(idx(idx_unclear))=beta_sample;

    % for other levels
    for s=2:L
        idx=idxLevelStart(s):idxLevelEnd(s);
        Indi_Neig = NeighborRelation2D(indi(idx),neig_size);
        % for neighbor=0
        idx0N=find(Indi_Neig==0);
        Nnon0=sum(indi(idx(idx0N)),1);
        beta_sample=betarnd(e0(s)+Nnon0, f0(s)+length(idx0N)-Nnon0);
        PI(idx(idx0N))=beta_sample;
        % for neighbor=2
        idx1N=find(Indi_Neig==4);
        Nnon1=sum(indi(idx(idx1N)),1);
        beta_sample=betarnd(e2(s)+Nnon1, f2(s)+length(idx1N)-Nnon1);
        PI(idx(idx1N))=beta_sample;
        % for neighbor is not 0 or 1
        idx_unclear = setdiff(idx-idxLevelStart(s)+1,union(idx0N,idx1N));
        Nnon_unclear=sum(indi(idx(idx_unclear)),1);
        beta_sample=betarnd(e1(s)+Nnon_unclear, f1(s)+length(idx_unclear)-Nnon_unclear);
        PI(idx(idx_unclear))=beta_sample;
    end
    PI(find(PI==1))=1-eps;
    PI(find(PI==0))=eps;
    
    % (4) alpha0
    res=v-Phi(:,z)*theta(z);
    alpha0=gamrnd(a+N/2, 1/(b+res'*res/2));
    
    % Collect samples
    if iter>MCMCpara.nBurnin & (mod(iter-MCMCpara.nBurnin,MCMCpara.thinFactor)==1 | MCMCpara.thinFactor==1)
        i = ceil((iter-MCMCpara.nBurnin)/MCMCpara.thinFactor);
        samples(i).theta = theta*std_v;
    end

%     if(mod(iter,15) == 0)
%         status = ['BCS_Neig_2D_V1 Iteration ',num2str(iter)];
%         disp(status);
%     end

end

m_theta=mean(cat(2,samples(:).theta),2);
img_re = waverec2(m_theta', S0, WaveletName);

border = size(img_re) - size(img_cl);
hborder = border./2;
img_re = img_re(hborder(1)+1:end-hborder(1), hborder(2)+1:end-hborder(2));

ERR = norm(img_re(:) - img_cl(:))./norm(img_cl(:));
RSNR = 20.*log10(1./ERR);
PSNRr = psnr(img_re(:),img_cl(:));
SSIMr = ssim_index(reshape(img_re,size(img_cl)),img_cl);
fprintf('BCS_Neig_2D_V1 is done! RSNR %.3f PSNR %.3f SSIM %.3f \n',RSNR,PSNRr,SSIMr);
