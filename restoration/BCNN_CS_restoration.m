%% RESTORATION - Image restoration by mrf model

%% [s s_mu] = sample_sigma_deblurring(reshape(x(:),mrf.imdims), y_valid, k) is change to matrix format
%% Kt_times_noise is changed to matrix format.

function [img_restored,psnr,ssim] = BCNN_CS_restoration(border, mrf, y, K_matrix, KtK_matrix,Kty,x_gt, sigma_gt, sigma, rb, doplot)
  
  %% Initialize unset variables, etc.
  if ~exist('sigma','var'), sigma = []; end
  if ~exist('rb','var'), rb = true; end
  if ~exist('doplot', 'var'), doplot = false; end
  %% Preprocessing signal
  x = rand(mrf.imdims)*255;
  do_noise_est = true;
            
  mrf.update_filter_matrices = true;
  mrf.conv_method = 'valid';
  
  npixels = prod(mrf.imdims);  
  
  % pre-compute some values for the sampling process
  Wt = vertcat(mrf.filter_matrices{1:mrf.nfilters}, speye(npixels));
  
  %% Signal Restoration
  
  % either do heuristic (faster) or conservative deblurring (as done for the results in the paper) 
  [img_restored psnr,ssim] = restoration_sampling(30, 10);
  
  
  %%===============================================     Restoration implementations
  function [img_restored,psnr,ssim] = restoration_sampling(max_iters, burnin)
    
    % start timer
    tic;

    % mrf object doesn't work with parfor loop (for whatever reason)
    mrf = mrf2struct(mrf);
    
    % initialize variables
    if do_noise_est
      s = zeros(1, 1);
      s_mu = zeros(1, 1);
      sigma_est = zeros(1, 1);
    end
    x_mu = zeros(npixels, 1);
        
    % temporary variables
    x_avg = zeros(npixels,1);
    if do_noise_est, sigma_avg = 0; end
    c_avg = 0;
      
    % loop for fixed number of iterations
    for iter = 1:max_iters
        % sample z, sigma, x
        z = mrf.sample_z(x(:));
        if do_noise_est
          [s s_mu] = sample_sigma_matrix(reshape(x(:),mrf.imdims), y, K_matrix);
          sigma_cur = s;
        else
          sigma_cur = sigma;
        end
        [x(:), x_mu(:)] = sample_x(mrf, z, sigma_cur,K_matrix, Wt, Kty, KtK_matrix, do_noise_est, x_avg);
        
        % running average of x and sigma after fixed burn-in phase
        if iter > burnin
          if rb
            x_tmp = x_mu(:);
            if do_noise_est, sigma_tmp = s_mu; end
          else
            x_tmp = x(:);
            if do_noise_est, sigma_tmp = s; end
          end
          x_avg = (x_tmp + c_avg * x_avg) / (c_avg + 1);
          if do_noise_est, sigma_avg = (sigma_tmp + c_avg * sigma_avg) / (c_avg + 1); end
          c_avg = c_avg + 1;
        end
        
        % measure image quality
        x_final = reshape(x_mu, mrf.imdims);
        x_restored = x_final(border+1:end-border, border+1:end-border);
        psnr0 = pml.image_proc.psnr(x_gt, x_restored);
        ssim0 = pml.image_proc.ssim_index(x_gt, x_restored);
        % display progress from every sampler
        fprintf('Iteration %02d/%02d, PSNR %.2f SSIM %.2f Sigma %.2f\n',iter,max_iters,psnr0,ssim0,sigma_cur)
    end
    
     % record elapsed time
    time = toc;
    % write individual results back to shared variables
    
    if do_noise_est, sigma_est = sigma_avg; end
           
    % compute final results
    img_restored(:) = x_avg;
    img_final = reshape(img_restored, mrf.imdims);
    img_restored = img_final(border+1:end-border, border+1:end-border);
    if do_noise_est, sigma_est = mean(sigma_est); end
    
    % measure image quality
    psnr1 = pml.image_proc.psnr(x_gt, img_restored);
    ssim1 = pml.image_proc.ssim_index(x_gt, img_restored);
    
    if psnr1 > psnr0
        psnr = psnr1;ssim = ssim1;
    else
        psnr = psnr0;ssim = ssim0;
        img_restored = x_restored;
    end
    
    % show results
    fprintf('sigma_est = %.2f, sigma_gt = %.2f :: PSNR = %.2fdB, SSIM = %.3f, runtime = %.2fm\n', ...
            sigma_est, sigma_gt, psnr, ssim, time(end)/60)
    if doplot
      figure(1), clf
      plot_images(1,x_gt,img_restored);
    end
  end
  
end

%%=====================================================     Deblurring functions
function [x, x_mu] = sample_x(this, z, sigma, K_matrix, Wt, Kty, KtK_matrix, do_noise_est, x_mu_init)

  mrf_filter = this.filter;
  imdims = this.imdims;
  npixels = prod(this.imdims);
  nfilters = this.nfilters;
  nexperts = this.nexperts;
  
  if do_noise_est
    KtK_matrix = KtK_matrix / sigma^2;
    Kty = Kty / sigma^2;
  end
  
  N = 0;
  for i = 1:nfilters
    z{i} = z{i}(:) * this.experts{min(i,nexperts)}.precision;
    N = N + numel(z{i});
  end
  
  z = {z{:}, this.epsilon * ones(npixels, 1)};
  N = N + npixels;
  Z = spdiags(vertcat(z{:}), 0, N, N);
  
  Q_approx = (Wt' * Z * Wt) + KtK_matrix;
  Noisey = randn(size(K_matrix,1),1);
  Kt_times_noise = K_matrix' * Noisey(:)/sigma;
  
  W_sqrtZ_r = (Wt' * sqrt(Z) * randn(N,1)) + Kt_times_noise(:);
  
  % do Cholesky decomposition of sparse approximation of precision matrix Q
  [L p s] = chol(Q_approx, 'lower', 'vector'); % Q_approx(s,s) = L*L'
  assert(p == 0, 'Matrix is not positive definite.');
  % use Cholesky factors as preconditioner for PCG, employ @Q_times to multiply with precision matrix Q
  pcgfun = @(b,x0) pcg(@(x) Q_times_matrix(reshape(x,imdims), z, mrf_filter, K_matrix, sigma, s), b, 1e-6, 250, L, L', x0);
  
  % solve the two linear equations systems
  x_mu = zeros(npixels,1); x_cov = zeros(npixels,1);
  [x_mu(s)  flag1] = pcgfun(Kty(s), x_mu_init);
  [x_cov(s) flag2] = pcgfun(W_sqrtZ_r(s), []);
  x = x_mu + x_cov;
  
  assert(flag1==0 && flag2==0, ...
         'PCG didn''t converge to the desired tolerance within the allotted number of iterations.')
  
end

% Sample from the posterior distribution p(\sigma|x,z,y,K)
function [s, s_mean] = sample_sigma_matrix(x, y, K_approx)
  % sigma^-2 is gamma-distributed with parameters a and b
  a = numel(y)/2 + 1;
  b = 2 ./ (sum((y(:)-(K_approx*x(:))).^2)+eps);
  
  % draw a single sample and compute the mean of the density
  s = gamrnd(a, b).^(-1/2);
  s_mean = (a*b).^(-1/2);
  
  % make sure sigma isn't too close to 0
  s = max(1e-1,s);
end

% Matrix-vector multiplication expressed through convolutions
% Used to avoid building the full precision matrix Q
function y = Q_times_matrix(img, z, f, K_matrix, sigma, perm)
  if nargin < 6, perm = (1:numel(img))'; else perm = perm(:); end
  img(perm) = img(:); % inverse permute image
  
  nfilters = numel(f);
  y = img(:) .* z{nfilters+1};
  
  for i = 1:nfilters
    tmp = conv2(img, f{i}, 'valid');
    tmp = conv2(tmp.*reshape(z{i}, size(tmp)), f{i}(end:-1:1,end:-1:1), 'full');
    y = y + tmp(:);
  end
  
  tmp = K_matrix * img(:) / sigma;
  tmp = K_matrix' * tmp(:) / sigma;
  
  y = y + tmp(:);
  
  y = y(perm); % permute output
end

function plot_images(fh,x_gt,x)
  ims = @(I) imshow(uint8(I), 'InitialMagnification', 'fit');
  figure(fh)
  subplot(1,2,1), ims(x_gt), title 'Original'
  subplot(1,2,2), ims(x), title({sprintf('PSNR = %.2fdB, SSIM = %.3f', ...
                                 pml.image_proc.psnr(x_gt, x), pml.image_proc.ssim_index(x_gt, x))});
end

%%=============     Convert MRF object to struct (required only for parfor loop)

function s = mrf2struct(mrf)
  s = struct;
  s.imdims = mrf.imdims;
  s.experts = mrf.experts;
  s.epsilon = mrf.epsilon;
  s.conv2 = @(img, f) conv2(img, f, 'valid');
  s.filter_matrices = mrf.filter_matrices;
  s.filter = mrf.filter;
  s.nfilters = mrf.nfilters;
  s.nexperts = mrf.nexperts;
  s.sample_z = @(x) sample_z(s,x);
  s.energy = @(x) energy(s,x);
end

function z = sample_z(this, x)
  z = cell(1, this.nfilters);
  for i = 1:this.nfilters
    d_filter = this.conv2(reshape(x, this.imdims), this.filter{i});
    expert = this.experts{min(i,this.nexperts)};
    pz = expert.z_distribution(d_filter(:)');
    z{i} = this.experts{min(i,this.nexperts)}.scales(sample_discrete(pz));
  end
end

function L = energy(this, x)
  img = reshape(x, this.imdims);
  L = 0.5 * this.epsilon * sum(x.^2);
  for i = 1:this.nfilters
    d_filter = this.conv2(img, this.filter{i});
    L = L + sum(this.experts{min(i,this.nexperts)}.energy(d_filter(:)'));
  end
end

% draw a single sample from each discrete distribution (normalized weights in columns)
% domain start & stride is assumed to be 1
function xs = sample_discrete(pmfs)
  [nweights, ndists] = size(pmfs);
  % create cdfs
  cdfs = cumsum(pmfs, 1);
  % uniform sample for each distribution
  ys = rand(1, ndists);
  % subtract uniform sample and set 1s where difference >= 0
  inds = bsxfun(@minus, cdfs, ys) >= 0;
  % multiply each column with weight indices
  inds = bsxfun(@times, inds, (1:nweights)');
  % set 0s to NaN
  inds(inds == 0) = NaN;
  % min. weight index > 0 (NaNs are ignored by 'min')
  xs = min(inds, [], 1);
end
