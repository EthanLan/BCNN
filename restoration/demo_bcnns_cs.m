%% DEMO_BCNNs - Demonstrate the superiority of BCNNs method over other classical SCS methods (BCS and TV)
% 
%  Author:  Xinjie Lan, University of Delaware
%  Contact: lxjbit@udel.edu

function demo_bcnns_cs
  close all;
  clear all;
  clc;
  
  [prev_dir, base_dir] = adjust_path;
  
  path(path,genpath(pwd));
  
  fprintf('Demo of BCNNs for CS restoration. \n')
  
  % The initialization of noise variance, this parameter is not important, BCNN can automatically estimate the real noise variance
  sigma = 2.25;
  % CS measurement ratio = |y|/|x|
  ratio = 0.25;
  
  %% Load BCNNs model
  % There are four models in the local folder. Different models have
  % different parameters,you can find detailed information of models in Table II of Section V
%   modelname = 'BCNN1';
%   modelname = 'BCNN2';
  modelname = 'BCNN3';
%   modelname = 'BCNN4';
  str_e2 = [modelname,'.mat'];
  bcnn_model1 = load(str_e2);
  bcnn = bcnn_model1.cnn;
  
  %% Specify the image dimension and index
  imdims = [64,64];
  idx = 8; % choose a test image from 'images' folder
  %% Generate the Gaussian measurement matrix 
  npixels = prod(imdims);
  M = npixels;
  N = round(M * ratio);    % number of CS measurements
  K_matrix = randn(N,M);
  K_matrix = K_matrix./repmat(sqrt(sum(K_matrix.^2,1)),[N,1]);
  K_matrix = sparse(K_matrix); 
  KtK_matrix = (K_matrix'*K_matrix);
  border = 5;  
  
  %% Obtain the CS measurement 'y'
  img_orign = double(imread(['images/', num2str(idx,'%02d'),'.png']));
  img_clean = imresize(img_orign, imdims,'bicubic');
  y = K_matrix * img_clean(:);
  Kty = K_matrix' * y(:);
  img_cleans = img_clean(border+1:end-border, border+1:end-border);
  
  %% Image restoration methods
  fprintf('Reconstrucing the %02dth image.\n',idx);
  % 2D wavelet transformation and sparse signal -- theta0 
  [C0, S0] = wavedec2(img_clean, 5, 'haar');
  y_wt = K_matrix * C0(:);
  % Parent and children relationships
  [IdxParent, IdxChildren, Ms]=WaveRelation2D(C0, S0);
  %% BCS    
  [img_re0,PSNRr0,SSIMr0] = BCS_Neig_2D_V1(img_clean,K_matrix,y_wt,Ms,S0,'haar');
  %% TV    
  [img_re1,PSNRr1,SSIMr1] = CS_TV(img_clean,K_matrix,y,imdims);
  %% BCNNs
  bcnn.imdims = size(img_clean);
  [img_re2,PSNRr2,SSIMr2] = BCNN_CS_restoration(border,bcnn,y, K_matrix, KtK_matrix,Kty,img_cleans, sigma, sigma, true, false);
  
  figure;
  subplot(1,3,1)
  imagesc(img_re0); colormap gray;
  title(sprintf('BCS PSNR %.2f, SSIM %.2f',PSNRr0,SSIMr0));
  subplot(1,3,2)
  imagesc(img_re1); colormap gray;
  title(sprintf('TV PSNR %.2f, SSIM %.2f',PSNRr1,SSIMr1));
  subplot(1,3,3)
  imagesc(img_re2); colormap gray;
  title(sprintf('%s, PSNR %.2f, SSIM %.2f',modelname,PSNRr2,SSIMr2));

  
  adjust_path(prev_dir, base_dir);
end

function [prev_dir, base_dir] = adjust_path(prev_dir, base_dir)
  if nargin == 2
    % restore working directory
    % restoring the path sometimes confuses MATLAB when running the code again ("clear classes" helps)
    cd(prev_dir); % rmpath(base_dir); 
    warning('on','MATLAB:mir_warning_maybe_uninitialized_temporary')
  else
    % save working directory and go to correct directory
    prev_dir = pwd; file_dir = fileparts(mfilename('fullpath')); cd(file_dir);
    last = @(v) v(end); base_dir = file_dir(1:last(strfind(file_dir, filesep))-1);
    % add base directory to path
    addpath(base_dir);
    warning('off','MATLAB:mir_warning_maybe_uninitialized_temporary')
  end
end