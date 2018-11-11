%% DEMO_LEARNING - learning BCNNs model from training dataset
%
%  Author:  Xinjie Lan
%  Contact: lxjbit@udel.edu

function demo_learning

  clear all;
  close all;
  clc;
  
  [prev_dir, base_dir] = adjust_path;
  
  % ================================== load suitable training data
  imdims = [20 20]; patches_per_image = 40;
  database = 'training91';
  data = image_patches.load(database, imdims, patches_per_image);
  
  % ================================== specify BCNN architecture,i.e. filter dimension, the number of filters,and GSM scales.
  learn_filter = true;
  neighbortype = 'N2';
  nfilters = 8;
  gsm = -7:2:7;
    
  %% ================================= options for stochastic gradient descent
  options = struct;
  options.MaxBatches      = 100;
  options.MinibatchSize   = 20;
  options.LearningRate    = 0.25;
  options.ConvergenceCheck = 1;
  options.LearningRateFactor = @(batch,minibatch) 1;
  options.LatestGradientWeight = 0.1;
  options.NeighborhoodSystem = neighbortype;
  options.FilterNumber = nfilters;
  options.GSMScale = gsm;
  options.ProjectPath = pwd;
  %% ================================= initialize cnn
  cnn = init_cnn(imdims,neighbortype,nfilters,gsm,learn_filter);
  options.Model = cnn;
  %% ================================= learn BCNN model by contrastive divergence learning algorithm
  tic, [cnn, learning_report] = cnn.cd(data, 1, options); toc
  
  display_report(cnn, learning_report);
  
  adjust_path(prev_dir, base_dir);
end

function [prev_dir, base_dir] = adjust_path(prev_dir, base_dir)
  if nargin == 2
    % restore working directory
    % restoring the path sometimes confuses MATLAB when running the code again ("clear classes" helps)
    cd(prev_dir); % rmpath(base_dir); 
  else
    % save working directory and go to correct directory
    prev_dir = pwd; file_dir = fileparts(mfilename('fullpath')); cd(file_dir);
    last = @(v) v(end); base_dir = file_dir(1:last(strfind(file_dir, filesep))-1);
    % add base directory to path
    addpath(base_dir);
  end
end

function cnn = init_cnn(imdims,neighbortype,nfilters,gsm,learn_filter)
  cnn = pml.distributions.gsm_cnn;
  cnn.conditional_sampling = true;
  cnn.imdims = imdims;
  cnn.learn_filter = learn_filter;
  
  cnn.experts{1}.precision = 1 / 500;
  cnn.experts{1}.scales = exp(gsm);
  cnn.experts{1}.weights = ones(cnn.experts{1}.nscales, 1);
  
  nexperts = nfilters; 
  switch neighbortype
      % NS1 structure,
      case 'N1'   
          filter_size = [3 3];
          cnn.filtertype = repmat([0;1;0;1;1;1;0;1;0],1,nfilters);
      % NS2 structure
      case 'N2'
          filter_size = [3 3];  
          cnn.filtertype = ones(prod(filter_size),nfilters);
      % NS3 structure
      case 'N3'
          filter_size = [5 5];
%           cnn.filtertype = ones(prod(filter_size),nfilters);
          mask = [0,0,1,0,0;0,1,1,1,0;1,1,1,1,1;0,1,1,1,0;0,0,1,0,0];
          cnn.filtertype = repmat(mask(:),1,nfilters);
      otherwise
          filter_size = [3 3];  
          cnn.filtertype = ones(prod(filter_size),nfilters);
  end
  
  J = randn(prod(filter_size), nexperts);
  A = eye(prod(filter_size));
  cnn = cnn.set_filter(A, J, filter_size);
  cnn.experts = repmat({cnn.experts{1}}, 1, nexperts);
end


function display_report(mrf, report)
  
  fhandle = 1;
  is_pairwise = isa(mrf, 'pml.distributions.gsm_pairwise_mrf');
  
  if is_pairwise, colormap(lines(mrf.nexperts)); else colormap(jet(mrf.nexperts)); end
  colors = colormap; colormap jet
  
  % generate weight indices for each expert
  ntotalweights = 0;
  weight_idx = cell(1, mrf.nexperts);
  for i = 1:mrf.nexperts
    weight_idx{i} = ntotalweights+1:ntotalweights+mrf.experts{i}.nscales;
    ntotalweights = weight_idx{i}(end);
  end
  
  %%=========================================================     weights progress
  figure(fhandle), clf, fhandle = fhandle + 1;
  func = @(weights) bsxfun(@rdivide, exp(weights), sum(exp(weights),2));
  for i = 1:mrf.nexperts
    weights = func(report.iter_x(weight_idx{i},:)');
    [nminibatches, nweights] = size(weights);
    fprintf('weights of expert %d: ', i), disp(mrf.experts{i}.weights')
    subplot(mrf.nexperts,1,i), plot(repmat((0:nminibatches-1)',1,nweights), weights)
    if i == 1, title 'Weight progress', end
    axis tight
  end
  
  if ~is_pairwise
    %%============================================     progress of filter parameters
    figure(fhandle), clf, fhandle = fhandle + 1;
    nweights = length(mrf.weights);
    nfilterparams = size(mrf.J_tilde,1);
    Fmeans = zeros(mrf.nfilters, size(report.iter_x,2));
    for i = 1:mrf.nfilters
      s = nweights+1+(i-1)*nfilterparams;
      F = report.iter_x(s:s-1+nfilterparams,:);
      F = mrf.A' * F;
      Fmeans(i,:) = mean(F);
      plot(F'), hold on
    end
    title 'Filter Progress'
    
    %%============================================================     final filters
    figure(fhandle), clf, colormap(gray(256)), fhandle = fhandle + 1;
    sqr = ceil(sqrt(mrf.nfilters));
    for i = 1:mrf.nfilters
      subplot(sqr,sqr,i), imagesc(mrf.filter(i)), axis image off
      colorbar, title(sprintf('Filter %d', i))
      ax = axis;
      line([ax(1) ax(3); ax(1) ax(4); ax(2) ax(4); ax(2) ax(3)], ...
           [ax(1) ax(4); ax(2) ax(4); ax(2) ax(3); ax(1) ax(3)], 'color', colors(i,:), 'linewidth', 5)
    end
    
  end
  
  %%===================================     experts: weight distribution and shape
  figure(fhandle), clf, fhandle = fhandle + 1;
  R = -255:.1:255;
  w = arrayfun(@(i) {mrf.experts{i}.weights}, 1:mrf.nexperts);
  l = arrayfun(@(i) {mrf.experts{i}.eval(R)'}, 1:mrf.nexperts);
  w = horzcat(w{:}); l = horzcat(l{:});
  if is_pairwise, colormap(lines(mrf.nexperts)); else colormap(jet(mrf.nexperts)); end
  
  subplot(2,1,1)
  if is_pairwise, bar(log(mrf.experts{1}.scales), w); else bar(log(mrf.experts{1}.scales), w, 2); end
  axis tight; ax = axis; axis([ax(1), ax(2), 0, 1]); title 'Weights'
  
  subplot(2,1,2)
  for i = 1:size(l,2)
    semilogy(R, l(:,i), 'color', colors(i,:), 'linewidth', 1); hold on
  end
  axis tight, title 'Experts'
  
end