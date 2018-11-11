
function [img_re,PSNRr,SSIMr] = CS_TV(img_clean,A,b,imdims)

  clear opts
  opts.mu = 2^12;
  opts.beta = 2^6;
  opts.mu0 = 2^4;       % trigger continuation shceme
  opts.beta0 = 2^-2;    % trigger continuation shceme
  opts.maxcnt = 10;
  opts.tol_inn = 1e-3;
  opts.tol = 1E-6;
  opts.maxit = 50;

  % reconstruction
  [img_re, out] = TVAL3(A,b,imdims(1),imdims(2),opts);
  
  img_cl = img_clean;
  border = size(img_re) - size(img_cl);
  hborder = border./2;
  img_re = img_re(hborder(1)+1:end-hborder(1), hborder(2)+1:end-hborder(2));

  ERR = norm(img_re(:) - img_cl(:))./norm(img_cl(:));
  RSNR = 20.*log10(1./ERR);
  PSNRr = psnr(img_re(:),img_cl(:));
  SSIMr = ssim_index(reshape(img_re,size(img_cl)),img_cl);
  fprintf('CS_TV is done! RSNR %.3f PSNR %.3f SSIM %.3f \n',RSNR,PSNRr,SSIMr);

end