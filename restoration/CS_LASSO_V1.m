
function [img_re,RSNR,PSNRr,SSIMr]=CS_LASSO_V1(img_cl,Phi, v,imdims)

    A = Phi;
    % measurement matrix
    Afun = @(z) A*z;
    Atfun = @(z) A'*z;
    % obsevations
    b = v;
    % initial point
    x0 = Atfun(b);

    epsilon = 5e-3;

    %LASSO
    xp1 = l1qc_logbarrier(x0, Afun, Atfun, b, epsilon,1e-3, 50, 1e-8, 200);
    %TV
%     xp2 = tveq_logbarrier(x0, Afun, Atfun, b, 1e-1, 2, 1e-8, 500);

    img_re = reshape(xp1, imdims);
    
    border = size(img_re) - size(img_cl);
    hborder = border./2;
    img_re = img_re(hborder(1)+1:end-hborder(1), hborder(2)+1:end-hborder(2));

    ERR = norm(img_re(:) - img_cl(:))./norm(img_cl(:));
    RSNR = 20.*log10(1./ERR);
    PSNRr = psnr(img_re(:),img_cl(:));
    SSIMr = ssim_index(reshape(img_re,size(img_cl)),img_cl);
    fprintf('CS_LASSO_2D_V1 is done! RSNR %.3f PSNR %.3f SSIM %.3f \n',RSNR,PSNRr,SSIMr);
end
