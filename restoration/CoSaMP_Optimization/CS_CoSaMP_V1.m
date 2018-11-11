
function [img_re,PSNRr,SSIMr]=CS_CoSaMP_V1(img_cl,Phi,v,Its,imdims)

    sparsity = 0.5 * size(Phi,2);
    xp1 = cosamp_new(v,Phi,sparsity,Its);

    img_re = reshape(xp1, imdims);
    
    border = size(img_re) - size(img_cl);
    hborder = border./2;
    img_re = img_re(hborder(1)+1:end-hborder(1), hborder(2)+1:end-hborder(2));

    ERR = norm(img_re(:) - img_cl(:))./norm(img_cl(:));
    RSNR = 20.*log10(1./ERR);
    PSNRr = psnr(img_re(:),img_cl(:));
    SSIMr = ssim_index(reshape(img_re,size(img_cl)),img_cl);
    fprintf('CS_CoSaMP_2D_V1 is done! RSNR %.3f PSNR %.3f SSIM %.3f \n',RSNR,PSNRr,SSIMr);
end
