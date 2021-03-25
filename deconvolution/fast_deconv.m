function [yout] = fast_deconv(yin, k, lambda, alpha, yout0)
%
%
% fast_deconv solves the deconvolution problem in the paper (see Equation (1))
% D. Krishnan, R. Fergus: "Fast Image Deconvolution using Hyper-Laplacian
% Priors", Proceedings of NIPS 2009.
%
% This paper and the code are related to the work and code of Wang
% et. al.:
%
% Y. Wang, J. Yang, W. Yin and Y. Zhang, "A New Alternating Minimization
% Algorithm for Total Variation Image Reconstruction", SIAM Journal on
% Imaging Sciences, 1(3): 248:272, 2008.
% and their FTVd code. 
  
% Input Parameters:
%
% yin: Observed blurry and noisy input grayscale image.
% k:  convolution kernel
% lambda: parameter that balances likelihood and prior term weighting
% alpha: parameter between 0 and 2
% yout0: if this is passed in, it is used as an initialization for the
% output deblurred image; if not passed in, then the input blurry image
% is used as the initialization
%
%
% Outputs:
% yout: solution
% 
% Note: for faster LUT interpolation, please download and install
% matlabPyrTools of Eero Simoncelli from
% www.cns.nyu.edu/~lcv/software.php. The specific MeX function required
% is pointOp (used in solve_image.m).
%
% Copyright (C) 2009. Dilip Krishnan and Rob Fergus
% Email: dilip,fergus@cs.nyu.edu

% continuation parameters
beta = 1;
beta_rate = 2*sqrt(2);
beta_max = 2^8;

% number of inner iterations per outer iteration
mit_inn = 1;

[m n] = size(yin); 
% initialize with input or passed in initialization
if (nargin == 5)
  yout = yout0;
else
  yout = yin; 
end;

% make sure k is a odd-sized
if ((mod(size(k, 1), 2) ~= 1) | (mod(size(k, 2), 2) ~= 1))
  fprintf('Error - blur kernel k must be odd-sized.\n');
  return;
end;
ks = floor((size(k, 1)-1)/2);

% compute constant quantities
% see Eqn. (3) of paper
[Nomin1, Denom1, Denom2] = computeDenominator(yin, k);

% x and y gradients of yout (with circular boundary conditions)
% other gradient filters may be used here and their transpose will then need to
% be used within the inner loop (see comment below) and in the function
% computeDenominator
youtx = [diff(yout, 1, 2), yout(:,1) - yout(:,n)]; 
youty = [diff(yout, 1, 1); yout(1,:) - yout(m,:)]; 

% store some of the statistics
costfun = [];
Outiter = 0;

%% Main loop
while beta < beta_max
    Outiter = Outiter + 1; 
    %fprintf('Outer iteration %d; beta %.3g\n',Outiter, beta);
    
    gamma = beta/lambda;
    Denom = Denom1 + gamma*Denom2;
    Inniter = 0;

    for Inniter = 1:mit_inn
      
      if (0)
        %%% Compute cost function - uncomment to see the original
        % minimization function costs at every iteration
        youtk = conv2(yout, k, 'same');
        % likelihood term
        lh = sum(sum((youtk - yin).^2 ));
        
        if (alpha == 1)
          cost = (lambda/2)*lh +  sum(abs(youtx(:))) + sum(abs(youty(:)));
        else
          cost = (lambda/2)*lh +  sum(abs(youtx(:)).^alpha) + sum(abs(youty(:)).^alpha);
        end;
        %fprintf('Inniter iteration %d; cost %.3g\n', Inniter, cost);
        
        costfun = [costfun, cost];
      end;
      %
      % w-subproblem: eqn (5) of paper
      %         
      Wx = solve_image(youtx, beta, alpha); 
      Wy = solve_image(youty, beta, alpha);
                   
      % 
      %   x-subproblem: eqn (3) of paper
      % 
      % The transpose of x and y gradients; if other gradient filters
      % (such as higher-order filters) are to be used, then add them
      % below the comment above as well
      
      Wxx = [Wx(:,n) - Wx(:, 1), -diff(Wx,1,2)]; 
      Wxx = Wxx + [Wy(m,:) - Wy(1, :); -diff(Wy,1,1)]; 
        
      Fyout = (Nomin1 + gamma*fft2(Wxx))./Denom; 
      yout = real(ifft2(Fyout));
      
      % update the gradient terms with new solution
      youtx = [diff(yout, 1, 2), yout(:,1) - yout(:,n)]; 
      youty = [diff(yout, 1, 1); yout(1,:) - yout(m,:)]; 

    end %inner
    beta = beta*beta_rate;
end %Outer


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Nomin1, Denom1, Denom2] = computeDenominator(y, k)
%
% computes denominator and part of the numerator for Equation (3) of the
% paper
%
% Inputs: 
%  y: blurry and noisy input
%  k: convolution kernel  
% 
% Outputs:
%      Nomin1  -- F(K)'*F(y)
%      Denom1  -- |F(K)|.^2
%      Denom2  -- |F(D^1)|.^2 + |F(D^2)|.^2
%

sizey = size(y);
otfk  = psf2otf(k, sizey); 
Nomin1 = conj(otfk).*fft2(y);
Denom1 = abs(otfk).^2; 
% if higher-order filters are used, they must be added here too
Denom2 = abs(psf2otf([1,-1],sizey)).^2 + abs(psf2otf([1;-1],sizey)).^2;
