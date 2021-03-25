function [w] = solve_image(v, beta, alpha)

% 
% solve the following component-wise separable problem
% min |w|^\alpha + \frac{\beta}{2} (w - v).^2
% 
% A LUT is used to solve the problem; when the function is first called
% for a new value of beta or alpha, a LUT is built for that beta/alpha 
% combination and for a range of values of v. The LUT stays persistent 
% between calls to solve_image. It will be recomputed the first time this
% function is called.

% range of input data and step size; increasing the range of decreasing
% the step size will increase accuracy but also increase the size of the
% LUT 
range = 10;
step  = 0.0001;

persistent lookup_v known_beta xx known_alpha
ind = find(known_beta==beta & known_alpha==alpha);
if isempty(known_beta | known_alpha)
  xx = [-range:step:range];
end
if any(ind)
  %fprintf('Reusing lookup table for beta %.3g and alpha %.3g\n', beta, alpha);
  %%% already computed 
  if (exist('pointOp') == 3) 
    % Use Eero Simoncelli's function to extrapolate
    w = pointOp(double(v),lookup_v(ind,:), -range, step, 0);
  else
    w = interp1(xx', lookup_v(ind,:)', v(:), 'linear', 'extrap');
    w = reshape(w, size(v,1), size(v,2));
  end;
else
     %%% now go and recompute xx for new value of beta and alpha
     tmp = compute_w(xx, beta, alpha);
     lookup_v =  [lookup_v; tmp(:)'];
     known_beta = [known_beta, beta];
     known_alpha = [known_alpha, alpha];
     
     %%% and lookup current v's in the new lookup table row.
     if (exist('pointOp') == 3) 
       % Use Eero Simoncelli's function to extrapolate
       w = pointOp(double(v),lookup_v(end,:), -range, step, 0);
     else
       w = interp1(xx', lookup_v(end,:)', v(:), 'linear', 'extrap');
       w = reshape(w, size(v,1), size(v,2));
     end;
     
     %fprintf('Recomputing lookup table for new value of beta %.3g and alpha %.3g\n', beta, alpha);
end 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% call different functions to solve the minimization problem 
% min |w|^\alpha + \frac{\beta}{2} (w - v).^2 for a fixed beta and alpha
%
function w = compute_w(v, beta, alpha)
  
if (abs(alpha - 1) < 1e-9)
  % assume alpha = 1.0
  w = compute_w1(v, beta);
  return;
end;

if (abs(alpha - 2/3) < 1e-9)
  % assume alpha = 2/3
  w = compute_w23(v, beta);
  return;
end;

if (abs(alpha - 1/2) < 1e-9)
  % assume alpha = 1/2
  w = compute_w12(v, beta);
  return;
end;
  
% for any other value of alpha, plug in some other generic root-finder
% here, we use Newton-Raphson
w = newton_w(v, beta, alpha);
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function w = compute_w23(v, beta)
% solve a quartic equation
% for alpha = 2/3

epsilon = 1e-6; %% tolerance on imag part of real root  
   
k = 8/(27*beta^3);
m = ones(size(v))*k;
  
% Now use formula from
% http://en.wikipedia.org/wiki/Quartic_equation (Ferrari's method)
% running our coefficients through Mathmetica (quartic_solution.nb)
% optimized to use as few operations as possible...
        
%%% precompute certain terms
v2 = v .* v;
v3 = v2 .* v;
v4 = v3 .* v;
m2 = m .* m;
m3 = m2 .* m;
  
%% Compute alpha & beta
alpha = -1.125*v2;
beta2 = 0.25*v3;
  
%%% Compute p,q,r and u directly.
q = -0.125*(m.*v2);
r1 = -q/2 + sqrt(-m3/27 + (m2.*v4)/256);

u = exp(log(r1)/3); 
y = 2*(-5/18*alpha + u + (m./(3*u))); 
    
W = sqrt(alpha./3 + y);
  
%%% now form all 4 roots
root = zeros(size(v,1),size(v,2),4);
root(:,:,1) = 0.75.*v  +  0.5.*(W + sqrt(-(alpha + y + beta2./W )));
root(:,:,2) = 0.75.*v  +  0.5.*(W - sqrt(-(alpha + y + beta2./W )));
root(:,:,3) = 0.75.*v  +  0.5.*(-W + sqrt(-(alpha + y - beta2./W )));
root(:,:,4) = 0.75.*v  +  0.5.*(-W - sqrt(-(alpha + y - beta2./W )));
  
    
%%%%%% Now pick the correct root, including zero option.
  
%%% Clever fast approach that avoids lookups
v2 = repmat(v,[1 1 4]); 
sv2 = sign(v2);
rsv2 = real(root).*sv2;
    
%%% condensed fast version
%%%             take out imaginary                roots above v/2            but below v
root_flag3 = sort(((abs(imag(root))<epsilon) & ((rsv2)>(abs(v2)/2)) &  ((rsv2)<(abs(v2)))).*rsv2,3,'descend').*sv2;
%%% take best
w=root_flag3(:,:,1);
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function w = compute_w12(v, beta)
% solve a cubic equation
% for alpha = 1/2
   
epsilon = 1e-6; %% tolerance on imag part of real root  
    
k = -0.25/beta^2;
m = ones(size(v))*k.*sign(v);

%%%%%%%%%%%%%%%%%%%%%%%%%%% Compute the roots (all 3)
t1 = (2/3)*v; 
  
v2 = v .* v;
v3 = v2 .* v;
  
%%% slow (50% of time), not clear how to speed up...
t2 = exp(log(-27*m - 2*v3 + (3*sqrt(3))*sqrt(27*m.^2 + 4*m.*v3))/3);
  
t3 = v2./t2;
  
%%% find all 3 roots
root = zeros(size(v,1),size(v,2),3);
root(:,:,1) = t1 + (2^(1/3))/3*t3 + (t2/(3*2^(1/3)));
root(:,:,2) = t1 - ((1+i*sqrt(3))/(3*2^(2/3)))*t3 - ((1-i*sqrt(3))/(6*2^(1/3)))*t2;
root(:,:,3) = t1 - ((1-i*sqrt(3))/(3*2^(2/3)))*t3 - ((1+i*sqrt(3))/(6*2^(1/3)))*t2;

root(find(isnan(root) | isinf(root))) = 0; %%% catch 0/0 case

%%%%%%%%%%%%%%%%%%%%%%%%%%%% Pick the right root
%%% Clever fast approach that avoids lookups
v2 = repmat(v,[1 1 3]); 
sv2 = sign(v2);
rsv2 = real(root).*sv2;
root_flag3 = sort(((abs(imag(root))<epsilon) & ((rsv2)>(2*abs(v2)/3)) &  ((rsv2)<(abs(v2)))).*rsv2,3,'descend').*sv2;
%%% take best
w=root_flag3(:,:,1);
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function w = compute_w1(v, beta)
% solve a simple max problem for alpha = 1

w = max(abs(v) - 1/beta, 0).*sign(v);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function w = newton_w(v, beta, alpha)
  
% for a general alpha, use Newton-Raphson; more accurate root-finders may
% be substituted here; we are finding the roots of the equation:
% \alpha*|w|^{\alpha - 1} + \beta*(v - w) = 0

iterations = 4;

x = v;

for a=1:iterations
  fd = (alpha)*sign(x).*abs(x).^(alpha-1)+beta*(x-v);
  fdd = alpha*(alpha-1)*abs(x).^(alpha-2)+beta;

  x = x - fd./fdd;
end;

q = find(isnan(x));
x(q) = 0;

% check whether the zero solution is the better one
z = beta/2*v.^2;
f   = abs(x).^alpha + beta/2*(x-v).^2;
w = (f<z).*x;
