function [P,Q,E,N,obj] = ERSLDA(X,label,lambda1,lambda2,eta,dim,beta,rho,Max_iter)

[m,n] = size(X);
max_beta = 10^5;
regu = 10^-5;
options = [];
options.ReducedDim = dim;
MAX_MATRIX_SIZE = 1600; 
EIGVECTOR_RATIO = 0.1;  
X1 = X';

%---------------------S_w, S_b--------------------------                                      
[dim,~] = size(X);                                                            
nclass = max(label);                                                             
mean_X = mean(X, 2);                                                        
Sw=zeros(dim,dim);                                                           
Sb=zeros(dim,dim);
for i = 1:nclass
    inx_i = find(label==i);                                                       
    X_i = X(:,inx_i);                                                          
    mean_Xi = mean(X_i,2);                                                    
    Sw = Sw + cov( X_i',1);                                                    
    Sb = Sb + length(inx_i)*(mean_Xi-mean_X)*(mean_Xi-mean_X)';                 
end     

%-----------------Initialization P------------------------
if (~exist('options','var'))                          
    options = [];                                                               
end                                                                             

ReducedDim = 0;
if isfield(options,'ReducedDim')
    ReducedDim = options.ReducedDim;                                 
end

[nSmp,nFea] = size(X1);                                                       
if (ReducedDim > nFea) || (ReducedDim <=0)                        
    ReducedDim = nFea;                                                     
end 

if issparse(X1)                                                               
    X1 = full(X1);                                                        
end
sampleMean = mean(X1,1);                                              
X1 = (X1 - repmat(sampleMean,nSmp,1));
X2 = X1';

if ~exist('ReducedDim','var')
    ReducedDim = 0;
end

[nSmp, mFea] = size(X2);
if mFea/nSmp > 1.0713
    ddata = X2*X2';
    ddata = max(ddata,ddata');
    
    dimMatrix = size(ddata,1);
    if (ReducedDim > 0) && (dimMatrix > MAX_MATRIX_SIZE) && (ReducedDim < dimMatrix*EIGVECTOR_RATIO)
        option = struct('disp',0);
        [U, eigvalue] = eigs(ddata,ReducedDim,'la',option);
        eigvalue = diag(eigvalue);
    else
        if issparse(ddata)
            ddata = full(ddata);
        end
        
        [U, eigvalue] = eig(ddata);
        eigvalue = diag(eigvalue);
        [~, index] = sort(-eigvalue);
        eigvalue = eigvalue(index);
        U = U(:, index);
    end
    clear ddata;
    
    maxEigValue = max(abs(eigvalue));
    eigIdx = find(abs(eigvalue)/maxEigValue < 1e-10);
    eigvalue(eigIdx) = [];
    U(:,eigIdx) = [];
    
    if (ReducedDim > 0) && (ReducedDim < length(eigvalue))
        eigvalue = eigvalue(1:ReducedDim);
        U = U(:,1:ReducedDim);
    end
    
    eigvalue_Half = eigvalue.^.5;
    S =  spdiags(eigvalue_Half,0,length(eigvalue_Half),length(eigvalue_Half));

    if nargout >= 3
        eigvalue_MinusHalf = eigvalue_Half.^-1;
        V = X'*(U.*repmat(eigvalue_MinusHalf',size(U,1),1));
    end
else
    ddata = X2'*X2;
    ddata = max(ddata,ddata');
    
    dimMatrix = size(ddata,1);
    if (ReducedDim > 0) && (dimMatrix > MAX_MATRIX_SIZE) && (ReducedDim < dimMatrix*EIGVECTOR_RATIO)
        option = struct('disp',0);
        [V, eigvalue] = eigs(ddata,ReducedDim,'la',option);
        eigvalue = diag(eigvalue);
    else
        if issparse(ddata)
            ddata = full(ddata);
        end
        
        [V, eigvalue] = eig(ddata);
        eigvalue = diag(eigvalue);
        
        [~, index] = sort(-eigvalue);
        eigvalue = eigvalue(index);
        V = V(:, index);
    end
    clear ddata;
    
    maxEigValue = max(abs(eigvalue));
    eigIdx = find(abs(eigvalue)/maxEigValue < 1e-10);
    eigvalue(eigIdx) = [];
    V(:,eigIdx) = [];
    
    if (ReducedDim > 0) && (ReducedDim < length(eigvalue))
        eigvalue = eigvalue(1:ReducedDim);
        V = V(:,1:ReducedDim);
    end
    
    eigvalue_Half = eigvalue.^.5;
    S =  spdiags(eigvalue_Half,0,length(eigvalue_Half),length(eigvalue_Half));
    
    eigvalue_MinusHalf = eigvalue_Half.^-1;
    U = X*(V.*repmat(eigvalue_MinusHalf',size(V,1),1));
end

P1 = U;
eigvalue=S;

eigvalue = full(diag(eigvalue)).^2;                                       

if isfield(options,'PCARatio')                                             
    sumEig = sum(eigvalue);                                               
    sumEig = sumEig*options.PCARatio;                                
    sumNow = 0;                                                               
    for idx = 1:length(eigvalue)                                              
        sumNow = sumNow + eigvalue(idx);                                 
        if sumNow >= sumEig                                                     
            break;
        end
    end
    P1 = P1(:,1:idx);                                            
end
%-----------------Other Initialization--------------------
Q = ones(m,dim);
E = zeros(m,n);
N = zeros(m,n);
A = zeros(m,n);
v = sqrt(sum(Q.*Q,2)+eps);
D = diag(0.5./(v));

for iter = 1:Max_iter
%-------------------------P------------------------------
    if (iter == 1)
        P = P1;
    else
        M = X-E-N+A/beta;
        [U1,~,V1] = svd(M*X'*Q,'econ');
        P = U1*V1';
        clear M;
    end
%-------------------------Q-----------------------------
    M = X-E-N+A/beta;
    Q1 = 2*(Sw-regu*Sb)+lambda1*D+beta*(X*X');
    Q2 = beta*X*M'*P;
    Q = Q1\Q2;
    D=diag(0.5*(sum(Q.*Q,2)+eps).^(0.5/2-1));    
%-------------------------E------------------------------
    eps1 = lambda2/beta;
    temp_E = X-P*Q'*X-N+A/beta;
    E = sign(temp_E).*(abs(temp_E)-eps1*0.5*abs(temp_E).^(0.5-1)); 
%-------------------------N-----------------------------
    N = (beta/(2*eta+beta))*(X-P*Q'*X-E+A/beta);
%-----------------------A, beta-------------------------
    A = A+beta*(X-P*Q'*X-E-N);
    beta = min(rho*beta,max_beta);
%-----------------------Check---------------------------
    leq = X-P*Q'*X-E-N;
    obj(iter) = trace(Q'*(Sw-regu*Sb)*Q)+lambda1*sum(sum(Q.*Q).^(0.25))+lambda2*sum(sum(Q,2).^(0.5))+eta*norm(N,"fro");
    if iter > 2
        if norm(leq, Inf) < 10^-7 && abs(obj(iter)-obj(iter-1))<0.00001
            iter
            break;
        end 
    end
end
