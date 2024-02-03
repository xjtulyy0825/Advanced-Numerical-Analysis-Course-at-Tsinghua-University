%问题描述：取A∈Rn×n为一对称正定矩阵（如 Hilbert 矩阵），取M=A3,A5和Ab做预处理矩阵，
%观察预处理结果. 其中 A3,A5分别为A的3,5条主对角线构成的矩阵，
%Ab = diag(A11, A22, ..., All)，Aii 是 A 的主对角线上的子矩阵，
%n阶Hilbert矩阵Hn=[hij], hij=1/(i+j?1)
clear;clc;
close all
global I J
tic
n=50;
max_iter=1500;
err=10^-16;
b=rand(n,1);
x_init=zeros(n,1);
mode=3;%1,2,3,4,5 for A3,A5,Ab,I,A
%max_iter=min(n,max_iter);
i=1:n;
j=1:n;
[I,J]=meshgrid(i,j);
A=1./(I+J-1);%也可由hilb(n)函数一键生成
A = gallery('lehmer',n);%对称正定矩阵A也可为其他形式



% x_direc=A\b;
% relative_direc=norm(b-A*x_direc)/norm(b);
% x_pcg = pcg(A,b,10^-16,10000);
% %x_pcg_M= pcg(A,b,10^-16,10000,M);
% relative_pcg=norm(b-A*x_pcg)/norm(b);
% %relative_pcg_M=norm(b-A*x_pcg_M)/norm(b);
% cond_A=cond(A);

cond_rec={};
relative_x={};
for mode=1:5
    M=get_M(A,n,mode);
    [cond_rec{mode},relative_x{mode}]=pcg_M(A,b,M,x_init,max_iter,err);
end
plot(1:length(relative_x{1}),log10(relative_x{1}),'*--',...
    1:length(relative_x{2}),log10(relative_x{2}),'^-',...
    1:length(relative_x{3}),log10(relative_x{3}),'o-.',...
    1:length(relative_x{4}),log10(relative_x{4}),'+'...
     )
legend(['M=A_{3},Cond(C^{-1}AC^{-1})=',num2str(cond_rec{1})],...
    ['M=A_{5},Cond(C^{-1}AC^{-1})=',num2str(cond_rec{2})],...
    ['M=A_{b},Cond(C^{-1}AC^{-1})=',num2str(cond_rec{3})],...
    ['M=I,Cond(A)=',num2str(cond_rec{4})]...
   ,'Location','southeast')
title(['n=',num2str(n),'  \epsilon=',num2str(err),'  MAX\_ITER=',num2str(max_iter)])
xlabel('迭代步数')
ylabel('log_{10}|r|/|b|')

toc
function M=get_M(A,n,mode)
global I J
if mode==1
    M=A.*(abs(I-J)<3);
elseif mode==2
    M=A.*(abs(I-J)<5);
elseif mode==3
    sum_L=0;
    sum_tick=0;
    for L=1:n
        sum_L=sum_L+L;
        if sum_L>n
            L=n-(sum_L-L);
            M(sum_tick+1:sum_tick+L,sum_tick+1:sum_tick+L)=A(sum_tick+1:sum_tick+L,sum_tick+1:sum_tick+L);
            break
        end
        M(sum_tick+1:sum_tick+L,sum_tick+1:sum_tick+L)=A(sum_tick+1:sum_tick+L,sum_tick+1:sum_tick+L);
        sum_tick=sum_L;
    end
elseif mode==4
    M=eye(n);%相当于普通共轭梯度
elseif mode==5
    M=A;%数学上一步收敛
end
end




function [cond_rec,relative_x]=pcg_M(A,b,M,x_init,max_iter,err)
r=b-A*x_init;
z=M\r;
p=z;
q=r'*z;
iter=1;
x=x_init;
while iter<=max_iter & norm(r)/norm(b)>err
    w=A*p;
    alpha=q/(p'*w);
    if (abs(p'*w) < 1e-10)
        break;
    end
    x=x+alpha*p;
    r=r-alpha*w;
    z=M\r;
    q_new=r'*z;
    beta=q_new/q;
    p=z+beta*p;
    q=q_new;
    relative_x(iter)=norm(r)/norm(b);
    iter=iter+1;
end

C=sqrtm(M);
cond_rec=cond(inv(C)*A*inv(C));
end




%未修正的共轭梯度法，来自网络，仅供对比
% function [x] = conjgrad(A, b, x)
% r = b - A * x;
% p = r;
% rsold = r' * r;
%
% for i = 1:length(b)
%     Ap = A * p;
%     alpha = rsold / (p' * Ap);
%     x = x + alpha * p;
%     r = r - alpha * Ap;
%     rsnew = r' * r;
%     if sqrt(rsnew) < 1e-10
%
%         break;
%     end
%     p = r + (rsnew / rsold) * p;
%     rsold = rsnew;
% end
% end


