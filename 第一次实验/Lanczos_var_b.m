%取初始近似解为零向量,右端项b仅由A的m个不同特征向量的线性组合表示时，
%Lanczos方法的收敛性如何?数值计算中方法的收敛性和m的大小关系如何?

clear;clc;
close all
n=20;
max_iter=1500;
err=10^-24;
m_min=2;
m_max=n;
m_num=5;
x_init=zeros(n,1);
Q = gallery('orthog',n,2);
lamda=rand(n,1);
D=diag(lamda);
A=Q*D*Q';
k=1;
lengend_content={};
for m=linspace(m_min,m_max,m_num)
    m=round(m);
    b=get_b(n,m,Q);
    [x,relative_x,iter,stat]=my_Lanczos(A,b,x_init,max_iter,err);
    plot(1:iter,log10(relative_x))
    xlabel('Iteration Steps');ylabel('log_{10}(|r|/|b|)');
    title(['n=',num2str(n)],'FontSize',15)
    lengend_content{k}=['m=',num2str(m)];
    legend(lengend_content)
    hold on
    k=k+1;
end

function [b]=get_b(n,m,Q)
pos=randperm(n,m);
coef=rand(m,1);
b=Q(:,pos)*coef;
end
function [x,relative_x,iter,stat]=my_Lanczos(A,b,x_init,max_iter,err)
x=[];
stat='Not Converge';
r0=b-A*x_init;
norm_r0=norm(r0);
q_new=r0/norm_r0;
beta_old=0;
n=size(b,1);
q_old=zeros(n,1);
Q=[];
iter=1;
while iter<=max_iter
    rr=A*q_new-beta_old*q_old;
    alpha=q_new'*A*q_new;
    T(iter,iter)=alpha;
    if iter>=2
        T(iter,iter-1)=beta_old;
        T(iter-1,iter)=beta_old;
    end
    rr=rr-alpha*q_new;
    beta_new=norm(rr);
    if abs(beta_new)<err
        stat='Lucky Interrupt '
        break
    end
    Q=[Q,q_new];
    q_old=q_new;
    q_new=rr/beta_new;
    beta_old=beta_new;
    e1=[norm_r0;zeros(iter-1,1)];
    y=T\e1;
    norm_r=abs(beta_new*y(end));
    if norm_r/norm(b)<err
        stat='method converge, got the accurate result';
        break
    end
    iter=iter+1;
    x_iter=x_init+Q*y;
    x=[x,x_iter];
    relative_x(iter)=norm_r/norm(b);
end
end
