%取初始近似解为零向量,右端项b仅由A的m个不同特征向量的线性组合表示时，
%Lanczos方法的收敛性如何?数值计算中方法的收敛性和m的大小关系如何?

clear;clc;
tic
close all
n=1100;
max_iter=1500;
err=10^-22;
gap_min=0.1;
gap_max=2;
gap_num=6;
x_init=zeros(n,1);
Q = gallery('orthog',n,2);
b=rand(n,1);
tick=1;
for gap=linspace(gap_min,gap_max,gap_num)
    lamda=gap+rand(n,1);
    for k=1:n
        lamda(k)=lamda(k)*(-1)^(randi(2,1));
    end
    D=diag(lamda);
    A=Q*D*Q';
    [~,singular,~]=svd(A);
    singular=min(diag(singular));
    [x1,relative_x1,iter1]=my_MINRES(A,b,x_init,max_iter,err);
    figure
    plot(1:iter1-1,log10(relative_x1));hold on
    [x2,relative_x2,iter2,stat]=my_Lanczos(A,b,x_init,max_iter,err);
    plot(1:iter2,log10(relative_x2));hold on
    title({['n=',num2str(n)],...
        ['Singular values of matrices=',num2str(singular)]},'Color','blue')
    xlabel('迭代步数')
    ylabel('log_{10}|r|/|b|')
    legend('MINRES','Lanczos')
    tick=tick+1;
end
toc
function [x,relative_x,iter]=my_MINRES(A,b,x_init,max_iter,err)
relative_x=[];
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
    T_append=[T;zeros(1,iter-1),beta_new];
    [QQ,R]=QR_tri(T_append);
    R=R(1:iter,1:iter);
    g_append=QQ(:,1)*norm_r0;
    g=g_append(1:iter);
    norm_r=abs(g_append(end));
    if norm_r/norm(b)<err
        stat='method converge, got the accurate result';
        break
    end
    relative_x_iter=norm_r/norm(b)
    relative_x=[relative_x,relative_x_iter];
    iter=iter+1
end
y=get_y(R,g,iter);
x=x_init+Q*y;
end
function [y]=get_y(R,g,iter)
y(iter)=g(iter)/R(iter,iter);
for k=iter-1:-1:1
    y(k)=(g(k)-R(k,k+1))/R(k,k);
end
y=y';
end

function [Q,R]=QR_tri(T)
[~,n]=size(T);
R=T;
R_temp=R;
Q=eye(n+1,n);
for k=1:n
    R_temp([k,k+1],:)=R([k,k+1],:);
    c=R_temp(k,k)/sqrt(R_temp(k,k)^2+R_temp(k+1,k)^2);
    s=R_temp(k+1,k)/sqrt(R_temp(k,k)^2+R_temp(k+1,k)^2);
    R(k,:)=c*R_temp(k,:)+s*R_temp(k+1,:);
    R(k+1,:)=-s*R_temp(k,:)+c*R_temp(k+1,:);
    Q_temp=Q;
    Q(k,:)=c*Q_temp(k,:)+s*Q_temp(k+1,:);
    Q(k+1,:)=-s*Q_temp(k,:)+c*Q_temp(k+1,:);
end
R=R(1:n,1:n);
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
    relative_x(iter)=norm_r/norm(b);
end
x=x_init+Q*y;
end
