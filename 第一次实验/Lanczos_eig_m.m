%当 A 只有 m 个不同特征值时，对于大的 m 和小的 m，观察有限精度下 Lanczos 方法如何收敛
clear;clc;
close all
n=2000;
m_min=5;%特征值个数下限
m_max=2000;%特征值个数上限
m_num=5;%特征值个数取值个数
max_iter=1500;
err=10^-24;
m_max=min(m_max,n);
b=rand(n,1);
x_init=zeros(n,1);
k=1;
x_set=[];relative_x_set=[];iter_set=[];
lengend_content={};
for m=linspace(m_min,m_max,m_num)
    m=round(m);
    A=get_A(n,m);
    [x,relative_x,iter,stat]=my_Lanczos(A,b,x_init,max_iter,err);
    x_set=[x_set,x];
    relative_x_set=[relative_x_set,relative_x];
    iter_set=[iter_set,iter];
    plot(1:iter,log10(relative_x))
    xlabel('Iteration Steps');ylabel('log_{10}(|r|/|b|)');
    title(['n=',num2str(size(b,1))],'FontSize',15)
    lengend_content{k}=['m=',num2str(m)];
    legend(lengend_content)
    hold on
    k=k+1;
end


function [A]=get_A(n,m)
while 1
    eig_m=rand(m,1);
    [c1,c2]=unique(eig_m);
    if length(c2)==m
        break
    end
end
pos= randperm(n,m);
lamda=-1*ones(n,1);
lamda(pos)=eig_m;
lamda(find(lamda==-1))=eig_m(randi(m,[length(find(lamda==-1)),1]));
D=diag(lamda);
Q = gallery('orthog',n,2);
A=Q*D*Q';
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
