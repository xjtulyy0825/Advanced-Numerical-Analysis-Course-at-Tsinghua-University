%对于同样的例子, 比较 CG 和 Lanczos 的计算结果
clear;clc;
close all
n=2000;
max_iter=1500;
%judge用于判断对A施加不同限制的影响.1:%特征值随机生成;2:最大特征值远大于第二大特征值;
%3:最小特征值远小于第二小特征值;最大特征值远大于第二大特征值&最小特征值远小于第二小特征值
judge=1;
err=10^-24;
b=rand(n,1);
x_init=zeros(n,1);
Q = gallery('orthog',n,2);
if judge==1%特征值随机生成
    lamda=rand(n,1);
elseif judge==2%最大特征值远大于第二大特征值
    lamda=rand(n,1);
    [~,tick]=max(lamda);
    lamda(tick)=lamda(tick)*10^4;
elseif judge==3%最小特征值远小于第二小特征值
    lamda=rand(n,1);
    [~,tick]=min(lamda);
    lamda(tick)=lamda(tick)*10^-4;
elseif judge==4%最大特征值远大于第二大特征值&最小特征值远小于第二小特征值
    lamda=rand(n,1);
    [~,tick_max]=max(lamda);
    lamda(tick_max)=lamda(tick_max)*10^4;
    [~,tick_min]=min(lamda);
    lamda(tick_min)=lamda(tick_min)*10^-4;
end
D=diag(lamda);
A=Q*D*Q';
[x,relative_x,iter,stat]=my_Lanczos(A,b,x_init,max_iter,err);
x_real=A\b;
relative_x_real=norm(b-A*x_real)/norm(b);
plot(1:iter,log10(relative_x),1:iter,log10(relative_x_real)*ones(1,iter))
xlabel('Iteration Steps');ylabel('log_{10}(|r|/|b|)');
title(['n=',num2str(size(b,1))],'FontSize',15)
legend('Lanczos','x^*')
text(3,log10(relative_x(iter)),['Cond(A)=',num2str(cond(A))],'Color','red','FontSize',14)
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
