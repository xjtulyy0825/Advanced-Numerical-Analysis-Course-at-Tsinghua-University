%��������˵�� CG ����ֵ��̬. ������ = ����ʱ CG �Ľ����?
%�� A ���������ֵԶ���ڵڶ�������ֵ, ��С����ֵԶС�ڵڶ�С����ֵʱ��������������Σ�
clear;clc;
close all
n=200;
max_iter=1500;
%judge�����ж϶�Aʩ�Ӳ�ͬ���Ƶ�Ӱ��.1:%����ֵ�������;2:�������ֵԶ���ڵڶ�������ֵ;
%3:��С����ֵԶС�ڵڶ�С����ֵ;�������ֵԶ���ڵڶ�������ֵ&��С����ֵԶС�ڵڶ�С����ֵ
judge=4;
err=10^-20;
b=rand(n,1);
x_init=zeros(n,1);
Q = gallery('orthog',n,2);
if judge==1%����ֵ�������
    lamda=rand(n,1);
elseif judge==2%�������ֵԶ���ڵڶ�������ֵ
    lamda=rand(n,1);
    [~,tick]=max(lamda);
    lamda(tick)=lamda(tick)*10^4;
elseif judge==3%��С����ֵԶС�ڵڶ�С����ֵ
    lamda=rand(n,1);
    [~,tick]=min(lamda);
    lamda(tick)=lamda(tick)*10^-4;
elseif judge==4%�������ֵԶ���ڵڶ�������ֵ&��С����ֵԶС�ڵڶ�С����ֵ
    lamda=rand(n,1);
    [~,tick_max]=max(lamda);
    lamda(tick_max)=lamda(tick_max)*10^4;
    [~,tick_min]=min(lamda);
    lamda(tick_min)=lamda(tick_min)*10^-4;
end
D=diag(lamda);
A=Q*D*Q';
[x,relative_x,iter]=my_CG(A,b,x_init,max_iter,err);
x_real=A\b;
relative_x_real=norm(b-A*x_real)/norm(b);
draw_r(A,b,x_init,max_iter,err)
function [x,relative_x,iter]=my_CG(A,b,x_init,max_iter,err)
r=b-A*x_init;
p=r;
q=r'*r;
iter=1;
x=x_init;
while iter<=max_iter & norm(r)/norm(b)>err
    w=A*p;
    alpha=q/(p'*w);
    x=x+alpha*p;
    r=r-alpha*w;
    q_new=r'*r;
    if (abs(q_new) < err)
        break;
    end
    beta=q_new/q;
    p=r+beta*p;
    q=q_new;
    iter=iter+1;
end
relative_x=norm(r)/norm(b);
end
function []=draw_r(A,b,x_init,max_iter,err)
    for iter=1:max_iter
         [~,relative_x(iter),num_iter(iter)]=my_CG(A,b,x_init,iter,err);
         x_real=A\b;
         relative_x_real(iter)=norm(b-A*x_real)/norm(b);
    end
    plot(1:max_iter,log10(relative_x),1:max_iter,log10(relative_x_real))
    xlabel('Iteration Steps');ylabel('log_{10}(|r|/|b|)');
    title(['n=',num2str(size(b,1))])
    legend('GC','x^*')
    text(size(b,1),log10(relative_x(size(b,1))),'\downarrow n=k','Color','red','FontSize',14)
end

