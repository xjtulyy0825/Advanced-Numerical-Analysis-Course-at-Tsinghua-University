%当 A 只有 m 个不同特征值时，对于大的 m 和小的 m，观察有限精度下 GMRES 方法如何收敛
clear;clc;
close all
n=2000;
m_min=5;%特征值个数下限
m_max=2000;%特征值个数上限
m_num=5;%特征值个数取值个数
m=1;%重启次数+1
max_iter=1500;
err_process=10^-24;
err=10^-13;


m_max=min(m_max,n);
b=rand(n,1);
x_init=zeros(n,1);
k=1;
x_set=[];relative_x_set=[];iter_set=[];
lengend_content={};
for  mm=linspace(m_min,m_max,m_num)

   mm=round(mm);
   A=get_A(n,mm);
    figure
    [x1,norm_r_set1,V1,H1,iter1]=GMRES(A,b,x_init,m,err_process,err,max_iter);
    [x2,norm_r_set2,V2,H2,iter2]=Arnoldi(A,b,x_init,m,err_process,err,max_iter);
    plot(1:iter1,log10(norm_r_set1(1:iter1)),'-.',1:iter2,log10(norm_r_set2(1:iter2)),'--')
    xlabel('Iteration Steps');ylabel('log_{10}(|r|/|b|)');
    [t1,t2] =title(['n=',num2str(size(b,1))],['Number of different eigenvalues:',num2str(mm)],...
        'Color','blue');
    t1.FontSize = 16;
    t2.Color=[0.7,0.2,0.4]
    t2.FontSize = 11;
    legend('GMRES','Arnoldi','location','best')
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



function [x,norm_r_set,V,H,sum_iter]=GMRES(A,b,x_init,m,err_process,err,max_iter)
sum_iter=0;
norm_r_set=[norm(b-A*x_init)/norm(b)];
while (1)
    r0=b-A*x_init;
    norm_r0=norm(r0);
    V=r0/norm(r0);
    H=[];

    for iter=1:m
        sum_iter=sum_iter+1;
        %%begin Arnoldi process
        w=A*V(:,iter);
        for raw=1:iter
            H(raw,iter)=V(:,raw)'*w;
            w=w- H(raw,iter)*V(:,raw);
        end
        H(raw+1,iter)=norm(w);

        %判断是否过程中断,中断不应该continue
        if abs(H(raw+1,iter))<err_process
            stat='process stop';
        end
        V(:,iter+1)=w/ H(raw+1,iter);
        %%end Arnoldi process
        g=[abs(norm(r0));zeros(iter,1)];
        [Q,R]=QR_tri(H(1:iter+1,1:iter));
        gm=Q(:,1)*norm_r0;

        R=R(1:iter,1:iter);
        y=R\gm(1:iter);

        x=x_init+V(:,1:iter)*y;
        norm_r=abs(gm(end));
        norm_r_set=[norm_r_set,norm_r/norm(b)];
        if (norm_r/norm(b)<err | sum_iter>=max_iter)
            break;
        end

    end
    if (norm_r/norm(b)<err | sum_iter>=max_iter)
        break;
    end
    V=V(:,1:iter);
    H=H(1:iter,1:iter);
    x_init=x;
end
end

function [Q,R]=QR_tri(H)
[~,n]=size(H);
R=H;
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

function [x,norm_r_set,V,H,sum_iter]=Arnoldi(A,b,x_init,m,err_process,err,max_iter)
norm_r_set=[norm(b-A*x_init)/norm(b)];
sum_iter=0;
while (1)
    r0=b-A*x_init;
    V=r0/norm(r0);
    H=[];

    for iter=1:m
        sum_iter=sum_iter+1;
        %%begin Arnoldi process
        w=A*V(:,iter);
        for raw=1:iter
            H(raw,iter)=V(:,raw)'*w;
            w=w- H(raw,iter)*V(:,raw);
        end
        H(raw+1,iter)=norm(w);

        %判断是否过程中断,中断不应该continue
        if abs(H(raw+1,iter))<err_process
            stat='process stop';
        end
        V(:,iter+1)=w/ H(raw+1,iter);
        %%end Arnoldi process

        %判断是否方法中断
        %     if min(svd(H))<err_method
        %         continue;
        %     end
        y=H(1:iter,1:iter)\[norm(r0);zeros(iter-1,1)];
        x=x_init+V(:,1:iter)*y;
        norm_r=abs(H(iter+1,iter)*y(end));
        norm_r_set=[norm_r_set,norm_r/norm(b)];

        if (norm_r/norm(b)<err| sum_iter>max_iter)
            break;
        end
    end
    if (norm_r/norm(b)<err | sum_iter>max_iter)
        break;
    end
    V=V(:,1:iter);
    H=H(1:iter,1:iter);
    x_init=x;
end
end
