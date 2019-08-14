function [equatPF equatRe]=power_flow_equations(nK, Admittanz, V, num)

    G=real(Admittanz.Y);
    B=imag(Admittanz.Y);
    g=real(Admittanz.y);
    b=imag(Admittanz.y);

    e=real(V);
    f=imag(V);

    nVk=size(num.Vk,1);
    nPl=size(num.Pl,1);
    nQl=size(num.Ql,1);

    equatPk=zeros(nK,1);
    for n=1:nK
        i=n;
        equatPk(n)=e(i)*(G(i,:)*e-B(i,:)*f)+f(i)*(G(i,:)*f+B(i,:)*e);
    end

    equatQk=zeros(nK,1);
    for n=1:nK
        i=n;
        equatQk(n)=f(i)*(G(i,:)*e-B(i,:)*f)-e(i)*(G(i,:)*f+B(i,:)*e);
    end

    %............voltage of node..............
    equatVk=zeros(nVk,1);
    for n=1:nVk
        i=num.Vk(n);
        equatVk(n)=sqrt(e(i)^2+f(i)^2);
    end

    %...........leistung of wire.............
    equatPl=zeros(nPl,1);
    for n=1:nPl
        i=num.Pl(n,1);
        j=num.Pl(n,2);
        equatPl(n)=(e(i)^2+f(i)^2)*g(i,j)-(e(i)*e(j)+f(i)*f(j))*g(i,j)+(e(i)*f(j)-e(j)*f(i))*b(i,j);
    end

    equatQl=zeros(nQl,1);
    for n=1:nQl
        i=num.Ql(n,1);
        j=num.Ql(n,2);
        equatQl(n)=-(e(i)^2+f(i)^2)*b(i,j)+(e(i)*e(j)+f(i)*f(j))*b(i,j)+(e(i)*f(j)-e(j)*f(i))*g(i,j);
    end

    equatPF=[equatPk; equatQk];
    equatRe=[ equatPl; equatQl; equatVk];

end
