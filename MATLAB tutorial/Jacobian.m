function [JacPF JacRe]=Jacobian(nK, Admittanz, V, num)
    %calculate the jacobian

    G=real(Admittanz.Y);
    B=imag(Admittanz.Y);
    g=real(Admittanz.y);
    b=imag(Admittanz.y);

    e=real(V);
    f=imag(V);

    nPk=size(num.Pk,1);
    nQk=size(num.Qk,1);
    nVk=size(num.Vk,1);
    nPl=size(num.Pl,1);
    nQl=size(num.Ql,1);

    JacPk=[];
    JacQk=[];
    JacVk=[];
    JacPl=[];
    JacQl=[];

    %% [dPk/de dPk/df]
    for k=1:nK
        i=k;
        %derivative regard to e
        for j=2:nK
            if i==j
                JacPk(k,j-1)=G(i,:)*e-B(i,:)*f+G(i,i)*e(i)+B(i,i)*f(i);
            else
                JacPk(k,j-1)=G(i,j)*e(i)+B(i,j)*f(i);
            end
        end
        %derivative regard to f
        for j=2:nK
            if i==j
                JacPk(k,j+nK-2)=B(i,:)*e+G(i,:)*f-B(i,i)*e(i)+G(i,i)*f(i);
            else
                JacPk(k,j+nK-2)=-B(i,j)*e(i)+G(i,j)*f(i);
            end
        end
    end

    %% [dQk/de dQk/df]
    for k=1:nK
        i=k;
        %derivative regard to e
        for j=2:nK
            if i==j
                JacQk(k,j-1)=-(B(i,:)*e+G(i,:)*f)-B(i,i)*e(i)+G(i,i)*f(i);
            else
                JacQk(k,j-1)=-B(i,j)*e(i)+G(i,j)*f(i);
            end
        end
        %derivative regard to f
        for j=2:nK
            if i==j
                JacQk(k,j+nK-2)=G(i,:)*e-B(i,:)*f-G(i,i)*e(i)-B(i,i)*f(i);
            else
                JacQk(k,j+nK-2)=-G(i,j)*e(i)-B(i,j)*f(i);
            end
        end
    end

    %% [dVk/de dVk/df]
    for k=1:nVk
        i=num.Vk(k);
        %derivative regard to e
        for j=2:nK
            if i==j
                JacVk(k,j-1)=e(i)/sqrt(e(i)^2+f(i)^2);
            else
                JacVk(k,j-1)=0;
            end
        end
        %derivative regard to f
        for j=2:nK
            if i==j
                JacVk(k,j+nK-2)=f(i)/sqrt(e(i)^2+f(i)^2);
            else
                JacVk(k,j+nK-2)=0;
            end
        end
    end

    %% [dPl/de dPl/df]
    for k=1:nPl
        i=num.Pl(k,1);
        j=num.Pl(k,2);
        %derivative regard to e
        for n=2:nK
            if n==i
                JacPl(k,n-1)=2*g(i,j)*e(i)-g(i,j)*e(j)+b(i,j)*f(j);
            elseif n==j
                JacPl(k,n-1)=-g(i,j)*e(i)-b(i,j)*f(i);
            else
                JacPl(k,n-1)=0;
            end
        end
        %derivative regard to f
        for n=2:nK
            if n==i
                JacPl(k,n+nK-2)=2*g(i,j)*f(i)-g(i,j)*f(j)-b(i,j)*e(j);
            elseif n==j
                JacPl(k,n+nK-2)=b(i,j)*e(i)-g(i,j)*f(i);
            else
                JacPl(k,n+nK-2)=0;
            end
        end
    end

    %% [dQl/de dQl/df]
    for k=1:nQl
        i=num.Ql(k,1);
        j=num.Ql(k,2);
        %derivative regard to e
        for n=2:nK
            if n==i
                JacQl(k,n-1)=-2*b(i,j)*e(i)+b(i,j)*e(j)+g(i,j)*f(j);
            elseif n==j
                JacQl(k,n-1)=b(i,j)*e(i)-g(i,j)*f(i);
            else
                JacQl(k,n-1)=0;
            end
        end
        %derivative regard to f
        for n=2:nK
            if n==i
                JacQl(k,n+nK-2)=-2*b(i,j)*f(i)+b(i,j)*f(j)-g(i,j)*e(j);
            elseif n==j
                JacQl(k,n+nK-2)=g(i,j)*e(i)+b(i,j)*f(i);
            else
                JacQl(k,n+nK-2)=0;
            end
        end
    end
    JacPF=[JacPk; JacQk;];
    JacRe=[ JacPl; JacQl; JacVk];

end
