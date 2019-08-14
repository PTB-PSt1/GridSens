function [nK Admittanz]=admittance_matrix(filename)

    %% data
    [Ndleitung measure baseMVA] = network_data(filename);

    nK=max(max(Ndleitung(:,1:2)));
    %%
    dimension_leitung_daten=size(Ndleitung);
    nL=dimension_leitung_daten(1);
    r_matrix=zeros(nK);
    x_matrix=zeros(nK);

    for i=1:nL
        k_start=Ndleitung(i,1);
        k_end=Ndleitung(i,2);
        r_matrix(k_start,k_end)=Ndleitung(i,3);
        x_matrix(k_start,k_end)=Ndleitung(i,4);
    end
    r_matrix=r_matrix+r_matrix';
    x_matrix=x_matrix+x_matrix';

    %% admittance y
    z_matrix=r_matrix+x_matrix*1j;
    Admittanz.y=zeros(nK);
    for i=1:nK
        for j=1:nK
            if z_matrix(i,j)~=0;
                Admittanz.y(i,j)=1/z_matrix(i,j);
            end
        end
    end

    %% admittance matrix Y
    Admittanz.Y=-Admittanz.y;
    for i=1:nK
        Admittanz.Y(i,i)=sum(Admittanz.y(i,:));
    end

end
