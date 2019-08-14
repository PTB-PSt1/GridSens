function [D] = Logi_matrix_D(nK, num, numfc)

    M = eye(nK);

    D.mP=M(:,num.Pk);    %measurement
    D.mQ=M(:,num.Qk);    %measurement
    D.nmP=M(:,numfc.Pk);    %forecast
    D.nmQ=M(:,numfc.Qk);    %forecast
    % D.mS=mdiag(D.mP,D.mQ);
    % D.nmS=mdiag(D.nmP,D.nmQ);
    DemP=size(D.mP);
    DemQ=size(D.mQ);
    mdP=zeros(DemP);
    mdQ=zeros(DemQ);
    D.mS=[D.mP,mdP;mdQ,D.mQ];

    DemP=size(D.nmP);
    DemQ=size(D.nmQ);
    mdP=zeros(DemP);
    mdQ=zeros(DemQ);
    D.nmS=[D.nmP,mdP;mdQ,D.nmQ];

end
