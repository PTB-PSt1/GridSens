function [nX xhat Pfilter Vhat Shat x0 Pfilter0 Q R A] = initialize_Kalman(nK, num, T)

    nMeas=size(num.Pk,1)+size(num.Qk,1)+size(num.Vk,1)+size(num.Pl,1)+size(num.Ql,1);   %number of measurements
    nX=2*nK-size(num.Pk,1)-size(num.Qk,1);                                              %number of states

    xhat=zeros(nX,T);               %states
    Pfilter=cell(1,T);              %P
    Vhat=zeros(2*nK,T);             %volatges
    Shat=zeros(2*nK,T);             %powers

    x0=zeros(nX,1);                 % start value at all states
    Pfilter0=1*eye(nX);             % start value P


    Q=10*eye(nX);                   % difference within the dynamic equations
    R=0.0001*eye(nMeas);            % difference within the static equations (uncertainties)

    r=0.95;
    A=eye(nX)*r;

end
