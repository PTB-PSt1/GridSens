%% Introduction
% This Matlab script runs the complete NLO algorithm and uses the supporting Matlab functions located in this folder.
% The approach is based on
% "State observation in medium-voltage grids with incomplete measurement infrastructure through online correction of power forecasts"
% which was extended by Sascha Eichstaedt, Guosong Lin, Natalia Makarava and Franko Schmaehling (all PTB) to allow for any kind of electric measurements.
%
% Data being used in this script is located in the Excel file "NLOData.xlsx", which contains measurement data for all buses.
% For learning purposes, in a first step the buses are chose which are considered without instrumentation and for these pseudo-measurements
% are assigned. Therefore, a new Excel file is created by this script which can be deleted afterwards.
% In practice, though, measurements and pseudo-measurements are determined from the start.



%% misc (clear command line, close all figures and delete the whole workspace)
clc
close all
clear all

%% options for the iterated extended Kalman filter
accuracy = 1e-9;
iterstop = 50;

%% Read measurement data
% read measurement data - at this point all buses are considered to be instrumented
filename = 'NLOData.xlsx';
sheet1 = xlsread(filename,1);
% Active and reactive bus power are changed for those buses which are to be considered as pseudo-measurements
Pk = xlsread(filename,2);
Qk = xlsread(filename,3);
% save the other data for later usage
sheet4 = xlsread(filename,4);
sheet5 = xlsread(filename,5);
sheet6 = xlsread(filename,6);
sheet7 = xlsread(filename,7);
% read pseudo-measurements for bus power from file
pseudoPk = xlsread(filename,8);
pseudoQk = xlsread(filename,9);

%% Create scenario with pseudo-measurements
% In this example we move node 3 from 'measurements' to 'pseudo-measurements'
% and move node 6 from 'pseudo-measurements' to 'measurements'
tmp_Pk = Pk(2,:); tmp_Qk = Qk(2,:);
Pk(2,:) = []; Qk(2,:) = [];
tmp_pseudoPk = pseudoPk(3,:); tmp_pseudoQk = pseudoQk(3,:);
pseudoPk(2,:) = []; pseudoQk(2,:) = [];
Pk(end+1,:) = tmp_Pk; Qk(end+1,:) = tmp_Qk;
pseudoPk(end+1,:) = tmp_pseudoPk; pseudoQk(end+1,:) = tmp_pseudoQk;
[~,sind] = sort(Pk(:,1)); Pk = Pk(sind,:);
[~,sind] = sort(Qk(:,1)); Qk = Qk(sind,:);
[~,sind] = sort(pseudoPk(:,1)); pseudoPk = pseudoPk(sind,:);
[~,sind] = sort(pseudoQk(:,1)); pseudoQk = pseudoQk(sind,:);
filename = [filename(1:strfind(filename,'.')-1),'_modified.xlsx'];
%save this configuration, write xls-file
xlswrite(filename,sheet1,1);
xlswrite(filename,Pk,2); xlswrite(filename,Qk,3);
xlswrite(filename,sheet4,4); xlswrite(filename,sheet5,5); xlswrite(filename,sheet6,6); xlswrite(filename,sheet7,7);
xlswrite(filename,pseudoPk,8); xlswrite(filename,pseudoQk,9);


%% NLO Step 1: read data files and create structures
% -> Define the admittance matrix, measurements at nodes (active/reactive power and voltage),
% measurements on branches (active/reactive power and voltage), forecast nodes (active/reactive power))
[nK Admittanz] = admittance_matrix(filename);
[num measure numfc forecast T] = read_meas_structure(filename);
[D] = Logi_matrix_D(nK, num, numfc);

% Array of all measurements organized as Matlab structs
Mess=[measure.Pk; measure.Qk; measure.Pl;  measure.Ql; measure.Vk];

% Arrays of active and reactive bus powers at all nodes constructed by combining pseudo-measurements and actual measurements
uPk=D.mP*measure.Pk + D.nmP*forecast.Pk;
uQk=D.mQ*measure.Qk + D.nmQ*forecast.Qk;
uSk=[uPk;uQk];

%% NLO Step 2: Initialize the Kalman filter
[nX xhat Pfilter Vhat Shat x0 Pfilter0 Q R A] = initialize_Kalman(nK, num, T);
Vhat0=[ones(nK,1); zeros(nK,1)];
if ~isempty(num.Vk)
    Vhat0(num.Vk)=measure.Vk(:,1);
end
eta=zeros(2*nK,1);

%% NLO Step 3: run the Kalman filter
for time=1:T
    if time==1   % At the first time step no previous information is available.
        xhatfc = x0;
        Pfilterfc = A*Pfilter0*A'+Q;
        mu = Vhat0;
    else        % Use previous estimate to forecast next value using the dynamic model.
        xhatfc = A*xhat(:,time-1);
        Pfilterfc = A*Pfilter{time-1}*A'+Q;
        mu = Vhat(:,time-1);
    end
    eta=xhatfc;
    varstop1=1;
    j1=1;
    while ((varstop1 > accuracy) && (j1 < iterstop)) % repeat the Kalman prediction step until errors due to linearization are minimized.
        %iterated extended kalman filter (IEKF)
        V = mu(1:nK)+1j*mu(nK+1:end);
        Vef = [real(V(2:end)); imag(V(2:end))];
        for anzPF = 1:10  % fixed number of power flow iterations to calculate nodal volatges
            [equatPF equatRe]  = power_flow_equations(nK, Admittanz, V, num);    % assignment of equations
            [JacPF JacRe] = Jacobian(nK, Admittanz, V, num);         % assignment of jacobi-matrix
            JacPF = JacPF([2:nK,nK+2:end],:);
            etanm  =D.nmS*eta;
            deltaV = JacPF\(uSk([2:nK,nK+2:end],time)+etanm([2:nK,nK+2:end])-equatPF([2:nK,nK+2:end]));
            Vef = Vef+deltaV;
            V(2:end) = Vef(1:nK-1)+1j*Vef(nK:end);
        end
        mu = [real(V); imag(V)];
        [equatPF equatRe] = power_flow_equations(nK, Admittanz, V, num);        % assignment of equations
        [JacPF JacRe] = Jacobian(nK, Admittanz, V, num);            % assignment of jacobi-matrix
        equatPF = D.mS\equatPF;
        equat = [equatPF;equatRe];
        JdVdDS = JacPF([2:nK,nK+2:end],:)\D.nmS([2:nK,nK+2:end],:);
        JdhdV = [D.mS'*JacPF; JacRe];
        H=JdhdV*JdVdDS;
        MessT=Mess(:,time);
        K = Pfilterfc * H' / (H * Pfilterfc * H' + R);              % "right inverse" (the heart of the kalman filter)
        temp1 = eta;
        eta = xhatfc + K * (MessT - equat - H * (xhatfc - eta));
        varstop1 = norm(temp1-eta);
        j1 = j1+1;
    end
    % Data assimilation step (correction part of the Kalman filter)
    Pfilter{time} = (eye(nX) - K * H) * Pfilterfc;
    xhat(:,time) = eta;
    Shat([1,nK+1],time)=uSk([1,nK+1],time);
    Shat([2:nK,nK+2:2*nK],time) = uSk([2:nK,nK+2:2*nK],time) + D.nmS([2:nK,nK+2:end],:)*xhat(:,time);
    Vhat(:,time) = mu;
    disp(['Time step:' num2str(time)]);

    V_Mag(:,time) = abs(V);
    V_rad = angle(V);
    V_deg(:,time) = rad2deg(V_rad);
end
