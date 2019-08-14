function [Ndleitung measure forecast base] = network_data(filename)

    Ndleitung=xlsread(filename,1);
    measure.Pk=xlsread(filename,2);
    measure.Qk=xlsread(filename,3);
    measure.Vk=xlsread(filename,4);
    measure.Pl=xlsread(filename,5);
    measure.Ql=xlsread(filename,6);
    X=xlsread(filename,7);
    base.V=X(1);
    base.MVA=X(2);

    forecast.Pk=xlsread(filename,8);
    forecast.Qk=xlsread(filename,9);

end
