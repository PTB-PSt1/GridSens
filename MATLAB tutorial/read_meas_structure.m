function [num measure numfc forecast T]=read_meas_structure(filename)

[Ndleitung measure forecast base] = network_data(filename);

    if ~isempty(measure.Pk)
        T=size(measure.Pk,2)-1;
    else
        T=size(measure.Vk,2)-1;
    end

    num.Pk=measure.Pk(:,1);
    num.Qk=measure.Qk(:,1);
    num.Vk=measure.Vk(:,1);
    num.Pl=measure.Pl(:,1:2);
    num.Ql=measure.Ql(:,1:2);
    measure.Pk=measure.Pk(:,2:end)/base.MVA;
    measure.Qk=measure.Qk(:,2:end)/base.MVA;
    measure.Vk=measure.Vk(:,2:end)/base.V;
    measure.Pl=measure.Pl(:,3:end)/base.MVA;
    measure.Ql=measure.Ql(:,3:end)/base.MVA;

    numfc.Pk=forecast.Pk(:,1);
    numfc.Qk=forecast.Qk(:,1);
    forecast.Pk=forecast.Pk(:,2:end)/base.MVA;
    forecast.Qk=forecast.Qk(:,2:end)/base.MVA;

end
