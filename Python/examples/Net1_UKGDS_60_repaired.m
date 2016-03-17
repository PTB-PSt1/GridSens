function mpc = Net1_UKGDS_60;
% File generated by psse2matpower from a PSS/E file.
%-----------------------------------------------------------------------
% Author:   Federico Milano
% E-mail:   Federico.Milano@uclm.es
% Author:   Juan Carlos Morataya
% E-mail:   jc.morataya@ieee.org
% Author:   Ray Zimmerman
% E-mail:   rz10@cornell.edu
% *** This filter is protected under the GPL terms ***
%-----------------------------------------------------------------------
% PSS/E Data Format File : StrathGridSimplified_30.raw
% 
% Commented lines are disconnected branches
% Conversion completed assuming PSS/E V29 data format.
%-----------------------------------------------------------------------
%% MATPOWER Case Format : Version 2;
mpc.version = '2';

%%-----  Power Flow Data  -----%%
%% system MVA base
mpc.baseMVA = 100;

%% bus data
%	bus_i	type	Pd	Qd	Gs	Bs	area	Vm	Va	baseKV	zone	Vmax	Vmin
mpc.bus = [
301	3	0	0	0	0	1	3.06	0	33	1	1.03	0.97;
1100	1	0	0	0	0	1	1	0	11	1	1.03	0.97;
1101	1	0.392	0.0784	0	0	1	1	0	11	1	1.03	0.97;
1102	1	0.392	0.0784	0	0	1	1	0	11	1	1.03	0.97;
1103	1	0.116	0.0232	0	0	1	1	0	11	1	1.03	0.97;
1104	1	0.392	0.0784	0	0	1	1	0	11	1	1.03	0.97;
1105	1	0.392	0.0784	0	0	1	1	0	11	1	1.03	0.97;
1106	1	0.116	0.0232	0	0	1	1	0	11	1	1.03	0.97;
1107	1	0.392	0.0784	0	0	1	1	0	11	1	1.03	0.97;
1108	1	0.392	0.0784	0	0	1	1	0	11	1	1.03	0.97;
1109	1	0.116	0.0232	0	0	1	1	0	11	1	1.03	0.97;
1110	1	0.394	0.0788	0	0	1	1	0	11	1	1.03	0.97;
1111	1	0.394	0.0788	0	0	1	1	0	11	1	1.03	0.97;
1112	1	0.396	0.0792	0	0	1	1	0	11	1	1.03	0.97;
1113	1	0.1	0.02	0	0	1	1	0	11	1	1.03	0.97;
1114	1	0.102	0.0204	0	0	1	1	0	11	1	1.03	0.97;
1115	1	0.426	0.0852	0	0	1	1	0	11	1	1.03	0.97;
1116	1	0.426	0.0852	0	0	1	1	0	11	1	1.03	0.97;
1117	1	0.426	0.0852	0	0	1	1	0	11	1	1.03	0.97;
1118	1	0.426	0.0852	0	0	1	1	0	11	1	1.03	0.97;
1119	1	0.426	0.0852	0	0	1	1	0	11	1	1.03	0.97;
1120	1	0.426	0.0852	0	0	1	1	0	11	1	1.03	0.97;
1121	1	0.426	0.0852	0	0	1	1	0	11	1	1.03	0.97;
1122	1	0.212	0.0424	0	0	1	1	0	11	1	1.03	0.97;
1123	1	0.212	0.0424	0	0	1	1	0	11	1	1.03	0.97;
1124	1	0.214	0.0428	0	0	1	1	0	11	1	1.03	0.97;
1125	1	0.214	0.0428	0	0	1	1	0	11	1	1.03	0.97;
1126	1	0.426	0.0852	0	0	1	1	0	11	1	1.03	0.97;
1127	1	0.426	0.0852	0	0	1	1	0	11	1	1.03	0.97;
1128	1	0.426	0.0852	0	0	1	1	0	11	1	1.03	0.97;
1129	1	0.426	0.0852	0	0	1	1	0	11	1	1.03	0.97;
1130	1	0.426	0.0852	0	0	1	1	0	11	1	1.03	0.97;
1131	1	0.426	0.0852	0	0	1	1	0	11	1	1.03	0.97;
1132	1	0.426	0.0852	0	0	1	1	0	11	1	1.03	0.97;
1133	1	0.212	0.0424	0	0	1	1	0	11	1	1.03	0.97;
1134	1	0.212	0.0424	0	0	1	1	0	11	1	1.03	0.97;
1135	1	0.214	0.0428	0	0	1	1	0	11	1	1.03	0.97;
1136	1	0.214	0.0428	0	0	1	1	0	11	1	1.03	0.97;
1137	1	0.434	0.0868	0	0	1	1	0	11	1	1.03	0.97;
1138	1	0.434	0.0868	0	0	1	1	0	11	1	1.03	0.97;
1139	1	0.436	0.0872	0	0	1	1	0	11	1	1.03	0.97;
1140	1	0.436	0.0872	0	0	1	1	0	11	1	1.03	0.97;
1141	1	0.436	0.0872	0	0	1	1	0	11	1	1.03	0.97;
1142	1	0.436	0.0872	0	0	1	1	0	11	1	1.03	0.97;
1143	1	0.436	0.0872	0	0	1	1	0	11	1	1.03	0.97;
1144	1	0.436	0.0872	0	0	1	1	0	11	1	1.03	0.97;
1145	1	0.436	0.0872	0	0	1	1	0	11	1	1.03	0.97;
1146	1	0.216	0.0432	0	0	1	1	0	11	1	1.03	0.97;
1147	1	0.218	0.0436	0	0	1	1	0	11	1	1.03	0.97;
1148	1	0.218	0.0436	0	0	1	1	0	11	1	1.03	0.97;
1149	1	0.218	0.0436	0	0	1	1	0	11	1	1.03	0.97;
1150	1	0.218	0.0436	0	0	1	1	0	11	1	1.03	0.97;
1151	1	0.342	0.0684	0	0	1	1	0	11	1	1.03	0.97;
1152	1	0.342	0.0684	0	0	1	1	0	11	1	1.03	0.97;
1153	1	0.344	0.0688	0	0	1	1	0	11	1	1.03	0.97;
1154	1	0.344	0.0688	0	0	1	1	0	11	1	1.03	0.97;
1155	1	0.344	0.0688	0	0	1	1	0	11	1	1.03	0.97;
1156	1	0.344	0.0688	0	0	1	1	0	11	1	1.03	0.97;
1157	1	0.344	0.0688	0	0	1	1	0	11	1	1.03	0.97;
1158	1	0.344	0.0688	0	0	1	1	0	11	1	1.03	0.97;
1159	1	0.344	0.0688	0	0	1	1	0	11	1	1.03	0.97;
1160	1	0.344	0.0688	0	0	1	1	0	11	1	1.03	0.97;
1161	1	0.344	0.0688	0	0	1	1	0	11	1	1.03	0.97;
1162	1	0.344	0.0688	0	0	1	1	0	11	1	1.03	0.97;
1163	1	0.344	0.0688	0	0	1	1	0	11	1	1.03	0.97;
1164	1	0.344	0.0688	0	0	1	1	0	11	1	1.03	0.97;
1165	1	0.344	0.0688	0	0	1	1	0	11	1	1.03	0.97;
1166	1	0.344	0.0688	0	0	1	1	0	11	1	1.03	0.97;
1167	1	0.222	0.0444	0	0	1	1	0	11	1	1.03	0.97;
1168	1	0.222	0.0444	0	0	1	1	0	11	1	1.03	0.97;
1169	1	0.224	0.0448	0	0	1	1	0	11	1	1.03	0.97;
1170	1	0.224	0.0448	0	0	1	1	0	11	1	1.03	0.97;
1171	1	0.224	0.0448	0	0	1	1	0	11	1	1.03	0.97;
1172	1	0.224	0.0448	0	0	1	1	0	11	1	1.03	0.97;
1173	1	0.224	0.0448	0	0	1	1	0	11	1	1.03	0.97;
1174	1	0.224	0.0448	0	0	1	1	0	11	1	1.03	0.97;
1175	1	0.224	0.0448	0	0	1	1	0	11	1	1.03	0.97;

];

%% generator data
%	bus	Pg	Qg	Qmax	Qmin	Vg	mBase	status	Pmax	Pmin	Pc1	Pc2	Qc1min	Qc1max	Qc2min	Qc2max	ramp_agc	ramp_10	ramp_30	ramp_q	apf
mpc.gen = [
301 	30      10      60	-60	3.06	100	1	60	-60	0	0	0	0	0	0	0	0	0	0	0;
% 1102	1.73	0.81	1.1	0.4	1	2.3	0	2.3	0	0	0	0	0	0	0	0	0	0	0	0;
% 1105	1.73	0.81	1.1	0.4	1	2.3	0	2.3	0	0	0	0	0	0	0	0	0	0	0	0;
% 1108	1.73	0.81	1.1	0.4	1	2.3	0	2.3	0	0	0	0	0	0	0	0	0	0	0	0;
% 1112	1.73	0.81	1.1	0.4	1	2.3	0	2.3	0	0	0	0	0	0	0	0	0	0	0	0;
% 1116	1.73	0.81	1.1	0.4	1	2.3	0	2.3	0	0	0	0	0	0	0	0	0	0	0	0;
% 1118	1.73	0.81	1.1	0.4	1	2.3	0	2.3	0	0	0	0	0	0	0	0	0	0	0	0;
1119	1.73	0.81	1.1	-1.0	1	0	1	10	0	0	0	0	0	0	0	0	0	0	0	0;
% 1121	1.73	0.81	1.1	0.4	1	2.3	0	2.3	0	0	0	0	0	0	0	0	0	0	0	0;
% 1127	1.73	0.81	1.1	0.4	1	2.3	0	2.3	0	0	0	0	0	0	0	0	0	0	0	0;
% 1129	1.73	0.81	1.1	0.4	1	2.3	0	2.3	0	0	0	0	0	0	0	0	0	0	0	0;
% 1130	1.73	0.81	1.1	0.4	1	2.3	0	2.3	0	0	0	0	0	0	0	0	0	0	0	0;
% 1132	1.73	0.81	1.1	0.4	1	2.3	0	2.3	0	0	0	0	0	0	0	0	0	0	0	0;
% 1138	1.73	0.81	1.1	0.4	1	2.3	0	2.3	0	0	0	0	0	0	0	0	0	0	0	0;
% 1140	1.73	0.81	1.1	0.4	1	2.3	0	2.3	0	0	0	0	0	0	0	0	0	0	0	0;
% 1141	1.73	0.81	1.1	0.4	1	2.3	0	2.3	0	0	0	0	0	0	0	0	0	0	0	0;
% 1143	1.73	0.81	1.1	0.4	1	2.3	0	2.3	0	0	0	0	0	0	0	0	0	0	0	0;
% 1145	1.73	0.81	1.1	0.4	1	2.3	0	2.3	0	0	0	0	0	0	0	0	0	0	0	0;
% 1152	1.73	0.81	1.1	0.4	1	2.3	0	2.3	0	0	0	0	0	0	0	0	0	0	0	0;
% 1155	1.73	0.81	1.1	0.4	1	2.3	0	2.3	0	0	0	0	0	0	0	0	0	0	0	0;
% 1159	1.73	0.81	1.1	0.4	1	2.3	0	2.3	0	0	0	0	0	0	0	0	0	0	0	0;
% 1162	1.73	0.81	1.1	0.4	1	2.3	0	2.3	0	0	0	0	0	0	0	0	0	0	0	0;
% 1166	1.73	0.81	1.1	0.4	1	2.3	0	2.3	0	0	0	0	0	0	0	0	0	0	0	0;

];

%% branch data
%	fbus	tbus	r	x	b	rateA	rateB	rateC	ratio	angle	status	angmin	angmax
mpc.branch = [
301     1100	0.04707	0.65409	0	0	0	0	3	-30	1	-360	360;
1100	1101	0.2038	0.1056	0	6.82	6.82	6.82	1	0	1	-360	360;
1101	1102	0.2038	0.1056	0	6.82	6.82	6.82	1	0	1	-360	360;
1102	1103	0.0624	0.017	0	4.84	4.84	4.84	1	0	1	-360	360;
1100	1104	0.2038	0.1056	0	6.82	6.82	6.82	1	0	1	-360	360;
1104	1105	0.2038	0.1056	0	6.82	6.82	6.82	1	0	1	-360	360;
1105	1106	0.0624	0.017	0	4.84	4.84	4.84	1	0	1	-360	360;
1100	1107	0.2038	0.1056	0	6.82	6.82	6.82	1	0	1	-360	360;
1107	1108	0.2038	0.1056	0	6.82	6.82	6.82	1	0	1	-360	360;
1108	1109	0.0624	0.017	0	4.84	4.84	4.84	1	0	1	-360	360;
1100	1110	0.266	0.1378	0	6.82	6.82	6.82	1	0	1	-360	360;
1110	1111	0.266	0.1378	0	6.82	6.82	6.82	1	0	1	-360	360;
1111	1112	0.266	0.1378	0	6.82	6.82	6.82	1	0	1	-360	360;
1111	1113	0.0663	0.018	0	4.84	4.84	4.84	1	0	1	-360	360;
1112	1114	0.0663	0.018	0	4.84	4.84	4.84	1	0	1	-360	360;
1100	1115	0.0745	0.0574	0	8.86	8.86	8.86	1	0	1	-360	360;
1115	1116	0.0745	0.0574	0	8.86	8.86	8.86	1	0	1	-360	360;
1116	1117	0.0745	0.0574	0	8.86	8.86	8.86	1	0	1	-360	360;
1117	1118	0.0745	0.0574	0	8.86	8.86	8.86	1	0	1	-360	360;
1118	1119	0.0745	0.0574	0	8.86	8.86	8.86	1	0	1	-360	360;
1119	1120	0.0745	0.0574	0	8.86	8.86	8.86	1	0	1	-360	360;
1120	1121	0.0745	0.0574	0	8.86	8.86	8.86	1	0	1	-360	360;
1116	1122	0.0542	0.0147	0	4.84	4.84	4.84	1	0	1	-360	360;
1118	1123	0.0542	0.0147	0	4.84	4.84	4.84	1	0	1	-360	360;
1119	1124	0.0542	0.0147	0	4.84	4.84	4.84	1	0	1	-360	360;
1121	1125	0.0542	0.0147	0	4.84	4.84	4.84	1	0	1	-360	360;
1100	1126	0.0745	0.0574	0	8.86	8.86	8.86	1	0	1	-360	360;
1126	1127	0.0745	0.0574	0	8.86	8.86	8.86	1	0	1	-360	360;
1127	1128	0.0745	0.0574	0	8.86	8.86	8.86	1	0	1	-360	360;
1128	1129	0.0745	0.0574	0	8.86	8.86	8.86	1	0	1	-360	360;
1129	1130	0.0745	0.0574	0	8.86	8.86	8.86	1	0	1	-360	360;
1130	1131	0.0745	0.0574	0	8.86	8.86	8.86	1	0	1	-360	360;
1131	1132	0.0745	0.0574	0	8.86	8.86	8.86	1	0	1	-360	360;
1127	1133	0.0542	0.0147	0	4.84	4.84	4.84	1	0	1	-360	360;
1129	1134	0.0542	0.0147	0	4.84	4.84	4.84	1	0	1	-360	360;
1130	1135	0.0542	0.0147	0	4.84	4.84	4.84	1	0	1	-360	360;
1132	1136	0.0542	0.0147	0	4.84	4.84	4.84	1	0	1	-360	360;
1100	1137	0.0917	0.0706	0	8.86	8.86	8.86	1	0	1	-360	360;
1137	1138	0.0917	0.0706	0	8.86	8.86	8.86	1	0	1	-360	360;
1138	1139	0.0917	0.0706	0	8.86	8.86	8.86	1	0	1	-360	360;
1139	1140	0.0917	0.0706	0	8.86	8.86	8.86	1	0	1	-360	360;
1140	1141	0.0917	0.0706	0	8.86	8.86	8.86	1	0	1	-360	360;
1141	1142	0.0917	0.0706	0	8.86	8.86	8.86	1	0	1	-360	360;
1142	1143	0.0917	0.0706	0	8.86	8.86	8.86	1	0	1	-360	360;
1143	1144	0.0917	0.0706	0	8.86	8.86	8.86	1	0	1	-360	360;
1144	1145	0.0917	0.0706	0	8.86	8.86	8.86	1	0	1	-360	360;
1138	1146	0.0571	0.0155	0	4.84	4.84	4.84	1	0	1	-360	360;
1140	1147	0.0571	0.0155	0	4.84	4.84	4.84	1	0	1	-360	360;
1141	1148	0.0571	0.0155	0	4.84	4.84	4.84	1	0	1	-360	360;
1143	1149	0.0571	0.0155	0	4.84	4.84	4.84	1	0	1	-360	360;
1145	1150	0.0571	0.0155	0	4.84	4.84	4.84	1	0	1	-360	360;
1100	1151	0.0665	0.0512	0	8.86	8.86	8.86	1	0	1	-360	360;
1151	1152	0.0665	0.0512	0	8.86	8.86	8.86	1	0	1	-360	360;
1152	1153	0.0665	0.0512	0	8.86	8.86	8.86	1	0	1	-360	360;
1153	1154	0.0665	0.0512	0	8.86	8.86	8.86	1	0	1	-360	360;
1154	1155	0.0665	0.0512	0	8.86	8.86	8.86	1	0	1	-360	360;
1155	1156	0.0665	0.0512	0	8.86	8.86	8.86	1	0	1	-360	360;
1156	1157	0.0665	0.0512	0	8.86	8.86	8.86	1	0	1	-360	360;
1157	1158	0.0665	0.0512	0	8.86	8.86	8.86	1	0	1	-360	360;
1158	1159	0.0665	0.0512	0	8.86	8.86	8.86	1	0	1	-360	360;
1159	1160	0.0665	0.0512	0	8.86	8.86	8.86	1	0	1	-360	360;
1160	1161	0.0665	0.0512	0	8.86	8.86	8.86	1	0	1	-360	360;
1161	1162	0.0665	0.0512	0	8.86	8.86	8.86	1	0	1	-360	360;
1162	1163	0.0665	0.0512	0	8.86	8.86	8.86	1	0	1	-360	360;
1163	1164	0.0665	0.0512	0	8.86	8.86	8.86	1	0	1	-360	360;
1164	1165	0.0665	0.0512	0	8.86	8.86	8.86	1	0	1	-360	360;
1165	1166	0.0665	0.0512	0	8.86	8.86	8.86	1	0	1	-360	360;
1152	1167	0.0729	0.0198	0	4.84	4.84	4.84	1	0	1	-360	360;
1154	1168	0.0729	0.0198	0	4.84	4.84	4.84	1	0	1	-360	360;
1155	1169	0.0729	0.0198	0	4.84	4.84	4.84	1	0	1	-360	360;
1157	1170	0.0729	0.0198	0	4.84	4.84	4.84	1	0	1	-360	360;
1159	1171	0.0729	0.0198	0	4.84	4.84	4.84	1	0	1	-360	360;
1161	1172	0.0729	0.0198	0	4.84	4.84	4.84	1	0	1	-360	360;
1162	1173	0.0729	0.0198	0	4.84	4.84	4.84	1	0	1	-360	360;
1164	1174	0.0729	0.0198	0	4.84	4.84	4.84	1	0	1	-360	360;
1166	1175	0.0729	0.0198	0	4.84	4.84	4.84	1	0	1	-360	360;
];
		

%%-----  OPF Data  -----%%
%% generator cost data
%	1	startup	shutdown	n	x1	y1	...	xn	yn
% %	2	startup	shutdown	n	c(n-1)	...	c0;
% mpc.gencost = [
% 	2	0	0	2	1	0;
% 	%2	0	0	2	1	0;
% 	%2	0	0	2	1	0;
% ];

%% bus names
mpc.busnames = [
	'SUPPLY_D'
	'DG1_SWBD'
	%'DYNAMIC '
	'DG2_SWBD'
	%'DYNAMIC '
	'DG1_SWBX'
	'DG2_SWBX'
];
