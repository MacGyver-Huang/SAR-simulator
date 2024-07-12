clc;
clear;
close all;

dir_RCS = "../PASSEDv4/TEST_AREA/res_Di3_AntPat/RCS";
dir_SAR = "../PASSEDv4/TEST_AREA/res_Di3_AntPat/SAR";
name = "UD04MODIFIED";
Look = 72.636719 ;%[deg]
file_PAR2 = "../PASSEDv4/TEST_AREA/res_Di3_AntPat/SAR/UD04MODIFIED_72.63_0.00_HH/p72.63_0.00_HH.slc.par";
file_SAR = "../PASSEDv4/TEST_AREA/res_Di3_AntPat/SAR/UD04MODIFIED_72.63_0.00_HH/72.63_0.00_HH_Level3_Srcmc.raw";
%ReadPreparePASSEDv4(dir_RCS, dir_SAR, name, Look, Asp, Pol, Level, Method)
%ReadSLCPar(file_par)
%ReadPASSEDv3SAR(file_SAR, par, SHOW, isFiltering, BetaRg, BetaAz)

SAR = ReadPreparePASSEDv4(dir_RCS,dir_SAR,name,Look,0,"HH",0,3);
par = ReadSLCPar(file_PAR2);
img = ReadPASSEDv3SAR(file_SAR,par);

imagesc(abs(img));
colorbar;