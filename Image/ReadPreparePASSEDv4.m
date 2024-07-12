function [file_SAR, file_par, file_RCS, file_Src, file_Srd, file_RCM, file_Srcmc, file_Srcmc_rd, folder] = ReadPreparePASSEDv4(dir_RCS, dir_SAR, name, Look, Asp, Pol, Level, Method)

% dir_RCS = dir_RCS{1};
% dir_SAR = dir_SAR{1};
% Pol = Pol{1};

% Method = 'PO';
% Method = 'PTD';
% Method = 'TOTAL';

strMethod = '';

if(strcmp(Method,'PO'))
	strMethod = '_PO';
elseif(strcmp(Method,'PTD'))
	strMethod = '_PTD';
else
	strMethod = '';
end
		

% folder = sprintf('SAR_%s_%.2f_%.2f_%s', name, Look, Asp, Pol);
% folder = sprintf('%s_SAR_%.2f_%.2f_%s', name, Look, Asp, Pol);
folder = sprintf('%s_%.2f_%.2f_%s%s', name, Look, Asp, Pol, strMethod);
% file_SAR = sprintf('%.2f_%.2f_%s_focused.raw', Look, Asp, Pol);
file_SAR = sprintf('%.2f_%.2f_%s%s_Level%d_focused.raw', Look, Asp, Pol, strMethod, Level);
% file_SAR = sprintf('%.2f_%.2f_%s_focused.raw', Look, Asp, Pol);
file_par      = sprintf('p%.2f_%.2f_%s%s.slc.par', Look, Asp, Pol, strMethod);
file_Src      = sprintf('%.2f_%.2f_%s%s_Level%d_Src.raw', Look, Asp, Pol, strMethod, Level);
file_Srd      = sprintf('%.2f_%.2f_%s%s_Level%d_Srd.raw', Look, Asp, Pol, strMethod, Level);
file_RCM      = sprintf('%.2f_%.2f_%s%s_RCM.raw', Look, Asp, Pol, strMethod);
file_Srcmc    = sprintf('%.2f_%.2f_%s%s_Level%d_Srcmc.raw', Look, Asp, Pol, strMethod, Level);
file_Srcmc_rd = sprintf('%.2f_%.2f_%s%s_Level%d_Srcmc_rd.raw', Look, Asp, Pol, strMethod, Level);

file_SAR      = [dir_SAR folder '/' file_SAR];
file_par      = [dir_SAR folder '/' file_par];
file_Src      = [dir_SAR folder '/' file_Src];
file_Srd      = [dir_SAR folder '/' file_Srd];
file_RCM      = [dir_SAR folder '/' file_RCM];
file_Srcmc    = [dir_SAR folder '/' file_Srcmc];
file_Srcmc_rd = [dir_SAR folder '/' file_Srcmc_rd];

% file_RCS = sprintf('%s_%.2f_rcs_%s_Level%d.dat', name, Look, Pol, Level+1);
if(Level == 0)
	file_RCS = sprintf('%s_%.2f_rcs_%s%s.dat', name, Look, Pol, strMethod);
else
	file_RCS = sprintf('%s_%.2f_rcs_%s%s_Level%d.dat', name, Look, Pol, strMethod, Level);
end
file_RCS = [dir_RCS file_RCS];


end
