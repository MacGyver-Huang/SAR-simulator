function [par, R0_range, Az, R0_c] = ReadSLCPar(file_par)

fin = fopen(file_par);
str = fgetl(fin);
par.pos = zeros(3, 5);
par.vel = zeros(3, 5);
par.llh = zeros(3, 5);
while ischar(str)
	tmp = strsplit(str, ':');
	if(numel(tmp) == 2)
		tag = tmp{1};
		var = tmp{2};
		var = var(2:numel(var));
		if(strcmp(tag,'center_range_raw'))
			tmp = strsplit(var, ' ');
			par.R0c_Raw=str2double(tmp{2});
		end
		if(strcmp(tag,'range_pixels'))
			tmp = strsplit(var, ' ');
			par.Nrg=str2double(tmp{2});
		end
		if(strcmp(tag,'azimuth_pixels'))
			tmp = strsplit(var, ' ');
			par.Naz=str2double(tmp{2});
		end
		
		if(strcmp(tag,'azimuth_pixel_spacing'))
			tmp = strsplit(var, ' ');
			par.sp_az=str2double(tmp{2});
		end
		if(strcmp(tag,'azimuth_resolution'))
			tmp = strsplit(var, ' ');
			par.rho_az=str2double(tmp{2});
		end
		if(strcmp(tag,'range_pixel_spacing'))
			tmp = strsplit(var, ' ');
			par.sp_rg=str2double(tmp{2});
		end
		if(strcmp(tag,'range_resolution'))
			tmp = strsplit(var, ' ');
			par.rho_rg=str2double(tmp{2});
		end
		if(strcmp(tag,'platform_altitude'))
			tmp = strsplit(var, ' ');
			par.h=str2double(tmp{2});
		end
		if(strcmp(tag,'sensor_position_vector'))
			tmp = strsplit(var, ' ');
			tmp2 = strsplit(tmp{2}, ' ');
			par.h=str2double(tmp{4});
		end
		if(strcmp(tag,'near_range_slc'))
			tmp = strsplit(var, ' ');
			par.Rn_slant=str2double(tmp{2});
		end
		if(strcmp(tag,'far_range_slc'))
			tmp = strsplit(var, ' ');
			par.Rf_slant=str2double(tmp{2});
		end
		if(strcmp(tag,'pulse_repetition_frequency'))
			tmp = strsplit(var, ' ');
			par.PRF=str2double(tmp{2});
		end
		% State vector
		if(strcmp(tag,'number_of_state_vectors'))
			tmp = strsplit(var, ' ');
			par.Nsv=str2double(tmp{2});
		end
		if(strcmp(tag,'time_of_first_state_vector'))
			tmp = strsplit(var, ' ');
			par.t0_sv=str2double(tmp{2});
		end
		if(strcmp(tag,'state_vector_interval'))
			tmp = strsplit(var, ' ');
			par.dt_sv=str2double(tmp{2});
		end
		% StateVector series
		if(strcmp(tag,'state_vector_position_1'))
			tmp = strsplit(var, ' ');
			par.pos(:,1) = [str2double(tmp{2}), str2double(tmp{3}), str2double(tmp{4})]';
		end
		if(strcmp(tag,'state_vector_velocity_1'))
			tmp = strsplit(var, ' ');
			par.vel(:,1) = [str2double(tmp{2}), str2double(tmp{3}), str2double(tmp{4})]';
		end
		if(strcmp(tag,'state_vector_position_2'))
			tmp = strsplit(var, ' ');
			par.pos(:,2) = [str2double(tmp{2}), str2double(tmp{3}), str2double(tmp{4})]';
		end
		if(strcmp(tag,'state_vector_velocity_2'))
			tmp = strsplit(var, ' ');
			par.vel(:,2) = [str2double(tmp{2}), str2double(tmp{3}), str2double(tmp{4})]';
		end
		if(strcmp(tag,'state_vector_position_3'))
			tmp = strsplit(var, ' ');
			par.pos(:,3) = [str2double(tmp{2}), str2double(tmp{3}), str2double(tmp{4})]';
		end
		if(strcmp(tag,'state_vector_velocity_3'))
			tmp = strsplit(var, ' ');
			par.vel(:,3) = [str2double(tmp{2}), str2double(tmp{3}), str2double(tmp{4})]';
			par.Vs_mean = VecAbs( par.vel(:,3) );
		end
		if(strcmp(tag,'state_vector_position_4'))
			tmp = strsplit(var, ' ');
			par.pos(:,4) = [str2double(tmp{2}), str2double(tmp{3}), str2double(tmp{4})]';
		end
		if(strcmp(tag,'state_vector_velocity_4'))
			tmp = strsplit(var, ' ');
			par.vel(:,4) = [str2double(tmp{2}), str2double(tmp{3}), str2double(tmp{4})]';
		end
		if(strcmp(tag,'state_vector_position_5'))
			tmp = strsplit(var, ' ');
			par.pos(:,5) = [str2double(tmp{2}), str2double(tmp{3}), str2double(tmp{4})]';
		end
		if(strcmp(tag,'state_vector_velocity_5'))
			tmp = strsplit(var, ' ');
			par.vel(:,5) = [str2double(tmp{2}), str2double(tmp{3}), str2double(tmp{4})]';
		end
		% Map
		if(strcmp(tag,'earth_semi_major_axis'))
			tmp = strsplit(var, ' ');
			par.Ea = str2double(tmp{2});
		end
		if(strcmp(tag,'earth_semi_minor_axis'))
			tmp = strsplit(var, ' ');
			par.Eb = str2double(tmp{2});
		end
		if(strcmp(tag,'map_coordinate_1'))
			tmp = strsplit(var, ' ');
			par.llh(:,1) = [str2double(tmp{2}), str2double(tmp{3}), str2double(tmp{4})]';
		end
		if(strcmp(tag,'map_coordinate_2'))
			tmp = strsplit(var, ' ');
			par.llh(:,2) = [str2double(tmp{2}), str2double(tmp{3}), str2double(tmp{4})]';
		end
		if(strcmp(tag,'map_coordinate_3'))
			tmp = strsplit(var, ' ');
			par.llh(:,3) = [str2double(tmp{2}), str2double(tmp{3}), str2double(tmp{4})]';
		end
		if(strcmp(tag,'map_coordinate_4'))
			tmp = strsplit(var, ' ');
			par.llh(:,4) = [str2double(tmp{2}), str2double(tmp{3}), str2double(tmp{4})]';
		end
		if(strcmp(tag,'map_coordinate_5'))
			tmp = strsplit(var, ' ');
			par.llh(:,5) = [str2double(tmp{2}), str2double(tmp{3}), str2double(tmp{4})]';
		end
	end
	str = fgetl(fin);
end
fclose(fin);

R0_range = linspace(par.Rn_slant, par.Rf_slant, par.Nrg);
R0_c = mean(R0_range);
R0_range = R0_range - R0_c;
Az = 0:1:par.Naz-1;
Az = Az * par.Vs_mean*(1/par.PRF);
Az = Az - mean(Az);

	
end