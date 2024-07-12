function sar = ReadPASSEDv3SAR(file_SAR, par, SHOW, isFiltering, BetaRg, BetaAz)

% Default
if(~exist('SHOW','var')); SHOW=false; end
if(~exist('isFiltering','var')); isFiltering=false; end
if(~exist('BetaRg','var')); BetaRg=5.2; end
if(~exist('BetaAz','var')); BetaAz=2.1; end

data = multibandread(file_SAR, [par.Naz, par.Nrg*2, 1], 'double', 0, 'bip', 'ieee-le');
% data = multibandread(file_SAR, [par.Naz, par.Nrg*2, 1], 'single', 0, 'bip', 'ieee-le');

idx_odd  = (1:2:par.Nrg*2);	% real
idx_even = (2:2:par.Nrg*2);	% imag

sar = complex(data(:,idx_odd), data(:,idx_even));

if(isFiltering)
	%% Range filtering for SAR
	% Slant range
	sz = size(sar);
	for jj=1:sz(1)
		% Make window
		win = fftshift(kaiser(sz(2), BetaRg))';
		% multiplcation
		sar(jj,:) = ifft(fft(sar(jj,:)) .* win);
	end
	% Azimuth
	sz = size(sar);
	for ii=1:sz(2)
		% Make window
		win = fftshift(kaiser(sz(1), BetaAz));
		% multiplcation
		sar(:,ii) = ifft(fft(sar(:,ii)) .* win);
	end
end

if(SHOW)
	figure; imagesc(abs(sar));
	set(gca, 'YDir', 'normal'); colorbar;
	axis tight;
	title(sprintf('SAR'));	xlabel('Slant range frequency [sample] -->'); ylabel('Azimuth [sample] -->');
end

end
