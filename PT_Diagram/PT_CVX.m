% Phrase-Transition Diagram for CVX recovery

input_file_name = '<file_name>';
image_path = strcat('<home_directory>',input_file_name,'.mat');
results_path = '<results_directory>';

% Load input image

load(image_path);
file_desc = whos('-file', image_path);
x = eval(file_desc.name);
var_erase = {file_desc.name,'var_erase','file_desc'};
clear(var_erase{:});
res = size(x);
if res(1) == res(2)
    N = res(1)*res(2);
else
    return;
end

% Phase-Transition diagram parameters

M = 100; % No. of problem instances
v = 30; % Dimension of the grid = v*v
tol = 0.1; % Tolerance

% Print settings

fprintf('Image path = %s\n', image_path);
fprintf('Grid size = %d\n', v);
fprintf('No. trials/problem = %d\n', M);
fprintf('Error tolerance = %d\n', tol);

% Initialization

start_index = 0.1;
step_size = 1/v;
stop_index = 1;
prob =zeros(v-1,v-1); %probability grid
nrm_max = zeros(v-1,v-1);
nrm_min = zeros(v-1,v-1);
snr_max = zeros(v-1,v-1);
snr_min = zeros(v-1,v-1);
snr_avg = zeros(v-1,v-1);

% Compute Phase-Transition graph

i=0;
for delta = start_index:step_size:stop_index
    i = i+1;
    j=0;
    for rho = start_index:step_size:stop_index
        j=j+1;
        n = round(delta*N);
        k = round(rho*n);        
		p = zeros(M,1);
		nrm = zeros(M,1);
		snr = zeros(M,1);
		
		parfor m = 1:M
			[nrm(m), snr(m)] = cvxWrapper(x, N, res, n, k);
			if(nrm(m) <= tol)
				p(m)=1;
			end
		end

		prob(j,i) = sum(p)/M*100;
		nrm_max(j,i) = max(nrm);
		nrm_min(j,i) = min(nrm);
		snr_max(j,i) = max(snr);
		snr_min(j,i) = min(snr);
		snr_avg(j,i) = mean(snr);
		
    end
end

% Store results

results_file_name = strcat(input_file_name, '_cvx_', num2str(v),'_',num2str(M),'.mat');
save(strcat(results_path,results_file_name));
fprintf('Results saved at %s\n', strcat(results_path, results_file_name));

function [nrm, snr] = cvxWrapper(x, N, res, n, k)

    % Generating Problem Instance
    
    A = dftmtx(sqrt(N));
    x0 = genSparseMatrix(x, N, k);
    p = randperm(N,n);
    y = transform(A,x0,p);
    opTV = linop_TV(size(x0), 'circular', 'cvx');

    % CVX Reconstruction
    
    x_rec = cvxTVReconstruction(A,y,N,p, opTV);

    % Check Success
    
    nrm = norm((x0-x_rec),2)/norm(x0,2);
    snr = -20*log10(nrm);
    
end

function x = cvxReconstruction(A,y,N,p)
            cvx_begin
            variable x(sqrt(N),sqrt(N))
            minimize(norm(x(:),1))
            subject to 
            transform(A,x,p)==y
            cvx_end
end

function x = cvxTVReconstruction(A,y,N,p, opTV)
            cvx_begin
            variable x(sqrt(N),sqrt(N))
            minimize(opTV(x));
            subject to 
            transform(A,x,p)==y
            cvx_end
end

function x = genSparseMatrix(x, N, k)
	indices = randperm(N,k);
	y = x(indices);
    x = zeros(sqrt(N),sqrt(N));
    x(indices) = y;
end

function z = transform(A,x,p)
z = A*x*A';
z = z(p);
end