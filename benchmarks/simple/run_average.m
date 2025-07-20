n_ave = 1000;

size = 10000;
nrhs = 24;
density = 0.1;

rng(0);

A = spdiags(5+rand(size, 1), 0, size, size) + sprand(size, size, density);
b = A * (rand(size, nrhs));

spA = triu(A);
[i,j,v] = find(spA);
spA = sparse(i,j,v);
disp("Upper");
t_start = tic;
for k = 1:n_ave
    x = spA \ b;
end
t_elapsed = toc(t_start) / n_ave;
disp(1000*t_elapsed);

spA = tril(A);
[i,j,v] = find(spA);
spA = sparse(i,j,v);
disp("Lower");
t_start = tic;
for k = 1:n_ave
    x = spA \ b;
end
t_elapsed = toc(t_start) / n_ave;
disp(1000*t_elapsed);