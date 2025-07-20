data1 = [
    69.001418, 70.153080;
    4.0575, 3.3886;
    3.154094, 3.032133;
];
data24 = [
    153.810113, 133.745938;
    22.8591, 19.2318;
    5.154108, 4.562857;
];
data1000 = [
    3744.103680, 2836.886040;
    949.4722, 793.3100;
    383.858148, 390.927249;
];


figure;

tl = tiledlayout(1, 3);

nexttile; tlplot_bar(data1, "$n_\mathrm{RHS}=1$");
nexttile; tlplot_bar(data24, "$n_\mathrm{RHS}=24$");
nexttile; tlplot_bar(data1000, "$n_\mathrm{RHS}=1000$");

title(tl, "Average Execution Time (lower is better)", Interpreter="latex", FontName="Times");
set(gcf, "Color", "white");

function tlplot_bar(data, tstr)
    b = bar(data, 1, FontName="Times");
    set(gca, "XTickLabel", {"SciPy", "MATLAB", "spsolve"});
    set(gca, "YMinorTick", "on");
    set(gca, "FontName", "Times");

    title(tstr, Interpreter="latex", FontName="Times");
    ylabel("Time (ms)", Interpreter="latex", FontName="Times");
    legend(["$\mathbf{Ux}=\mathbf{b}$", "$\mathbf{Lx}=\mathbf{b}$"], Interpreter="latex");


    bardata = [b.YEndPoints];
    labels = strings(1, length(bardata));
    for k = 1:length(bardata)
        refpos = floor((k-1)/3)*3 + 1;
        labels(k) = sprintf("%.2fx", bardata(refpos)/bardata(k));
    end


    text( ...
        [b.XEndPoints], [b.YEndPoints], ...
        labels, FontName="Times", FontSize=8, ...
        HorizontalAlignment="center", VerticalAlignment="bottom" ...
    );

    grid on;
end