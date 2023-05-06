using MMTO

l = 100.0
d = 1.0
eps = 1e-3

fea = FEAProblem(l, l, d, d, 1.0);
set_bc!(fea, :load, -1000.0, :v,  Point2D(l, 2l / 5))
set_bc!(fea, :disp, 0.0, [:u, :v], Point2D(0.0 - eps, l - eps), Point2D(2 * l / 5 + eps, l + eps))
filt = Filter(2d, fea)

mmtop = MMTOProblem(fea, [2.1e5, 0.71e5], 3.0)
set_region!(mmtop, :passive, Point2D(2l / 5, 2l / 5), Point2D(l, l))
set_region!(mmtop, :fixed, Point2D(l - 6.0 , 2l / 5 - 4.0), Point2D(l, 2l / 5))

sol_compl, x_compl = solve(mmtop, :Compl_min, :Volume, filt, false, 0.5, [0.1, 0.2], [7.2e-6, 2.8e-6], [600.0, 350.0], 300);
fig1 = display_solution(:Density, sol_compl, mmtop, x_compl, mat_names=["Сталь", "Алюминий", "Пустота"])
fig2 = display_solution(:VM_Stress, sol_compl, mmtop, x_compl)

sol_stress, x_stress = solve(mmtop, :Mass_min, :Stress, filt, false, 0.5, [0.1, 0.2], [7.2e-6, 2.8e-6], [600.0, 350.0], 300);
fig3 = display_solution(:Density, sol_stress, mmtop, x_stress, mat_names=["Сталь", "Алюминий", "Пустота"])
fig4 = display_solution(:VM_Stress, sol_stress, mmtop, x_stress)

sol_both, x_both = solve(mmtop, :Compl_min, [:Stress,:Volume], filt, false, 0.5, [0.1, 0.2], [7.2e-6, 2.8e-6], [600.0, 350.0], 300);
fig7 = display_solution(:Density, sol_both, mmtop, x_both, mat_names=["Сталь", "Алюминий", "Пустота"])
fig8 = display_solution(:VM_Stress, sol_both, mmtop, x_both)

fea1 = FEAProblem(l, l, d, d, 1.0);
set_bc!(fea1, :load, -500.0, :v, Point2D(l, 2l / 5))
set_bc!(fea1, :disp, 0.0, [:u, :v], Point2D(0.0 - eps, l - eps), Point2D(2 * l / 5 + eps, l + eps))
filt1 = Filter(2d, fea1)

mmtop_single = MMTOProblem(fea1, [0.71e5], 3.0)
set_region!(mmtop_single, :passive, Point2D(2l / 5, 2l / 5), Point2D(l, l))
set_region!(mmtop_single, :fixed, Point2D(l - 6.0, 2l / 5 - 4.0), Point2D(l, 2l / 5))

sol_single, x_single = solve(mmtop_single, :Mass_min, :Stress, filt1, false, 0.5, [0.3], [2.8e-6], [350.0], 300);
fig5 = display_solution(:Density, sol_single, mmtop_single, x_single, mat_names=["Алюминий", "Пустота"])
fig6 = display_solution(:VM_Stress, sol_single, mmtop_single, x_single)
