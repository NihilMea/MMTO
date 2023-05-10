using MMTO

l = 100.0
d = 1.0
eps = 1e-3

fea = FEAProblem(l, l, d, d, 1.0);
set_bc!(fea, :load, -1000.0, :v, Point2D(l, 2l / 5))
set_bc!(fea, :disp, 0.0, [:u, :v], Point2D(0.0, l), Point2D(2 * l / 5, l))
filt = Filter(4d, fea)

mmtop = MMTOProblem(fea, [0.71e5], 3.0,0.5, 24.0)
set_region!(mmtop, :passive, Point2D(2l / 5, 2l / 5), Point2D(l, l))
set_region!(mmtop, :fixed, Point2D(l - 6.0, 2l / 5 - 4.0), Point2D(l, 2l / 5))

sol, x = solve(mmtop, :Mass_min, :Stress, filt, false, 0.5, [0.4], [2.8e-6], [350.0], 600)

display_solution(:Density, sol, mmtop, x; mat_names=["алюминий", "пустота"])
display_solution(:VM_Stress, sol, mmtop, x)