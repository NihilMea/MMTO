using MMTO


l = 100.0
d = 5.0
fea = FEAProblem(3l, l, d, d, 1.0);
set_bc!(fea, :load, -1000.0, :v, Point2D(0.0, l))
set_bc!(fea, :disp, 0.0, :u, Point2D(0.0, 0.0), Point2D(0.0, l))
set_bc!(fea, :disp, 0.0, :v, Point2D(3l, 0.0))
filt = Filter(4d, fea)

mmtop = MMTOProblem(fea, [0.71e5], 3.0, 0.5, 24.0)
set_region!(mmtop, :fixed, Point2D(0.0, l - 3.0), Point2D(8.0, l))
set_region!(mmtop, :fixed, Point2D(3l - 4.0, 0.0), Point2D(3l, 3.0))

sol, x, it = solve(mmtop, :Mass_min, :Stress, filt, false, 0.9, [0.1,0.4], [2.8e-6], [ 350.0], 600)

display_solution(:Density, sol, mmtop, x; mat_names=["титан","алюминий", "пустота"])
display_solution(:VM_Stress, sol, mmtop, x)

