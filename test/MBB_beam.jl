using MMTO


l = 40.0
d=1.0
fea = FEAProblem(3l, l, d, d, 1.0);
set_bc!(fea, :load, -500.0, :v, Point2D(0.0, l))
set_bc!(fea, :disp, 0.0, :u, Point2D(0.0, 0.0), Point2D(0.0, l))
set_bc!(fea, :disp, 0.0, :v, Point2D(3l, 0.0))
filt = Filter(1.5, fea)

mmtop = MMTOProblem(fea, [0.71e5], 3.0)
# set_region!(mmtop, :passive, Point2D(l / 3, 0.0), Point2D(2*l/3, h/3))
sol, x = solve(mmtop, filt, 0.5, 0.4, 2.8e-6, 350.0, 200)
# @run sol, x = solve(mmtop, filt, 0.5, 0.4, 2.8e-6, 350.0, 100)

viz(fea, x, colorbar=true, colornum=12, colormap=:grayC)
