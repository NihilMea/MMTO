using MMTO

h = 100.0
l = 400.0
d = 10.0
fea = FEAProblem(l, h, d, d, 1.0);
set_bc!(fea, :load, -1.0, :v, Point2D(l, h / 2))
set_bc!(fea, :disp, 0.0, [:u, :v], Point2D(0.0, 0.0), Point2D(0.0, h))
filt = Filter(2*d, fea)
# set_bc!(fea, :disp, 0.0, :v, Point2D(l, 0.0))

mmtop = MMTOProblem(fea, [1.0], 3.0)
# set_region!(mmtop, :passive, Point2D(l / 3, 0.0), Point2D(2*l/3, h/3))
# sol, x = solve(mmtop, filt, 0.5, 0.4, 1e-6, 0.1, 100)

sol, x = solve(mmtop, filt, 0.5, 0.4, 1e-9, 1.0, 100)
viz(fea, x, colorbar=true, colornum=12, colormap=:grayC)

