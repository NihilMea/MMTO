using MMTO

l = 100.0
d = 1.0
fea = FEAProblem(l, l, d, d, 1.0);
set_bc!(fea, :load, -500.0/4, :v, Point2D(l-3d, 2 * l / 5),Point2D(l, 2 * l / 5))
set_bc!(fea, :disp, 0.0, [:u, :v], Point2D(0.0, l), Point2D(2 * l / 5, l))
filt = Filter(1.5d, fea)


mmtop = MMTOProblem(fea, [0.71e5], 3.0)
set_region!(mmtop, :passive, Point2D(2l / 5, 2l / 5), Point2D(l, l))
set_region!(mmtop, :fixed, Point2D(l - 3d, 2l / 5 - 2d), Point2D(l, 2l / 5))
sol, x = solve(mmtop, filt, 0.5, 0.4, 2.8e-6, 350.0, 300)
σ = calc_stress(fea, sol, :vM)
σ[mmtop.fixed_elements] .= 0.0
viz(fea, σ, colorbar=true, colornum=12)
viz(fea, x, colorbar=true, colornum=12, colormap=:grayC)