using MMTO 

l = 100.0
d = 1.0
eps=1e-3
fea = FEAProblem(l, l, d, d, 1.0);
set_bc!(fea, :load, -500.0/4.0, :v, Point2D(l-3d-eps, 2 * l / 5-eps),Point2D(l+eps,2l/5+eps))
set_bc!(fea, :disp, 0.0, [:u, :v], Point2D(0.0-eps, l-eps), Point2D(2 * l / 5+eps, l+eps))
filt = Filter(1.5d, fea)

mmtop = MMTOProblem(fea, [0.7e5], 3.0)
set_region!(mmtop, :passive, Point2D(2l / 5, 2l / 5), Point2D(l, l))
set_region!(mmtop, :fixed, Point2D(l - 3d, 2l / 5 - 2d), Point2D(l, 2l / 5))

sol, x = solve(mmtop, :Mass_min, :Stress, filt, false, 0.5, [0.3], [2.8e-6, 1e-12], [350.0], 300);

sol, x = solve(mmtop, :Compl_min, [:Volume,:Stress], filt, false, 0.5, [0.4], [2.8e-6, 1e-12], [350.0], 300);

val, w = calc_mat_type(mmtop, x);
σ = calc_stress(fea, sol, :vM);
σ[val.<0.5] .= NaN;
σ[mmtop.fixed_elements] .= NaN;
viz(fea, σ, colorbar=true, colornum=12)
viz(fea, val, colorbar=true, colornum=3, colormap=:RdPu_9)
