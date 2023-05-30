using MMTO

l = 100.0
d = 1.0

fea = FEAProblem(l, l, d, d, 1.0);
set_bc!(fea, :load, -500.0 / 3, :v, Point2D(l - 2d, 2l / 5), Point2D(l, 2l / 5))
set_bc!(fea, :disp, 0.0, [:u, :v], Point2D(0.0, l), Point2D(2 * l / 5, l))
filt = Filter(4d, fea)

mats = [Material("сталь", 2.1e5, 7.2e-6, 700.0, 0.2)]
# push!(mats,Material("титан", 1.1e5, 4.5e-6, 500.0, 0.1))
# push!(mats,Material("алюминий", 0.71e5, 2.8e-6, 350.0, 0.1))

mmtop = MMTOProblem(fea, [mat.E for mat in mats], 3.0, 0.5, 24.0)
set_region!(mmtop, :passive, Point2D(2l / 5, 2l / 5), Point2D(l, l))
set_region!(mmtop, :fixed, Point2D(l - 6.0, 2l / 5 - 4.0), Point2D(l, 2l / 5))

sol, x = solve(mmtop, :Mass_min, :Stress, filt, false, [0.5], [mat.V_lim for mat in mats], [mat.dens for mat in mats], [mat.S for mat in mats], 600)

fig1 = display_solution(:Density, sol, mmtop, x; mat_names=vcat([mat.name for mat in mats], ""))
fig2 = display_solution(:VM_Stress, sol, mmtop, x)