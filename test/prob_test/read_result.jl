using JLD2, GLMakie

res = jldopen("results.jld2")
f0 = res["f0_arr"] # get values of target function
lines(1:length(f0),f0) # plot target function values at each iteration
