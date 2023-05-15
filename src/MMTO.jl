module MMTO

using SparseArrays, StaticArrays, LinearAlgebra, ColorSchemes, Colors, JLD2
import GLMakie as Mke


export Point2D, FEAProblem, calc_stress, set_E!, set_bc!, solve, solve!, elem_dofs
export stress_constraint!, region_update!, MSIMP!
export Filter, apply_filter
export MMTOProblem, set_region!, calc_mat_type
export viz, display_solution
export parse_file

include("FEA.jl")
include("Filter.jl")
include("MMASolver.jl")



struct MMTOProblem
    fea::FEAProblem
    # topopt region values
    active_elements::Vector{Bool}
    passive_elements::Vector{Bool}
    fixed_elements::Vector{Bool}

    # x::Vector{Float64} #design variables
    E0::Vector{Float64} # basic materials 
    q::Float64 # elastic modulus penalization parameter
    s::Float64 # stress penalization parameter
    p::Float64 # p-norm parameter
end

function MMTOProblem(fea::FEAProblem, E0::Vector{Float64}, q::Float64, s::Float64, p::Float64)
    E0 = vcat(E0, 1e-3)
    return MMTOProblem(fea, fill(true, fea.num_el), fill(false, fea.num_el), fill(false, fea.num_el), E0, q, s, p)
end

function set_region!(mmtop::MMTOProblem, type::Symbol, p1::Point2D, p2::Point2D)
    x1 = min(p1[1], p2[1])
    x2 = max(p1[1], p2[1])
    y1 = min(p1[2], p2[2])
    y2 = max(p1[2], p2[2])

    for (el_id, el) in enumerate(eachrow(mmtop.fea.elems_coords))
        if (el[1] >= x1 && el[1] <= x2 && el[2] >= y1 && el[2] <= y2)
            mmtop.active_elements[el_id] = false
            mmtop.passive_elements[el_id] = false
            mmtop.fixed_elements[el_id] = false
            if type == :active
                mmtop.active_elements[el_id] = true
            elseif type == :passive
                mmtop.passive_elements[el_id] = true
            elseif type == :fixed
                mmtop.fixed_elements[el_id] = true
            else
                error("WrongRegionTypePassed")
            end
        end
    end
end

function region_update!(mmtop::MMTOProblem, x::Vector{Float64})
    x[mmtop.passive_elements] .= 1e-3
    x[mmtop.fixed_elements] .= 1.0
end

function region_update!(mmtop::MMTOProblem, x::Matrix{Float64})
    x[mmtop.passive_elements, :] .= 1e-3
    x[mmtop.fixed_elements, :] .= 1.0
    x[findall(x -> x < 1e-3, x)] .= 1e-3
    x[findall(x -> x > 1.0, x)] .= 1.0

end

function SIMP!(mmtop::MMTOProblem, E::SubArray{Float64}, dE::Vector{Float64}, x::SubArray{Float64})
    E .= x .^ mmtop.p .* mmtop.E0[1]
    dE .= mmtop.p .* x .^ (mmtop.p - 1) .* mmtop.E0[1]
end

function MSIMP!(mmtop::MMTOProblem, E::Vector{Float64}, dE::Matrix{Float64}, x::Matrix{Float64})
    q = mmtop.q
    E0 = mmtop.E0
    M = length(E0)
    w = zeros(length(E), M)
    dwdx = zeros(length(E), M, M - 1)

    for i in 1:M
        if i == 1
            w[:, i] = prod(x[:, 1:(M-i)] .^ q, dims=2)
            dwdx[:, i, :] = q ./ x[:, :] .* prod(x[:, 1:(M-i)] .^ q, dims=2)
        else
            w[:, i] = (1 .- x[:, M-i+1]) .^ q .* prod(x[:, 1:(M-i)] .^ q, dims=2)
            for j in 1:(M-i)
                dwdx[:, i, j] = q ./ x[:, j] .* prod(x[:, 1:(M-i)] .^ q, dims=2) .* (1 .- x[:, M-i+1]) .^ q
            end
            dwdx[:, i, M-i+1] = -prod(x[:, 1:(M-i)] .^ q, dims=2) .* q .* (1 .- x[:, M-i+1]) .^ (q - 1)
        end
    end
    E .= sum(w .* transpose(E0), dims=2)
    dE .= dropdims(sum(dwdx .* transpose(E0), dims=2), dims=2)

end

function calc_mat_type(mmtop::MMTOProblem, x::Matrix{Float64})
    E0 = mmtop.E0
    M = length(E0)
    w = zeros(size(x, 1), M)
    for i in 1:M
        if i == 1
            w[:, i] = prod(x[:, 1:(M-i)], dims=2)
        else
            w[:, i] = (1 .- x[:, M-i+1]) .* prod(x[:, 1:(M-i)], dims=2)
        end
    end
    return vec(sum(w .* transpose(collect((M-1):-1:0)), dims=2)), w
end

#TODO update for MSIMP
function compliance!(mmtop::MMTOProblem, df0dx::SubArray{Float64}, sol::FEASolution, E::Vector{Float64}, dE::Matrix{Float64})
    # df0dx = zeros(length(findall(mmtop.active_elements)))
    Ke0 = mmtop.fea.Ke0
    f0 = 0.0
    for (i, el_id) in enumerate(findall(mmtop.active_elements))
        el_dofs = elem_dofs(mmtop.fea, el_id)
        u = sol.U[el_dofs]
        uKu = dot(u, Ke0, u)
        df0dx[i, :] .= -dE[el_id, :] .* uKu
        f0 += E[el_id] .* uKu
    end
    f = maximum(abs.(sol.F))
    df0dx ./= f
    # f0 = dot(sol.U, sol.F)
    return f0 / f
end

function mass!(mmtop::MMTOProblem, df0dx::SubArray{Float64}, x::SubArray{Float64}, dens::Vector{Float64})
    q = 1
    M = length(dens)
    w = zeros(size(x, 1), M)
    dwdx = zeros(size(x, 1), M, M - 1)

    for i in 1:M
        if i == 1
            w[:, i] = prod(x[:, 1:(M-i)] .^ q, dims=2)
            dwdx[:, i, :] = q ./ x[:, :] .* prod(x[:, 1:(M-i)] .^ q, dims=2)
        else
            w[:, i] = (1 .- x[:, M-i+1]) .^ q .* prod(x[:, 1:(M-i)] .^ q, dims=2)
            for j in 1:(M-i)
                dwdx[:, i, j] = q ./ x[:, j] .* prod(x[:, 1:(M-i)] .^ q, dims=2) .* (1 .- x[:, M-i+1]) .^ q
            end
            dwdx[:, i, M-i+1] = -prod(x[:, 1:(M-i)] .^ q, dims=2) .* q .* (1 .- x[:, M-i+1]) .^ (q - 1)
        end
    end
    act_el_num = length(findall(mmtop.active_elements))
    # f0 = sum(sum(w .* transpose(dens), dims=2) .* mmtop.fea.Ve) ./ (act_el_num * mmtop.fea.Ve * dens[1])
    # df0dx .= dropdims(sum(dwdx .* transpose(dens), dims=2) .* mmtop.fea.Ve, dims=2) ./ (act_el_num * mmtop.fea.Ve * dens[1])
    f0 = sum(sum(w .* transpose(dens), dims=2) .* mmtop.fea.Ve)
    df0dx .= dropdims(sum(dwdx .* transpose(dens), dims=2) .* mmtop.fea.Ve, dims=2)
    return f0
end

function volume_constraint!(mmtop::MMTOProblem, mat::Int, dfidx::SubArray{Float64}, x::Matrix{Float64}, V_lim::Vector{Float64})
    q = 1
    M = length(V_lim) + 1
    w = zeros(size(x, 1))
    dwdx = zeros(size(x, 1), M - 1)

    if mat == 1
        w = prod(x[:, 1:(M-1)] .^ q, dims=2)
        dwdx = q ./ x[:, :] .* prod(x[:, 1:(M-1)] .^ q, dims=2)
    else
        w = (1 .- x[:, M-mat+1]) .^ q .* prod(x[:, 1:(M-mat)] .^ q, dims=2)
        for j in 1:(M-mat)
            dwdx[:, j] = q ./ x[:, j] .* prod(x[:, 1:(M-mat)] .^ q, dims=2) .* (1 .- x[:, M-mat+1]) .^ q
        end
        dwdx[:, M-mat+1] = -prod(x[:, 1:(M-mat)] .^ q, dims=2) .* q .* (1 .- x[:, M-mat+1]) .^ (q - 1)
    end
    act_el_num = length(findall(mmtop.active_elements))
    fi = sum(w[mmtop.active_elements] .* mmtop.fea.Ve) / (act_el_num * mmtop.fea.Ve) / V_lim[mat] - 1.0
    dfidx[mmtop.active_elements, :] .= dwdx[mmtop.active_elements, :] .* mmtop.fea.Ve / (act_el_num * mmtop.fea.Ve) / V_lim[mat]
    dfidx[mmtop.passive_elements, :] .= 0.0
    dfidx[mmtop.fixed_elements, :] .= 0.0
    return fi
end

function stress_constraint!(mmtop::MMTOProblem, dfidx::SubArray{Float64}, sol::FEASolution, dE::Matrix{Float64}, x::SubArray{Float64}, σ_max::Vector{Float64})
    return stress_constraint!(mmtop, 1, dfidx, sol, dE, x, σ_max)
end

function stress_constraint!(mmtop::MMTOProblem, mat::Int, dfidx::SubArray{Float64}, sol::FEASolution, dE::Matrix{Float64}, x::Matrix{Float64}, σ_max::Vector{Float64})
    T = [1.0 -0.5 0.0
        -0.5 1.0 0.0
        0.0 0.0 3.0]
    s = mmtop.s
    B = mmtop.fea.Bσ
    D0 = mmtop.fea.D0
    Ke0 = mmtop.fea.Ke0
    N = mmtop.fea.num_el
    p = mmtop.p
    act_elems = sort(union(findall(mmtop.active_elements)))

    n_act = length(act_elems)
    dσPNdx = zeros(N)
    dσVMdσ = zeros(N, 3)
    σ = zeros(n_act, 3)
    σVM = zeros(n_act)

    M = length(σ_max) + 1
    w = zeros(size(x, 1))
    dwdx = zeros(size(x, 1), M - 1)
    if mat == 1
        w = prod(x[:, 1:(M-1)] .^ s, dims=2)
        dwdx = s ./ x[:, :] .* prod(x[:, 1:(M-1)] .^ s, dims=2)
    else
        w = (1 .- x[:, M-mat+1]) .^ s .* prod(x[:, 1:(M-mat)] .^ s, dims=2)
        for j in 1:(M-mat)
            dwdx[:, j] = s ./ x[:, j] .* prod(x[:, 1:(M-mat)] .^ s, dims=2) .* (1 .- x[:, M-mat+1]) .^ s
        end
        dwdx[:, M-mat+1] = -prod(x[:, 1:(M-mat)] .^ s, dims=2) .* s .* (1 .- x[:, M-mat+1]) .^ (s - 1)
    end

    dwdx[mmtop.fixed_elements, :] .= 0.0
    dwdx[mmtop.passive_elements, :] .= 0.0

    for (i, el_id) in enumerate(act_elems)
        el_dofs = elem_dofs(mmtop.fea, el_id)
        σ[i, :] = D0 * B * sol.U[el_dofs]
        σ_pen = w[el_id] .* (mmtop.E0[mat] .* σ[i, :])
        σVM[i] = sqrt(dot(σ_pen, T, σ_pen))
        if w[el_id] == 0
            dσVMdσ[el_id, :] .= 0.0
        else
            dσVMdσ[el_id, :] = (T * σ_pen) / σVM[i]
        end
    end
    σPN = sum((σVM ./ σ_max[mat]) .^ p) / n_act
    dσPNdσVM = zeros(N)
    dσPNdσVM[act_elems] .= σPN^(1 / p - 1) .* σVM .^ (p - 1) ./ n_act ./ σ_max[mat] .^ p
    σPN = σPN^(1 / p)
    rhs = zeros(mmtop.fea.num_dof)
    dσPNdx = zeros(N, M - 1)
    for (i, el_id) in enumerate(act_elems)
        el_dofs = elem_dofs(mmtop.fea, el_id)
        rhs[el_dofs] += vec(w[el_id] * mmtop.E0[mat] * dσPNdσVM[el_id] .* transpose(dσVMdσ[el_id, :]) * D0 * B)
        dσPNdx[el_id, :] = dwdx[el_id, :] .* (mmtop.E0[mat] * dσPNdσVM[el_id] * transpose(dσVMdσ[el_id, :]) * σ[i, :])
    end
    λ = sol.K \ rhs
    for (i, el_id) in enumerate(act_elems)
        el_dofs = elem_dofs(mmtop.fea, el_id)
        u = sol.U[el_dofs]
        dKu = transpose(dE[el_id, :]) .* (Ke0 * u)
        dσPNdx[el_id, :] -= transpose(transpose(λ[el_dofs]) * dKu)
    end

    fi = σPN - 1.0
    dfidx .= dσPNdx
    dfidx[mmtop.fixed_elements, :] .= 0.0
    dfidx[mmtop.passive_elements, :] .= 0.0
    return fi
end

function calc_stress(mmtop::MMTOProblem, sol::FEASolution, x::Matrix{Float64}, σ_max::Vector{Float64}, type::Symbol)

    T = [1.0 -0.5 0.0
        -0.5 1.0 0.0
        0.0 0.0 3.0]
    s = mmtop.s
    B = mmtop.fea.Bσ
    D0 = mmtop.fea.D0
    N = mmtop.fea.num_el
    act_elems = 1:N

    M = length(σ_max) + 1
    w = zeros(N, M)
    for mat in 1:M
        if mat == 1
            w[:, mat] = prod(x[:, 1:(M-1)] .^ s, dims=2)
        else
            w[:, mat] = (1 .- x[:, M-mat+1]) .^ s .* prod(x[:, 1:(M-mat)] .^ s, dims=2)
        end
    end

    _, w_mat = calc_mat_type(mmtop, x)
    σ_pen = zeros(N, 3)
    mats = argmax.(eachrow(w_mat))

    for (i, el_id) in enumerate(act_elems)
        el_dofs = elem_dofs(mmtop.fea, el_id)
        σ = D0 * B * sol.U[el_dofs]
        σ_pen[i, :] = (w[el_id, mats[i]] * mmtop.E0[mats[i]] ).* σ
    end

    if type == :x
        σ = σ_pen[:, 1]
    elseif type == :y
        σ = σ_pen[:, 2]
    elseif type == :xy
        σ = σ_pen[:, 3]
    elseif type == :vM # von-Mises stress
        σ = [sqrt(transpose(s) * T * s) for s in eachrow(σ_pen)]
    elseif type == :MS # margin of safety
        σ = [sqrt(transpose(s) * T * s) for s in eachrow(σ_pen)] ./ [mat == M ? 1e-3 : σ_max[mat] for mat in mats]
    else
        error("Wrong output type provided")
    end
    return σ
end

# function stress_constraint!(mmtop::MMTOProblem, mat::Int, dfidx::SubArray{Float64}, sol::FEASolution, dE::Matrix{Float64}, x::SubArray{Float64}, σ_max::Vector{Float64})
#     T = [1.0 -0.5 0.0
#         -0.5 1.0 0.0
#         0.0 0.0 3.0]
#     s = 0.5
#     p = 8
#     B = mmtop.fea.Bσ
#     D0 = mmtop.fea.D0
#     Ke0 = mmtop.fea.Ke0
#     N = mmtop.fea.num_el
#     dσPNdx = zeros(N)
#     dσVMdσ = zeros(N, 3)
#     σ = zeros(N, 3)
#     σVM = zeros(N)

#     M = length(σ_max) + 1
#     w = zeros(size(x, 1))
#     dwdx = zeros(size(x, 1), M - 1)
#     if mat == 1
#         w = prod(x[:, 1:(M-1)] .^ s, dims=2)
#         dwdx = s ./ x[:, :] .* prod(x[:, 1:(M-1)] .^ s, dims=2)
#     else
#         w = (1 .- x[:, M-mat+1]) .^ s .* prod(x[:, 1:(M-mat)] .^ s, dims=2)
#         for j in 1:(M-mat)
#             dwdx[:, j] = s ./ x[:, j] .* prod(x[:, 1:(M-mat)] .^ s, dims=2) .* (1 .- x[:, M-mat+1]) .^ s
#         end
#         dwdx[:, M-mat+1] = -prod(x[:, 1:(M-mat)] .^ s, dims=2) .* s .* (1 .- x[:, M-mat+1]) .^ (s - 1)
#     end

#     for el_id in 1:N
#         el_dofs = elem_dofs(mmtop.fea, el_id)
#         σ[el_id, :] = D0 * B * sol.U[el_dofs]
#         σ_pen = w[el_id] .* (mmtop.E0[mat] .* σ[el_id, :])
#         σVM[el_id] = sqrt(dot(σ_pen, T, σ_pen))
#         if w[el_id] == 0
#             dσVMdσ[el_id, :] .= 0.0
#         else
#             dσVMdσ[el_id, :] = (T * σ_pen) / σVM[el_id]
#         end
#     end
#     σPN = sum((σVM./σ_max[mat] ) .^ p) / N
#     dσPNdσVM = σPN^(1 / p - 1) .* σVM .^ (p - 1) ./ N ./σ_max[mat]^p
#     σPN = σPN^(1 / p)
#     rhs = zeros(mmtop.fea.num_dof)
#     dσPNdx = zeros(N, M - 1)
#     for el_id in 1:N
#         el_dofs = elem_dofs(mmtop.fea, el_id)
#         rhs[el_dofs] += vec(w[el_id] * mmtop.E0[mat] * dσPNdσVM[el_id] .* transpose(dσVMdσ[el_id, :]) * D0 * B)
#         dσPNdx[el_id, :] = dwdx[el_id, :] .* (mmtop.E0[mat] * dσPNdσVM[el_id] * transpose(dσVMdσ[el_id, :]) * σ[el_id, :])
#     end
#     λ = sol.K \ rhs
#     for el_id in 1:N
#         el_dofs = elem_dofs(mmtop.fea, el_id)
#         u = sol.U[el_dofs]
#         dKu = transpose(dE[el_id, :]) .* (Ke0 * u)
#         dσPNdx[el_id, :] -= transpose(transpose(λ[el_dofs]) * dKu)
#     end

#     fi = σPN - 1.0
#     dfidx .= dσPNdx
#     dfidx[mmtop.fixed_elements, :] .= 0.0
#     dfidx[mmtop.passive_elements, :] .= 0.0
#     return fi
# end

function solve(mmtop::MMTOProblem, targ::Symbol, constr::Union{Symbol,Vector{Symbol}}, filt::Filter, use_proj::Bool, x_init::Union{Float64,Vector{Float64}}, V_lim::Vector{Float64}, dens::Vector{Float64}, σ_max::Vector{Float64}, maxoutit::Int)
    dens = vcat(dens, 1e-12)
    mat_num = length(mmtop.E0) - 1 # number of materials (void not included)
    n = length(findall(mmtop.active_elements)) * mat_num
    if typeof(constr) == Symbol
        constr = [constr]
    end
    m = length(constr) * mat_num
    opt_params = OptimizerParameters()
    prob_type = MMAProblemType(1.0, zeros(m), ones(m), fill(1e3, m))
    fea = mmtop.fea

    x_min = fill(1e-3, n)
    x_max = fill(1.0, n)
    l = copy(x_min)
    u = copy(x_max)
    if typeof(x_init) == Float64
        if mat_num - 1 != 0
            x_all = repeat(hcat(x_init, zeros(mat_num - 1)), fea.num_el)
        else
            x_all = fill(x_init, fea.num_el, mat_num)
        end
    else
        if length(x_init) != mat_num
            error("Wrong number of initial values for design variables")
        end
        x_all = repeat(transpose(x_init), fea.num_el)
    end
    x_all[mmtop.passive_elements, :] .= 1e-3
    x_all[mmtop.fixed_elements, :] .= 1.0

    x = reshape(view(x_all, mmtop.active_elements, :), :, 1)
    x_prev = copy(x)
    x_pprev = copy(x)

    E = fea.E
    dE = zeros(length(E), mat_num)

    ρ_all = apply_filter(filt, x_all)
    # Heaviside filtration NOTE: not working
    if use_proj
        β = 0.0
        dρdx = zeros(size(ρ_all))
        apply_projection!(β, ρ_all, dρdx)
    end
    ρ = @view ρ_all[mmtop.active_elements, :]
    MSIMP!(mmtop, E, dE, ρ_all)
    region_update!(mmtop, ρ_all)
    sol = solve(fea)

    df0dx_full = zeros(size(x_all))
    dfidx_full = zeros(size(x_all)..., m)
    df0dx = view(df0dx_full, mmtop.active_elements, :)
    dfidx = view(dfidx_full, mmtop.active_elements, :, :)

    if targ == :Mass_min
        f0 = mass!(mmtop, df0dx, ρ, dens)
    elseif targ == :Compl_min
        f0 = compliance!(mmtop, df0dx, sol, E, dE)
    end

    df0dx_full .= apply_filter(filt, df0dx_full)
    fi = zeros(m)

    for (i, constraint) in enumerate(constr)
        for mat in 1:mat_num
            if constraint == :Stress
                fi[mat+mat_num*(i-1)] = stress_constraint!(mmtop, mat, view(dfidx_full, :, :, mat + mat_num * (i - 1)), sol, dE, ρ_all, σ_max)
            elseif constraint == :Volume
                fi[mat+mat_num*(i-1)] = volume_constraint!(mmtop, mat, view(dfidx_full, :, :, mat + mat_num * (i - 1)), ρ_all, V_lim)
            end
            dfidx_full[:, :, mat+mat_num*(i-1)] .= apply_filter(filt, dfidx_full[:, :, mat+mat_num*(i-1)])
        end
    end

    if use_proj
        df0dx_full .= dρdx .* df0dx_full
        dfidx_full .= dρdx .* dfidx_full
    end

    outit = 0
    exit_flag = false
    exit_count = 0
    while true
        outit = outit + 1

        # The MMA subproblem is solved at the point xval:
        if typeof(fi) !== Vector{Float64}
            fi = [fi]
        end
        x_new, y, z, λ, ξ, η, μ, ζ, s, l, u = mmasub(n, m, opt_params, prob_type, f0, fi, vec(reshape(df0dx, :)), reshape(dfidx, :, m),
            vec(x), vec(x_prev), vec(x_pprev), x_min, x_max, l, u, outit)
        x_pprev .= x_prev
        x_prev .= x
        x .= x_new
        # apply filter to updated
        ρ_all .= apply_filter(filt, x_all)
        # Heaviside filtration NOTE: not working
        if use_proj
            apply_projection!(β, ρ_all, dρdx)
        end
        MSIMP!(mmtop, E, dE, ρ_all)
        region_update!(mmtop, ρ_all)
        sol = solve(fea)

        if targ == :Mass_min
            f0_new = mass!(mmtop, df0dx, ρ, dens)
            f0_prev = f0
            f0 = f0_new
        elseif targ == :Compl_min
            f0_new = compliance!(mmtop, df0dx, sol, E, dE)
            f0_prev = f0
            f0 = f0_new
        end

        df0dx_full .= apply_filter(filt, df0dx_full)

        for (i, constraint) in enumerate(constr)
            for mat in 1:mat_num
                if constraint == :Stress
                    fi[mat+mat_num*(i-1)] = stress_constraint!(mmtop, mat, view(dfidx_full, :, :, mat + mat_num * (i - 1)), sol, dE, ρ_all, σ_max)
                elseif constraint == :Volume
                    fi[mat+mat_num*(i-1)] = volume_constraint!(mmtop, mat, view(dfidx_full, :, :, mat + mat_num * (i - 1)), ρ_all, V_lim)
                end
                dfidx_full[:, :, mat+mat_num*(i-1)] .= apply_filter(filt, dfidx_full[:, :, mat+mat_num*(i-1)])
            end
        end

        if use_proj
            df0dx_full .= dρdx .* df0dx_full
            dfidx_full .= dρdx .* dfidx_full
            if outit > 50 && β <= 5.0
                β += 1.1^outit
            end
        end
        # The residual vector of the KKT conditions is calculated:
        # kktnorm, residumax = kktcheck(m, n, vec(x), y, z, λ, ξ, η, μ, ζ, s, x_min, x_max, fi, vec(reshape(df0dx, :)), reshape(dfidx, :, m), prob_type)
        residu = abs(f0 - f0_prev) / f0_prev
        print("Iter: ", outit, " Targ. func: ", f0, " constr: ", fi, " residu: ", residu, "\n")

        if outit >= maxoutit
            print("Finished, reached max iterations")
            break
        elseif residu < 1e-4
            if exit_flag
                exit_count += 1
            else
                exit_count = 0
                exit_flag = true
            end
            if exit_count > 20
                print("Objective function residual is less then 0.01% in 20 iterations")
                break
            end
        else
            exit_flag = false
        end
    end
    return sol, ρ_all, f0, outit
end

function solve(mmtop::MMTOProblem, targ::Symbol, constr::Union{Symbol,Vector{Symbol}}, filt::Filter, use_proj::Bool, x::Matrix{Float64}, V_lim::Vector{Float64}, dens::Vector{Float64}, σ_max::Vector{Float64}, maxoutit::Int, cur_it::Int)
    dens = vcat(dens, 1e-12)
    mat_num = length(mmtop.E0) - 1 # number of materials (void not included)
    n = length(findall(mmtop.active_elements)) * mat_num
    if typeof(constr) == Symbol
        constr = [constr]
    end
    m = length(constr) * mat_num
    opt_params = OptimizerParameters()
    prob_type = MMAProblemType(1.0, zeros(m), ones(m), fill(1e3, m))
    fea = mmtop.fea

    x_min = fill(1e-3, n)
    x_max = fill(1.0, n)
    l = copy(x_min)
    u = copy(x_max)
    x_all = x
    x_all[mmtop.passive_elements, :] .= 1e-3
    x_all[mmtop.fixed_elements, :] .= 1.0

    x = reshape(view(x_all, mmtop.active_elements, :), :, 1)
    x_prev = copy(x)
    x_pprev = copy(x)

    E = fea.E
    dE = zeros(length(E), mat_num)

    ρ_all = apply_filter(filt, x_all)
    # Heaviside filtration NOTE: not working
    if use_proj
        β = 0.0
        dρdx = zeros(size(ρ_all))
        apply_projection!(β, ρ_all, dρdx)
    end
    ρ = @view ρ_all[mmtop.active_elements, :]
    MSIMP!(mmtop, E, dE, ρ_all)
    region_update!(mmtop, ρ_all)
    sol = solve(fea)

    df0dx_full = zeros(size(x_all))
    dfidx_full = zeros(size(x_all)..., m)
    df0dx = view(df0dx_full, mmtop.active_elements, :)
    dfidx = view(dfidx_full, mmtop.active_elements, :, :)

    if targ == :Mass_min
        f0 = mass!(mmtop, df0dx, ρ, dens)
    elseif targ == :Compl_min
        f0 = compliance!(mmtop, df0dx, sol, E, dE)
    end

    df0dx_full .= apply_filter(filt, df0dx_full)
    fi = zeros(m)

    for (i, constraint) in enumerate(constr)
        for mat in 1:mat_num
            if constraint == :Stress
                fi[mat+mat_num*(i-1)] = stress_constraint!(mmtop, mat, view(dfidx_full, :, :, mat + mat_num * (i - 1)), sol, dE, ρ_all, σ_max)
            elseif constraint == :Volume
                fi[mat+mat_num*(i-1)] = volume_constraint!(mmtop, mat, view(dfidx_full, :, :, mat + mat_num * (i - 1)), ρ_all, V_lim)
            end
            dfidx_full[:, :, mat+mat_num*(i-1)] .= apply_filter(filt, dfidx_full[:, :, mat+mat_num*(i-1)])
        end
    end

    if use_proj
        df0dx_full .= dρdx .* df0dx_full
        dfidx_full .= dρdx .* dfidx_full
    end
    kkttol = 1e-6

    kktnorm = kkttol + 100
    outit = cur_it
    exit_flag = false
    exit_count = 0
    while true
        outit = outit + 1

        # The MMA subproblem is solved at the point xval:
        if typeof(fi) !== Vector{Float64}
            fi = [fi]
        end
        x_new, y, z, λ, ξ, η, μ, ζ, s, l, u = mmasub(n, m, opt_params, prob_type, f0, fi, vec(reshape(df0dx, :)), reshape(dfidx, :, m),
            vec(x), vec(x_prev), vec(x_pprev), x_min, x_max, l, u, outit)
        x_pprev .= x_prev
        x_prev .= x
        x .= x_new
        # apply filter to updated
        ρ_all .= apply_filter(filt, x_all)
        # Heaviside filtration NOTE: not working
        if use_proj
            apply_projection!(β, ρ_all, dρdx)
        end
        MSIMP!(mmtop, E, dE, ρ_all)
        region_update!(mmtop, ρ_all)
        sol = solve(fea)

        if targ == :Mass_min
            f0_new = mass!(mmtop, df0dx, ρ, dens)
            f0_prev = f0
            f0 = f0_new
        elseif targ == :Compl_min
            f0_new = compliance!(mmtop, df0dx, sol, E, dE)
            f0_prev = f0
            f0 = f0_new
        end

        df0dx_full .= apply_filter(filt, df0dx_full)

        for (i, constraint) in enumerate(constr)
            for mat in 1:mat_num
                if constraint == :Stress
                    fi[mat+mat_num*(i-1)] = stress_constraint!(mmtop, mat, view(dfidx_full, :, :, mat + mat_num * (i - 1)), sol, dE, ρ_all, σ_max)
                elseif constraint == :Volume
                    fi[mat+mat_num*(i-1)] = volume_constraint!(mmtop, mat, view(dfidx_full, :, :, mat + mat_num * (i - 1)), ρ_all, V_lim)
                end
                dfidx_full[:, :, mat+mat_num*(i-1)] .= apply_filter(filt, dfidx_full[:, :, mat+mat_num*(i-1)])
            end
        end

        if use_proj
            df0dx_full .= dρdx .* df0dx_full
            dfidx_full .= dρdx .* dfidx_full
            if outit > 50 && β <= 5.0
                β += 1.1^outit
            end
        end
        # The residual vector of the KKT conditions is calculated:
        # kktnorm, residumax = kktcheck(m, n, vec(x), y, z, λ, ξ, η, μ, ζ, s, x_min, x_max, fi, vec(reshape(df0dx, :)), reshape(dfidx, :, m), prob_type)
        residu = abs(f0 - f0_prev) / f0_prev
        print("Iter: ", outit, " Targ. func: ", f0, " constr: ", fi, " residu: ", residu, "\n")

        if outit >= maxoutit
            print("Finished, reached max iterations")
            break
        elseif residu < 1e-4
            if exit_flag
                exit_count += 1
            else
                exit_count = 0
                exit_flag = true
            end
            if exit_count > 20
                print("Objective function residual is less then 0.01% in 20 iterations")
                break
            end
        else
            exit_flag = false
        end

    end
    return sol, ρ_all, f0, outit
end
include("visualize.jl")
include("parser.jl")


end # module MMTO
