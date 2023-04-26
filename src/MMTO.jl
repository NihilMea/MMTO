module MMTO

using SparseArrays, StaticArrays, LinearAlgebra

export Point2D, FEAProblem, calc_stress, update_E!, set_bc!, solve, solve!, elem_dofs
export Filter
export MMTOProblem, set_region!,calc_mat_type

include("FEA.jl")
include("Filter.jl")
include("MMASolver.jl")
include("visualize.jl")


struct MMTOProblem
    fea::FEAProblem
    # topopt region values
    active_elements::Vector{Bool}
    passive_elements::Vector{Bool}
    fixed_elements::Vector{Bool}

    # x::Vector{Float64} #design variables
    E0::Vector{Float64} # basic materials 
    p::Float64 # penalization parameter
end

function MMTOProblem(fea::FEAProblem, E0::Vector{Float64}, p::Float64)
    E0 = vcat(E0, 1e-3)
    return MMTOProblem(fea, fill(true, fea.num_el), fill(false, fea.num_el), fill(false, fea.num_el), E0, p)
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
end

function SIMP!(mmtop::MMTOProblem, E::AbstractVector{Float64}, dE::Vector{Float64}, x::AbstractVector{Float64})
    E .= x .^ mmtop.p .* mmtop.E0[1]
    dE .= mmtop.p .* x .^ (mmtop.p - 1) .* mmtop.E0[1]
end

function MSIMP!(mmtop::MMTOProblem, E::Vector{Float64}, dE::Matrix{Float64}, x::AbstractMatrix{Float64})
    p = mmtop.p
    E0 = mmtop.E0
    M = length(E0)
    w = zeros(length(E), M)
    dwdx = zeros(length(E), M, M - 1)

    for i in 1:M
        if i == 1
            w[:, i] = prod(x[:, 1:(M-i)] .^ p, dims=2)
            dwdx[:, i, :] = p ./ x[:, :] .* prod(x[:, 1:(M-i)] .^ p, dims=2)
        else
            w[:, i] = (1 .- x[:, M-i+1]) .^ p .* prod(x[:, 1:(M-i)] .^ p, dims=2)
            for j in 1:(M-i+1)
                dwdx[:, i, j] = p ./ x[:, j] .* prod(x[:, 1:(M-i)] .^ p, dims=2) .* (1 .- x[:, M-i+1]) .^ p
            end
            dwdx[:, i, M-i+1] = -prod(x[:, 1:(M-i)] .^ p, dims=2) .* p .* (1 .- x[:, M-i+1]) .^ (p - 1) .* x[:, M-i+1]
        end
    end
    E .= sum(w .* transpose(E0), dims=2)
    dE .= dropdims(sum(dwdx .* transpose(E0), dims=2), dims=2)

end

function calc_mat_type(mmtop::MMTOProblem, x::AbstractMatrix{Float64})
    E0 = mmtop.E0
    M = length(E0)
    w = zeros(size(x,1), M)
    for i in 1:M
        if i == 1
            w[:, i] = prod(x[:, 1:(M-i)], dims=2)
        else
            w[:, i] = (1 .- x[:, M-i+1]) .* prod(x[:, 1:(M-i)], dims=2)
        end
    end
    return vec(sum(w.*transpose(collect((M-1):-1:0)),dims=2))
end

#TODO update for MSIMP
function compliance!(mmtop::MMTOProblem, df0dx::AbstractMatrix{Float64}, sol::FEASolution, dE::Matrix{Float64})
    # df0dx = zeros(length(findall(mmtop.active_elements)))
    Ke0 = mmtop.fea.Ke0
    for (i, el_id) in enumerate(findall(mmtop.active_elements))
        el_dofs = elem_dofs(mmtop.fea, el_id)
        u = sol.U[el_dofs]
        df0dx[i] = -dE[i] * dot(u, Ke0, u)
    end
    f0 = dot(sol.U, sol.F)
    return f0
end

function mass!(mmtop::MMTOProblem, df0dx::AbstractMatrix{Float64}, x::AbstractMatrix{Float64}, dens::Vector{Float64})
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
            for j in 1:(M-i+1)
                dwdx[:, i, j] = q ./ x[:, j] .* prod(x[:, 1:(M-i)] .^ q, dims=2) .* (1 .- x[:, M-i+1]) .^ q
            end
            dwdx[:, i, M-i+1] = -prod(x[:, 1:(M-i)] .^ q, dims=2) .* q .* (1 .- x[:, M-i+1]) .^ (q - 1) .* x[:, M-i+1]
        end
    end
    f0 = sum(sum(w .* transpose(dens), dims=2) .* mmtop.fea.Ve)
    df0dx .= dropdims(sum(dwdx .* transpose(dens), dims=2) .* mmtop.fea.Ve, dims=2)
    return f0
end

#TODO update for MSIMP
function volume_constraint!(mmtop::MMTOProblem, dfidx::AbstractMatrix{Float64}, x::AbstractMatrix{Float64}, V_lim::Vector{Float64})
    N = length(findall(mmtop.active_elements))

    for i in 1:N
        dfidx[i] = 1.0
    end
    fi = sum(x) / N - V_lim
    return fi
end

function stress_constraint!(mmtop::MMTOProblem, dfidx::AbstractMatrix{Float64}, sol::FEASolution, dE::Matrix{Float64}, x::AbstractMatrix{Float64}, σ_max::Vector{Float64})
    return stress_constraint!(mmtop, 1, dfidx, sol, dE, x, σ_max)
end

function stress_constraint!(mmtop::MMTOProblem, mat::Int, dfidx::AbstractMatrix{Float64}, sol::FEASolution, dE::Matrix{Float64}, x::AbstractMatrix{Float64}, σ_max::Vector{Float64})
    T = [1.0 -0.5 0.0
        -0.5 1.0 0.0
        0.0 0.0 3.0]
    s = 0.5
    p = 8
    B = mmtop.fea.Bσ
    D0 = mmtop.fea.D0
    Ke0 = mmtop.fea.Ke0
    N = mmtop.fea.num_el
    dσPNdx = zeros(N)
    dσVMdσ = zeros(N, 3)
    σ = zeros(N, 3)
    σVM = zeros(N)

    M = length(σ_max) + 1
    w = zeros(size(x, 1))
    dwdx = zeros(size(x, 1), M - 1)

    if mat == 1
        w = prod(x[:, 1:(M-1)] .^ s, dims=2)
        dwdx[:, :] = s ./ x[:, :] .* prod(x[:, 1:(M-1)] .^ s, dims=2)
    else
        w[:] = (1 .- x[:, M-mat+1]) .^ s .* prod(x[:, 1:(M-mat)] .^ s, dims=2)
        for j in 1:(M-mat+1)
            dwdx[:, j] = s ./ x[:, j] .* prod(x[:, 1:(M-mat)] .^ s, dims=2) .* (1 .- x[:, M-mat+1]) .^ s
        end
        dwdx[:, M-mat+1] = -prod(x[:, 1:(M-mat)] .^ s, dims=2) .* s .* (1 .- x[:, M-mat+1]) .^ (s - 1) .* x[:, M-mat+1]
    end

    for el_id in 1:N
        el_dofs = elem_dofs(mmtop.fea, el_id)
        σ[el_id, :] = D0 * B * sol.U[el_dofs]
        σ_pen = w[el_id] .* (mmtop.E0[mat] .* σ[el_id, :])
        σVM[el_id] = sqrt(dot(σ_pen, T, σ_pen))
        dσVMdσ[el_id, :] = (T * σ_pen) / σVM[el_id]
    end
    σPN = sum(σVM .^ p) / N
    dσPNdσVM = (σPN^(1 / p - 1) / N) .* (σVM) .^ (p - 1)
    σPN = σPN^(1 / p)
    rhs = zeros(mmtop.fea.num_dof)
    dσPNdx = zeros(N, M - 1)
    for el_id in 1:N
        el_dofs = elem_dofs(mmtop.fea, el_id)
        rhs[el_dofs] += vec(w[el_id] * mmtop.E0[mat] * dσPNdσVM[el_id] .* transpose(dσVMdσ[el_id, :]) * D0 * B)
        dσPNdx[el_id, :] = dwdx[el_id, :] .* (mmtop.E0[mat] * dσPNdσVM[el_id] * transpose(dσVMdσ[el_id, :]) * σ[el_id, :])
    end
    λ = sol.K \ rhs
    for el_id in 1:N
        el_dofs = elem_dofs(mmtop.fea, el_id)
        u = sol.U[el_dofs]
        dKu = transpose(dE[el_id, :]) .* (Ke0 * u)
        dσPNdx[el_id, :] -= transpose(transpose(λ[el_dofs]) * dKu)
    end

    fi = σPN - σ_max[mat]
    dfidx .= dσPNdx
    dfidx[mmtop.fixed_elements, :] .= 0.0
    dfidx[mmtop.passive_elements, :] .= 0.0

    return fi
end

function solve(mmtop::MMTOProblem, filt::Filter, x_init::Float64, V_lim::Vector{Float64}, dens::Vector{Float64}, σ_max::Vector{Float64}, maxoutit::Int)
    mat_num = length(mmtop.E0) - 1 # number of materials (void not included)
    n = length(findall(mmtop.active_elements))
    m = 1 * mat_num
    opt_params = OptimizerParameters()
    prob_type = MMAProblemType(1.0, zeros(m), ones(m), fill(1e3, m))
    fea = mmtop.fea

    x_min = fill(1e-3, n * mat_num)
    x_max = fill(1.0, n * mat_num)
    l = copy(x_min)
    u = copy(x_max)
    x_all = fill(x_init, fea.num_el, mat_num)
    x_all[mmtop.passive_elements, :] .= 1e-3
    x_all[mmtop.fixed_elements, :] .= 1.0

    x = reshape(view(x_all, mmtop.active_elements, :), :, 1)
    x_prev = copy(x)
    x_pprev = copy(x)

    E = fea.E
    # E[mmtop.passive_elements] .= mmtop.E0[end]
    # E[mmtop.fixed_elements] .= mmtop.E0[1]
    dE = zeros(length(E), mat_num)
    ρ_all = apply_filter(filt, x_all)
    region_update!(mmtop, ρ_all)
    ρ = @view ρ_all[mmtop.active_elements, :]
    MSIMP!(mmtop, E, dE, ρ_all)
    sol = solve(fea)

    df0dx_full = zeros(size(x_all))
    dfidx_full = zeros(size(x_all)..., m)
    df0dx = view(df0dx_full, mmtop.active_elements, :)
    dfidx = view(dfidx_full, mmtop.active_elements, :, :)
    f0 = mass!(mmtop, df0dx, ρ, dens)
    df0dx_full .= apply_filter(filt, df0dx_full)
    fi = zeros(mat_num)
    for mat in 1:mat_num
        fi[mat] = stress_constraint!(mmtop, view(dfidx_full, :, :, mat), sol, dE, ρ_all, σ_max)
        dfidx_full[:, :, mat] .= apply_filter(filt, dfidx_full[:, :, mat])
    end
    # f0 = compliance!(mmtop, df0dx, sol, E, dE)
    # fi = volume_constraint!(mmtop, view(dfidx_full, :, 1), ρ, V_lim)
    # df0dx_full .= apply_filter(filt, df0dx_full)
    # for col in 1:m
    #     dfidx_full[:, :, col] .= apply_filter(filt, dfidx_full[:, :, col])
    # end


    kkttol = 1e-6

    kktnorm = kkttol + 10
    outit = 0
    while (kktnorm > kkttol) & (outit < maxoutit)
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
        region_update!(mmtop, ρ_all)
        MSIMP!(mmtop, E, dE, ρ_all)
        sol = solve(fea)
        f0 = mass!(mmtop, df0dx, ρ, dens)
        df0dx_full .= apply_filter(filt, df0dx_full)
        for mat in 1:mat_num
            fi[mat] = stress_constraint!(mmtop, view(dfidx_full, :, :, mat), sol, dE, ρ_all, σ_max)
            dfidx_full[:, :, mat] .= apply_filter(filt, dfidx_full[:, :, mat])
        end
        # f0 = compliance!(mmtop, df0dx, sol, E, dE)
        # fi = volume_constraint!(mmtop, view(dfidx_full, :, 1), ρ, V_lim)
        # df0dx_full .= apply_filter(filt, df0dx_full)
        # for col in 1:m
        #     dfidx_full[:, col] .= apply_filter(filt, dfidx_full[:, col])
        # end
        # The residual vector of the KKT conditions is calculated:
        kktnorm, residumax = kktcheck(m, n, vec(x), y, z, λ, ξ, η, μ, ζ, s, x_min, x_max, fi, vec(reshape(df0dx, :)), reshape(dfidx, :, m), prob_type)
        print("Iter: ", outit, " Targ. func: ", f0, " constr: ", fi, "\n")
    end
    return sol, ρ_all
end

end # module MMTO
