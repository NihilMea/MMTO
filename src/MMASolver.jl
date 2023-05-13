using Parameters

abstract type AbstractOptSolver end

export OptimizerParameters

struct MMAProblemType
    a0::Float64
    ai::Vector{Float64}
    di::Vector{Float64}
    ci::Vector{Float64}
end

@with_kw struct OptimizerParameters
    raa0::Float64 = 1e-5
    albefa::Float64 = 0.1
    move::Float64 = 0.05
    asyinit::Float64 = 0.4
    asydecr::Float64 = 0.7
    asyincr::Float64 = 1.1
    res_tol::Float64 = 0.0 # residual tolerance
    epsilon_min::Float64 = 1e-7
end

function mmasub(n::Int, m::Int, opt_params::OptimizerParameters, pr_type::MMAProblemType, f0::Float64, fi::Vector{Float64}, df0dx::AbstractArray{Float64}, dfidx::AbstractArray{Float64}, x::AbstractArray{Float64}, x_prev::AbstractArray{Float64}, x_pprev::AbstractArray{Float64},
    x_min::Vector{Float64}, x_max::Vector{Float64}, l::Vector{Float64}, u::Vector{Float64}, k::Int)
    dfidx = transpose(dfidx)
    x_max_min = x_max - x_min
    if k < 3
        l = x - opt_params.asyinit * x_max_min
        u = x + opt_params.asyinit * x_max_min
    else
        γ = Vector{Float64}(undef, length(x))
        for i in 1:length(γ)
            ddx = (x[i] - x_prev[i]) .* (x_prev[i] - x_pprev[i])
            if ddx < 0
                γ[i] = opt_params.asydecr
            elseif ddx > 0
                γ[i] = opt_params.asyincr
            else
                γ[i] = 1.0
            end
        end
        l = x - γ .* (x_prev - l)
        u = x + γ .* (u - x_prev)
        l = min.(l, x - 0.01 .* x_max_min)
        l = max.(l, x - 10.0 .* x_max_min)
        u = max.(u, x + 0.01 .* x_max_min)
        u = min.(u, x + 10.0 .* x_max_min)
    end
    # Calculation of the bounds alfa and beta :
    ux = u - x
    xl = x - l
    α = max.(x_min, l + opt_params.albefa * xl, x - opt_params.move * x_max_min)
    β = min.(x_max, u - opt_params.albefa * ux, x + opt_params.move * x_max_min)
    # Calculations of p0, q0, P, Q and b :
    df0_pls = max.(df0dx, 0)
    df0_min = max.(-df0dx, 0)
    dfi_pls = max.(dfidx, 0)
    dfi_min = max.(-dfidx, 0)
    p0 = ux .^ 2 .* (1.001 * df0_pls + 0.001 * df0_min + opt_params.raa0 ./ x_max_min)
    q0 = xl .^ 2 .* (0.001 * df0_pls + 1.001 * df0_min + opt_params.raa0 ./ x_max_min)
    p = transpose(ux) .^ 2 .* (1.001 * dfi_pls + 0.001 * dfi_min .+ transpose(opt_params.raa0 ./ x_max_min))
    q = transpose(xl) .^ 2 .* (0.001 * dfi_pls + 1.001 * dfi_min .+ transpose(opt_params.raa0 ./ x_max_min))
    b = vec(sum(p ./ transpose(ux) + q ./ transpose(xl), dims=2)) - fi
    x, y, z, λ, ξ, η, μ, ζ, s = solve_subproblem(n, m, l, u, α, β, p0, q0, p, q, b, opt_params, pr_type)
    return x, y, z, λ, ξ, η, μ, ζ, s, l, u
end

function solve_subproblem(n::Int, m::Int, l::Vector{Float64}, u::Vector{Float64}, alpha::Vector{Float64},
    beta::Vector{Float64}, p0::Vector{Float64}, q0::Vector{Float64}, p::Matrix{Float64}, q::Matrix{Float64},
    b::Vector{Float64}, opt_params::OptimizerParameters, pr_type::MMAProblemType)
    ε = 1.0

    x = 0.5 * (alpha + beta)
    y = ones(m)
    z = 1.0
    λ = ones(m)
    ξ = max.(1.0, 1.0 ./ (x - alpha))
    η = max.(1.0, 1.0 ./ (beta - x))
    μ = max.(1.0, pr_type.ci ./ 2.0)
    ζ = 1.0
    s = ones(m)

    res_w_norm = 0.0

    while ε > opt_params.epsilon_min
        iter = 1
        while true
            iter += 1
            dx, dy, dz, dλ, dξ, dη, dμ, dζ, ds, res_w_old_norm = find_newton_step(ε, x, y, z, λ, ξ,
                η, μ, ζ, s, n, m, l, u, alpha, beta, p0, q0, p, q, b, pr_type)
            x, y, z, λ, ξ, η, μ, ζ, s, res_w_inf_norm, res_w_norm = calc_next_step(n,m,l, u, alpha, beta, p0, q0, p, q, b, pr_type, res_w_old_norm, ε,
                x, y, z, λ, ξ, η, μ, ζ, s,
                dx, dy, dz, dλ, dξ, dη, dμ, dζ, ds)

            if iter > 100 || res_w_inf_norm < 0.9 * ε
                break
            end
        end
        ε = 0.1 * ε
    end
    return x, y, z, λ, ξ, η, μ, ζ, s
end

function find_newton_step(ε::Float64, x::Array{Float64}, y::Vector{Float64},
    z::Float64, λ::Vector{Float64}, ξ::Vector{Float64},
    η::Vector{Float64}, μ::Vector{Float64}, ζ::Float64, s::Vector{Float64},
    n::Int, m::Int, l::Vector{Float64}, u::Vector{Float64}, alpha::Vector{Float64},
    beta::Vector{Float64}, p0::Vector{Float64}, q0::Vector{Float64}, p::Matrix{Float64}, q::Matrix{Float64},
    b::Vector{Float64}, pr_type::MMAProblemType)
    residu = zeros(n + m + 1 + m + n + n + m + 1 + m)
    x_ind = 1:n
    y_ind =(n).+(1:m)
    z_ind =(n+m).+(1:1)
    λ_ind =(n+m+1).+(1:m)
    ξ_ind =(n+m+1+m).+(1:n)
    η_ind =(n+m+1+m+n).+(1:n)
    μ_ind =(n+m+1+m+2n).+(1:m)
    ζ_ind =(n+m+1+m+2n+m).+(1:1)
    s_ind =(n+m+1+m+2n+m+1).+(1:m)

    r_x = @view residu[x_ind]
    r_y = @view residu[y_ind]
    r_z = @view residu[z_ind]
    r_λ = @view residu[λ_ind]
    r_ξ = @view residu[ξ_ind]
    r_η = @view residu[η_ind]
    r_μ = @view residu[μ_ind]
    r_ζ = @view residu[ζ_ind]
    r_s = @view residu[s_ind]

    # finding newton step
    xa = x - alpha
    bx = beta - x
    ux = u - x
    xl = x - l
    ux2 = ux .* ux
    xl2 = xl .* xl
    uxinv = 1.0 ./ ux
    xlinv = 1.0 ./ xl
    G = p ./ transpose(ux2) - q ./ transpose(xl2)
    g = p * uxinv + q * xlinv
    # psi = plam ./ ux2 - qlam ./ xl2

    plam = p0 + transpose(p) * λ
    qlam = q0 + transpose(q) * λ
    dpsi = plam ./ ux2 - qlam ./ xl2
    ddpsi = 2.0 * plam ./ (ux2 .* ux) + 2.0 * qlam ./ (xl2 .* xl)

    Dx = ddpsi + ξ ./ xa + η ./ bx
    invDx = 1.0 ./ Dx
    invDy = 1.0 ./ (pr_type.di + μ ./ y)
    Dlam = s ./ λ
    Dly = Dlam + invDy
    invDly = 1.0 ./ Dly
    δxt = dpsi - ε ./ xa + ε ./ bx
    δyt = pr_type.ci + pr_type.di .* y - λ - ε ./ y
    δzt = pr_type.a0 - λ' * pr_type.ai - ε ./ z
    δλt = g - pr_type.ai .* z - y - b + ε ./ λ
    δλyt = δλt + δyt .* invDy
    ai = pr_type.ai
    Gt = transpose(G)
    # build linear system  
    if m < n
        A11 = G * (invDx .* Gt)
        A11[diagind(A11)] += Dly
        A12 = ai
        A21 = transpose(ai)
        A22 = -ζ / z
        A = [A11 A12
            A21 A22]
        lhs = [δλyt - G * (invDx .* δxt); δzt]
        # getting solution
        dλdz = A \ lhs
        dλ = dλdz[1:end-1]
        dz = dλdz[end]
        dx = -(invDx .* Gt * dλ) - δxt .* invDx
    else
        A11 = Gt * (invDly .* G)
        A11[diagind(A11)] += Dx
        A12 = -(Gt .* invDly) * ai
        A21 = -transpose(ai) * (invDly .* G)
        A22 = ζ / z .+ transpose(ai) * (invDly .* ai)
        A = [A11 A12
            A21 A22]
        lhs = [-δxt - transpose(G) * (invDly .* δλyt); -δxt + transpose(ai) * (invDly .* δλyt)]
        # getting solution
        dxdz = A \ lhs
        dx = dxdz[1:end-1]
        dz = dλdz[end]
        dλ = (invDly .* G * d) - invDly .* ai * dz + δλyt .* invDly
    end
    dy = (dλ - δyt) .* invDy
    dξ = -ξ .* dx ./ xa - ξ + ε ./ xa
    dη = η .* dx ./ bx - η + ε ./ bx
    dμ = -μ .* dy ./ y - μ + ε ./ y
    dζ = -ζ / z * dz - ζ + ε / z
    ds = -s .* dλ ./ λ - s + ε ./ λ
    r_x .= dpsi - ξ + η
    r_y .= pr_type.ci + pr_type.di .* y - λ - μ
    r_z .= pr_type.a0 - ζ - transpose(λ) * pr_type.ai
    r_λ .= g - pr_type.ai .* z - y + s - b
    r_ξ .= ξ .* xa .- ε
    r_η .= η .* bx .- ε
    r_μ .= μ .* y .- ε
    r_ζ .= ζ * z .- ε
    r_s .= λ .* s .- ε
    res_w_norm = norm(residu)
    return dx, dy, dz, dλ, dξ, dη, dμ, dζ, ds, res_w_norm
end

function calc_next_step(n::Int,m::Int,l::Vector{Float64}, u::Vector{Float64}, alpha::Vector{Float64},
    beta::Vector{Float64}, p0::Vector{Float64}, q0::Vector{Float64}, p::Matrix{Float64}, q::Matrix{Float64},
    b::Vector{Float64}, pr_type::MMAProblemType, res_w_old_norm::Float64, ε::Float64,
    x::Vector{Float64}, y::Vector{Float64}, z::Float64, λ::Vector{Float64}, ξ::Vector{Float64}, η::Vector{Float64}, μ::Vector{Float64}, ζ::Float64, s::Vector{Float64},
    dx::Vector{Float64}, dy::Vector{Float64}, dz::Float64, dλ::Vector{Float64}, dξ::Vector{Float64}, dη::Vector{Float64}, dμ::Vector{Float64}, dζ::Float64, ds::Vector{Float64})

    res_w = zeros(n + m + 1 + m + n + n + m + 1 + m)
    x_ind = 1:n
    y_ind =(n).+(1:m)
    z_ind =(n+m).+(1:1)
    λ_ind =(n+m+1).+(1:m)
    ξ_ind =(n+m+1+m).+(1:n)
    η_ind =(n+m+1+m+n).+(1:n)
    μ_ind =(n+m+1+m+2n).+(1:m)
    ζ_ind =(n+m+1+m+2n+m).+(1:1)
    s_ind =(n+m+1+m+2n+m+1).+(1:m)

    r_x = @view res_w[x_ind]
    r_y = @view res_w[y_ind]
    r_z = @view res_w[z_ind]
    r_λ = @view res_w[λ_ind]
    r_ξ = @view res_w[ξ_ind]
    r_η = @view res_w[η_ind]
    r_μ = @view res_w[μ_ind]
    r_ζ = @view res_w[ζ_ind]
    r_s = @view res_w[s_ind]


    t_x = maximum(-1.01 * [dy; dz; dλ; dξ; dη; dμ; dζ; ds] ./ [y; z; λ; ξ; η; μ; ζ; s])
    t_a = maximum(-1.01 * dx ./ (x - alpha))
    t_b = maximum(1.01 * dx ./ (beta - x))
    t = 1.0 / max(1.0, t_x, t_a, t_b)
    xold = x
    yold = y
    zold = z
    λold = λ
    ξold = ξ
    ηold = η
    μold = μ
    ζold = ζ
    sold = s
    iter = 0
    res_w_norm = 2.0 * res_w_old_norm
    res_w = zeros(length(res_w))
    while res_w_norm > res_w_old_norm && iter < 100
        iter = iter + 1
        x = xold + t * dx
        y = yold + t * dy
        z = zold + t * dz
        λ = λold + t * dλ
        ξ = ξold + t * dξ
        η = ηold + t * dη
        μ = μold + t * dμ
        ζ = ζold + t * dζ
        s = sold + t * ds
        # res calc
        xa = x - alpha
        bx = beta - x
        uxinv = 1.0 ./ (u - x)
        xlinv = 1.0 ./ (x - l)
        g = p * uxinv + q * xlinv
        plam = p0 + transpose(p) * λ
        qlam = q0 + transpose(q) * λ
        dpsi = plam .* (uxinv .* uxinv) - qlam .* (xlinv .* xlinv)
        r_x .= dpsi - ξ + η
        r_y .= pr_type.ci + pr_type.di .* y - λ - μ
        r_z .= pr_type.a0 - ζ - transpose(λ) * pr_type.ai
        r_λ .= g - pr_type.ai .* z - y + s - b
        r_ξ .= ξ .* xa .- ε
        r_η .= η .* bx .- ε
        r_μ .= μ .* y .- ε
        r_ζ .= ζ * z .- ε
        r_s .= λ .* s .- ε
        res_w_norm = norm(res_w)
        t = t / 2.0
    end
    res_w_inf_norm = norm(res_w, Inf)
    return x, y, z, λ, ξ, η, μ, ζ, s, res_w_inf_norm, res_w_norm
end

function kktcheck(m::Int, n::Int, x::AbstractVector{Float64}, y::Vector{Float64},
    z::Float64, λ::Vector{Float64}, ξ::Vector{Float64},
    η::Vector{Float64}, μ::Vector{Float64}, ζ::Float64, s::Vector{Float64},
    x_min::Vector{Float64}, x_max::Vector{Float64}, fi::Vector{Float64}, df0dx::AbstractVector{Float64}, dfidx::AbstractMatrix{Float64}, pr_type::MMAProblemType)
    rex = df0dx + sum(dfidx .* transpose(λ), dims=2) - ξ + η
    rey = pr_type.ci + pr_type.di .* y - μ - λ
    rez = pr_type.a0 - ζ - dot(pr_type.ai, λ)
    relam = fi - pr_type.ai * z - y + s
    rexsi = ξ .* (x - x_min)
    reeta = η .* (x_max - x)
    remu = μ .* y
    rezet = ζ * z
    res = λ .* s
    residu1 = vcat(rex, rey, rez)
    residu2 = vcat(relam, rexsi, reeta, remu, rezet, res)
    residu = vcat(residu1, residu2)
    residunorm = sqrt(dot(residu, residu))
    residumax = norm(residu, Inf)
    return residunorm, residumax
end