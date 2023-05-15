struct Filter
    Kf::SparseMatrixCSC{Float64}
    T::SparseMatrixCSC{Float64}
    T_mean::SparseMatrixCSC{Float64}
    r::Float64
end

function Filter(r::Float64, fea::FEAProblem)
    R = r / 2 / sqrt(3)
    # Define the coordinates of the element nodes
    X = SMatrix{4,2}([-fea.el_x/2 -fea.el_y/2
        fea.el_x/2 -fea.el_y/2
        fea.el_x/2 fea.el_y/2
        -fea.el_x/2 fea.el_y/2])
    # Define the Gauss points
    gp = 1.0 / sqrt(3)
    xi_gp = SVector{4}([-gp, gp, gp, -gp])
    eta_gp = SVector{4}([-gp, -gp, gp, gp])
    # Define interpolation function s vector
    N(xi, eta) = [(1 - xi) * (1 - eta) / 4.0, (1 + xi) * (1 - eta) / 4.0, (1 + xi) * (1 + eta) / 4.0, (1 - xi) * (1 + eta) / 4.0]
    # Initialize the element stiffness matrix
    Ke = zeros(SMatrix{4,4,Float64})
    Te = zeros(SVector{4,Float64})
    Ve = 0.0
    # Loop over the Gauss points
    for (xi, eta) in zip(xi_gp, eta_gp)
        # Calculate the derivatives of the shape functions with respect to xi
        dN1_dxi = -0.25 * (1 - eta)
        dN2_dxi = 0.25 * (1 - eta)
        dN3_dxi = 0.25 * (1 + eta)
        dN4_dxi = -0.25 * (1 + eta)
        # Calculate the derivatives of the shape functions with respect to eta
        dN1_deta = -0.25 * (1 - xi)
        dN2_deta = -0.25 * (1 + xi)
        dN3_deta = 0.25 * (1 + xi)
        dN4_deta = 0.25 * (1 - xi)
        # Calculate the derivatives of the shape functions with respect to xi and eta
        dN_dxideta = SMatrix{2,4}([dN1_dxi dN2_dxi dN3_dxi dN4_dxi
            dN1_deta dN2_deta dN3_deta dN4_deta])
        # Calculate the Jacobian matrix
        J = dN_dxideta * X
        # Calculate the determinant of the Jacobian matrix
        detJ = det(J)
        # Calculate the derivatives of the shape functions with respect to x and y
        dN_dxdy = inv(J) * dN_dxideta
        N_xi_eta = N(xi, eta)
        Te += N_xi_eta * detJ
        Ke += (R^2 * transpose(dN_dxdy) * dN_dxdy + N_xi_eta * transpose(N_xi_eta)) * detJ
        Ve += detJ
    end
    Ke_size = length(Ke)
    K_nnz = Ke_size * fea.num_el
    iK = zeros(Int, K_nnz)
    jK = zeros(Int, K_nnz)
    valK = zeros(Float64, K_nnz)

    Te_size = 4
    T_nnz = Te_size * fea.num_el
    iT = zeros(Int, T_nnz)
    jT = zeros(Int, T_nnz)
    valT = zeros(Float64, T_nnz)
    valN = zeros(Float64, T_nnz)


    for i in 1:fea.num_el
        inds = Ke_size * (i - 1) .+ (1:Ke_size)
        el_dofs = vec(reshape(transpose(fea.nodes_at_elems[i, :]), :, 1))
        iK[inds] = repeat(el_dofs, inner=length(el_dofs))
        jK[inds] = repeat(el_dofs, outer=length(el_dofs))
        valK[inds] = Ke[:]

        indsT = Te_size * (i - 1) .+ (1:Te_size)
        iT[indsT] = el_dofs
        jT[indsT] = fill(i, Te_size)
        valT[indsT] = Te
        valN[indsT] = N(0,0)
    end
    Kf = sparse(iK, jK, valK, fea.num_nd, fea.num_nd)
    Kf = (Kf + transpose(Kf)) / 2
    T = sparse(iT, jT, valT, fea.num_nd, fea.num_el)
    T_mean = transpose(sparse(iT, jT, valN, fea.num_nd, fea.num_el))
    return Filter(Kf, T, T_mean, r)
end

function apply_filter(filt::Filter, x::AbstractVector{Float64})
    Kf = filt.Kf
    T = filt.T
    C = cholesky(Kf)
    ρ = filt.T_mean * (C \ (T * x))
    return ρ
end

function apply_filter(filt::Filter, x::AbstractMatrix{Float64})
    Kf = filt.Kf
    T = filt.T
    C = cholesky(Kf)
    ρ = filt.T_mean * (C \ (T * x))
    return ρ
end

function apply_projection!(β::Float64, x::AbstractMatrix{Float64}, dρdx::AbstractMatrix{Float64})
    dρdx .= β .* exp.(-β .* x) .+ exp(-β)
    x .= 1 .- exp.(-β .* x) .+ x .* exp(-β)
    # return ρ, dρdx
end