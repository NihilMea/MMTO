Point2D = SVector{2,Float64}

"""
Structure to hold 2D plane stress FEA problem information
* num_el - number of elements
* num_nd - number of nodes
* num_dof - number of degrees of freedom
* nodes_coords - nodes coordinates
* dofs_at_nodes - degrees of freedom at nodes
* nodes_at_elems - nodes at element
* elems_coords - element centers
"""
struct FEAProblem
    num_el::Int # number of elements
    num_nd::Int # number of nodes
    num_dof::Int # number of degrees of freedom
    nodes_coords::Array{Float64,2}  # nodes coordinates
    dofs_at_nodes::Array{Int,2}     # degrees of freedom at nodes
    nodes_at_elems::Array{Int,2}    # nodes at element
    elems_coords::Array{Float64,2}  # element centers

    el_x::Float64
    el_y::Float64
    Ve::Float64 # element volume

    Ke0::SMatrix{8,8,Float64} # initial element stiffness matrix
    Bσ::SMatrix{3,8,Float64} # strain-displacement matrix at element center
    D0::SMatrix{3,3,Float64} # elastic matrix 

    E::Array{Float64,1}    # array of element elastic matrices
    # bc values
    load_values::Vector{Float64}
    load_dofs::Vector{Int}
    disp_values::Vector{Float64}
    disp_dofs::Vector{Int}
end



"""
Constructor for FEAProblem structure
* a - x axis length
* b - y axis length
* el_x - element length along x axis
* el_y - element length along y axis
"""
function FEAProblem(a::Float64, b::Float64, el_x::Float64, el_y::Float64, t::Float64)
    num_el_x = round(Int, a / el_x)
    num_el_y = round(Int, b / el_y)
    num_el = num_el_x * num_el_y
    num_nd_x = num_el_x + 1
    num_nd_y = num_el_y + 1
    num_nd = num_nd_x * num_nd_y
    num_dof = 2 * num_nd
    nodes_coords = zeros(Float64, num_nd, 2)
    dofs_at_nodes = zeros(Int, num_nd, 2)
    # calc node positions
    for i in 1:num_nd_x
        for j in 1:num_nd_y
            nd_id = (i - 1) * num_nd_y + j
            nodes_coords[nd_id, 1] = (i - 1) * el_x
            nodes_coords[nd_id, 2] = (j - 1) * el_y
            dofs_at_nodes[nd_id, 1] = 2 * nd_id - 1
            dofs_at_nodes[nd_id, 2] = 2 * nd_id
        end
    end
    #create elements and nodes at elements
    nodes_at_elems = zeros(Int, num_el, 4)
    elems_coords = zeros(Float64, num_el, 2)
    for i in 1:num_el_x
        for j in 1:num_el_y
            el_id = (i - 1) * num_el_y + j
            nodes_at_elems[el_id, 1] = el_id + (i - 1)
            nodes_at_elems[el_id, 2] = el_id + num_nd_y + (i - 1)
            nodes_at_elems[el_id, 3] = el_id + num_nd_y + (i - 1) + 1
            nodes_at_elems[el_id, 4] = el_id + (i - 1) + 1
            elems_coords[el_id, 1] = sum(nodes_coords[nodes_at_elems[el_id, :], 1]) / 4.0
            elems_coords[el_id, 2] = sum(nodes_coords[nodes_at_elems[el_id, :], 2]) / 4.0
        end
    end
    Ke0, Bσ, D0 = calc_Ke(el_x, el_y, t)
    E = zeros(num_el)
    load_values = Float64[]
    load_dofs = Int[]
    disp_values = Float64[]
    disp_dofs = Int[]

    return FEAProblem(num_el, num_nd, num_dof, nodes_coords, dofs_at_nodes, nodes_at_elems, elems_coords, el_x, el_y,
        el_x * el_y * t, Ke0, Bσ, D0, E, load_values, load_dofs, disp_values, disp_dofs)
end

"""
Building initial element stiffness matrix
"""
function calc_Ke(dx::Float64, dy::Float64, t::Float64)
    # Define the coordinates of the element nodes
    X = SMatrix{4,2}([0.0 0.0
        dx 0.0
        dx dy
        0.0 dy])
    # Define the Gauss points
    gp = 1.0 / sqrt(3.0)
    xi = SVector{4}([-gp, gp, gp, -gp])
    eta = SVector{4}([-gp, -gp, gp, gp])
    # Define the material properties
    mu = 0.3
    D0 = SMatrix{3,3}(1.0 / (1 - mu * mu) * [1.0 mu 0.0
        mu 1.0 0.0
        0.0 0.0 (1-mu)/2])
    # Initialize the element stiffness matrix
    Ke0 = zeros(SMatrix{8,8,Float64})
    # Loop over the Gauss points
    for i in eachindex(xi)
        # Calculate the derivatives of the shape functions with respect to xi
        dN1_dxi = -0.25 * (1 - eta[i])
        dN2_dxi = 0.25 * (1 - eta[i])
        dN3_dxi = 0.25 * (1 + eta[i])
        dN4_dxi = -0.25 * (1 + eta[i])
        # Calculate the derivatives of the shape functions with respect to eta
        dN1_deta = -0.25 * (1 - xi[i])
        dN2_deta = -0.25 * (1 + xi[i])
        dN3_deta = 0.25 * (1 + xi[i])
        dN4_deta = 0.25 * (1 - xi[i])
        # Calculate the derivatives of the shape functions with respect to xi and eta
        dN_dxideta = SMatrix{2,4}([dN1_dxi dN2_dxi dN3_dxi dN4_dxi
            dN1_deta dN2_deta dN3_deta dN4_deta])
        # Calculate the Jacobian matrix
        J = dN_dxideta * X
        # Calculate the determinant of the Jacobian matrix
        detJ = det(J)
        # Calculate the derivatives of the shape functions with respect to x and y
        dN_dxdy = inv(J) * dN_dxideta
        # Calculate the strain-displacement matrix
        B = SMatrix{3,8}([dN_dxdy[1, 1] 0.0 dN_dxdy[1, 2] 0.0 dN_dxdy[1, 3] 0.0 dN_dxdy[1, 4] 0.0
            0.0 dN_dxdy[2, 1] 0.0 dN_dxdy[2, 2] 0.0 dN_dxdy[2, 3] 0.0 dN_dxdy[2, 4]
            dN_dxdy[2, 1] dN_dxdy[1, 1] dN_dxdy[2, 2] dN_dxdy[1, 2] dN_dxdy[2, 3] dN_dxdy[1, 3] dN_dxdy[2, 4] dN_dxdy[1, 4]])
        # Calculate the element stiffness matrix
        Ke0 += transpose(B) * D0 * B * detJ * t
    end
    # Calculate the derivatives of the shape functions with respect to xi and eta
    dN_dxideta = SMatrix{2,4}([-0.25 0.25 0.25 -0.25
        -0.25 -0.25 0.25 0.25])
    # Calculate the Jacobian matrix
    J = dN_dxideta * X
    # Calculate the derivatives of the shape functions with respect to x and y
    dN_dxdy = inv(J) * dN_dxideta
    # Calculate the strain-displacement matrix at element center
    Bσ = SMatrix{3,8}([dN_dxdy[1, 1] 0.0 dN_dxdy[1, 2] 0.0 dN_dxdy[1, 3] 0.0 dN_dxdy[1, 4] 0.0
        0.0 dN_dxdy[2, 1] 0.0 dN_dxdy[2, 2] 0.0 dN_dxdy[2, 3] 0.0 dN_dxdy[2, 4]
        dN_dxdy[2, 1] dN_dxdy[1, 1] dN_dxdy[2, 2] dN_dxdy[1, 2] dN_dxdy[2, 3] dN_dxdy[1, 3] dN_dxdy[2, 4] dN_dxdy[1, 4]])
    # Return the element stiffness matrix, strain-displacement matrix, and material properties
    return Ke0, Bσ, D0
end

"""
Function that checks if BC is already set at selected DOF
"""
function check_bc(fea::FEAProblem, type::Symbol, dof::Int)
    if type == :load
        return findfirst(isequal(dof), fea.load_dofs) !== nothing
    elseif type == :disp
        return findfirst(isequal(dof), fea.disp_dofs) !== nothing
    else
        error("WrongBCType")
    end
end

"""
Function that return dof number for node id and direction
"""
function dof(nd_id::Int, dir::Symbol)
    n_dof = 2
    if dir == :u || dir == :x
        return n_dof * nd_id - 1
    elseif dir == :v || dir == :y
        return n_dof * nd_id
    else
        error("WrongDirectionForBCApplication")
    end
end

"""
Set boundary condition for nodes in rectangular region with
p1 and p2 as diagonal corners
type could be :load or :disp
"""
function set_bc!(fea::FEAProblem, type::Symbol, value::Float64, direction::Union{Symbol,Vector{Symbol}}, p1::Point2D, p2::Point2D)
    if findfirst(isequal(type), [:load, :disp]) === nothing
        error("WrongBCType")
    end

    if typeof(direction) !== Vector{Symbol}
        direction = [direction]
    end

    x1 = min(p1[1], p2[1])
    x2 = max(p1[1], p2[1])
    y1 = min(p1[2], p2[2])
    y2 = max(p1[2], p2[2])

    for (nd_id, nd) in enumerate(eachrow(fea.nodes_coords))
        if (nd[1] >= x1 && nd[1] <= x2 && nd[2] >= y1 && nd[2] <= y2)
            for dir in direction
                DOF = dof(nd_id, dir)
                if type == :load && !check_bc(fea, type, DOF)
                    push!(fea.load_values, value)
                    push!(fea.load_dofs, DOF)
                elseif type == :disp && !check_bc(fea, type, DOF)
                    push!(fea.disp_values, value)
                    push!(fea.disp_dofs, DOF)
                end
            end
        end
    end
end

"""
Set boundary condition for node near point
"""
function set_bc!(fea::FEAProblem, type::Symbol, value::Float64, direction::Union{Symbol,Vector{Symbol}}, p1::Point2D)
    if findfirst(isequal(type), [:load, :disp]) === nothing
        error("WrongBCType")
    end

    nearest_nd_id = 0
    dist2 = Inf
    for (nd_id, nd) in enumerate(eachrow(fea.nodes_coords))
        dist2_temp = sum((nd - p1) .* (nd - p1))
        if dist2_temp < dist2
            nearest_nd_id = nd_id
            dist2 = dist2_temp
        end
    end

    if !(typeof(direction) == Vector{Symbol})
        direction = [direction]
    end

    for dir in direction
        DOF = dof(nearest_nd_id, dir)
        if type == :load && !check_bc(fea, type, DOF)
            push!(fea.load_values, value)
            push!(fea.load_dofs, DOF)
        elseif type == :disp && !check_bc(fea, type, DOF)
            push!(fea.disp_values, value)
            push!(fea.disp_dofs, DOF)
        end
    end
end

"""
Assembling FEA global matrix
"""
function assemble_K(fea::FEAProblem)
    Ke_size = length(fea.Ke0)
    nnz_num = Ke_size * fea.num_el
    iK = zeros(Int, nnz_num)
    jK = zeros(Int, nnz_num)
    valK = zeros(Float64, nnz_num)

    for i in 1:fea.num_el
        Ke = fea.E[i] * fea.Ke0
        inds = Ke_size * (i - 1) .+ (1:Ke_size)
        el_dofs = vec(reshape(transpose(fea.dofs_at_nodes[fea.nodes_at_elems[i, :], :]), :, 1))
        iK[inds] = repeat(el_dofs, inner=length(el_dofs))
        jK[inds] = repeat(el_dofs, outer=length(el_dofs))
        valK[inds] = Ke[:]
    end
    num_dof = length(fea.dofs_at_nodes)
    K = sparse(iK, jK, valK, num_dof, num_dof)
    return (K + transpose(K)) / 2.0
end

"""
Assembling FEA external force
"""
function ext_force(fea::FEAProblem)
    iF = zeros(Int, length(fea.load_dofs))
    valF = zeros(Float64, length(fea.load_values))
    for i in 1:length(fea.load_dofs)
        iF[i] = fea.load_dofs[i]
        valF[i] = fea.load_values[i]
    end
    return sparsevec(iF, valF, fea.num_dof)
end

function elem_dofs(fea::FEAProblem, el_id::Int)
    return vec(reshape(transpose(fea.dofs_at_nodes[fea.nodes_at_elems[el_id, :], :]), :, 1))
end
"""
Structure for FEA Solution
"""
struct FEASolution
    K::SparseMatrixCSC{Float64}
    F::SparseVector{Float64}
    U::Vector{Float64}
end

"""
Calculation of element stress
"""
function calc_stress(fea::FEAProblem, sol::FEASolution, type::Symbol)
    S = zeros(Float64, fea.num_el, 3)
    for i in 1:fea.num_el
        el_dofs = elem_dofs(fea, i)
        S[i, :] = fea.E[i] * fea.D0 * fea.Bσ * sol.U[el_dofs]
    end
    T = [1.0 -0.5 0.0
        -0.5 1.0 0.0
        0.0 0.0 3.0]
    if type == :x
        σ = S[:, 1]
    elseif type == :y
        σ = S[:, 2]
    elseif type == :xy
        σ = S[:, 3]
    elseif type == :vM
        σ = [sqrt(transpose(s) * T * s) for s in eachrow(S)]
    else
        error("Wrong output type provided")
    end
    return σ
end

"""
Solver for FEA problem
* fea - FEAProblem structure
"""
function solve(fea::FEAProblem)
    K = assemble_K(fea)
    F = ext_force(fea)
    # apply displacement
    for (dof, val) in zip(fea.disp_dofs, fea.disp_values)
        K[dof, dof] += 1e20
        F[dof] = val * K[dof, dof]
    end
    U = vec(Matrix(cholesky(K) \ F))
    return FEASolution(K, F, U)
end

"""
Solver for FEA problem
* fea - FEAProblem structure
"""
function solve!(fea::FEAProblem, sol::FEASolution)
    sol.K .= assemble_K(fea)
    sol.F .= ext_force(fea)
    # apply displacement
    for (dof, val) in zip(fea.disp_dofs, fea.disp_values)
        sol.K[dof, dof] += 1e20
        sol.F[dof] = val * sol.K[dof, dof]
    end
    sol.U .= vec(Matrix(cholesky(sol.K) \ sol.F))
    return sol
end