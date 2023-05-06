


function display_solution(type::Symbol, sol::FEASolution, mmtop::MMTOProblem, x::Matrix{Float64}; mat_names::Vector{String}=String[], facets::Bool=false)
    if type == :Density
        val, w = calc_mat_type(mmtop, x)
        mat_num = length(mmtop.E0)
        mat = mat_num .- argmax.(eachrow(w))
        fig = viz(mmtop.fea, mat, legend=true, legend_names=mat_names[end:-1:1], colornum=3, colormap=:RdPu_9, showfacets=facets)
    elseif type == :X_Displ
        val, w = calc_mat_type(mmtop, x)
        fea = mmtop.fea
        Te_size = 4
        T_nnz = Te_size * fea.num_el
        iT = zeros(Int, T_nnz)
        jT = zeros(Int, T_nnz)
        valN = zeros(Float64, T_nnz)
        u_n = sol.U[1:2:end]
        for i in 1:fea.num_el
            el_dofs = vec(reshape(transpose(fea.nodes_at_elems[i, :]), :, 1))
            indsT = 4 * (i - 1) .+ (1:4)
            iT[indsT] = el_dofs
            jT[indsT] = fill(i, Te_size)
            valN[indsT] .= 1 / 4
        end
        T_mean = transpose(sparse(iT, jT, valN, fea.num_nd, fea.num_el))
        u = T_mean * u_n
        u[val.<0.5] .= NaN
        fig = viz(mmtop.fea, u, colorbar=true, colornum=16, showfacets=facets, colorbar_name="мм")
    elseif type == :Y_Displ
        val, w = calc_mat_type(mmtop, x)
        fea = mmtop.fea
        Te_size = 4
        T_nnz = Te_size * fea.num_el
        iT = zeros(Int, T_nnz)
        jT = zeros(Int, T_nnz)
        valN = zeros(Float64, T_nnz)
        u_n = sol.U[2:2:end]
        for i in 1:fea.num_el
            el_dofs = vec(reshape(transpose(fea.nodes_at_elems[i, :]), :, 1))
            indsT = 4 * (i - 1) .+ (1:4)
            iT[indsT] = el_dofs
            jT[indsT] = fill(i, Te_size)
            valN[indsT] .= 1 / 4
        end
        T_mean = transpose(sparse(iT, jT, valN, fea.num_nd, fea.num_el))
        u = T_mean * u_n
        u[val.<0.5] .= NaN
        fig = viz(mmtop.fea, u, colorbar=true, colornum=16, showfacets=facets, colorbar_name="мм")
    elseif type == :VM_Stress
        val, w = calc_mat_type(mmtop, x)
        σ = calc_stress(mmtop.fea, sol, :vM)
        σ[val.<0.5] .= NaN
        σ[mmtop.fixed_elements] .= NaN
        fig = viz(mmtop.fea, σ, colorbar=true, colornum=16, showfacets=facets, colorbar_name="МПа")
    elseif type == :X_Stress
        val, w = calc_mat_type(mmtop, x)
        σ = calc_stress(mmtop.fea, sol, :x)
        σ[val.<0.5] .= NaN
        σ[mmtop.fixed_elements] .= NaN
        fig = viz(mmtop.fea, σ, colorbar=true, colornum=16, showfacets=facets, colorbar_name="МПа")
    elseif type == :X_Stress
        val, w = calc_mat_type(mmtop, x)
        σ = calc_stress(mmtop.fea, sol, :y)
        σ[val.<0.5] .= NaN
        σ[mmtop.fixed_elements] .= NaN
        fig = viz(mmtop.fea, σ, colorbar=true, colornum=16, showfacets=facets, colorbar_name="МПа")
    elseif type == :X_Stress
        val, w = calc_mat_type(mmtop, x)
        σ = calc_stress(mmtop.fea, sol, :y)
        σ[val.<0.5] .= NaN
        σ[mmtop.fixed_elements] .= NaN
        fig = viz(mmtop.fea, σ, colorbar=true, colornum=16, showfacets=facets, colorbar_name="МПа")
    else
        error("Wrong display type")
    end
end

#TODO:  improve interface
"""
* showfacets - show element borders (true or false, default is false)
* colornum - number of colors (Int, default is 12)
* colorbar - showing colorbar (true or false, default is false)
* colormap - plot colormap (default to :jet)
* legend - show legend (default to false)
* legend_names - names that show up in the legend ( vector of strings)
"""
function viz(fea::FEAProblem, values::Vector{<:Real}; showfacets::Bool=false,
    colornum::Int=12, colorbar::Bool=false, colorbar_name::String="", colormap::Union{Symbol,ColorScheme}=:jet, legend::Bool=false, legend_names::Union{Nothing,Vector{String}}=nothing)
    if colormap isa Nothing
        colmap = Mke.cgrad(:jet, colornum, categorical=true)
    elseif colormap isa Symbol
        colmap = colorschemes[colormap]
        colmap = Mke.cgrad(colormap, colornum, categorical=true)
    else
        colmap = colormap
    end

    nel = fea.num_el
    nnd = fea.num_nd

    if length(values) == nnd
        vertices = fea.nodes_coords
        faces = zeros(Int, 2 * nel, 3)
        for face in 1:nel
            faces[2*face-1, :] = fea.nodes_at_elems[face, 1:3]
            faces[2*face, :] = fea.nodes_at_elems[face, [3, 4, 1]]
        end
        to_plot = values
    elseif length(values) == nel
        elnnd = 4 * nel
        vertices = zeros(elnnd, 2)
        to_plot = zeros(elnnd)
        faces = zeros(Int, 2 * nel, 3)
        for face in 1:nel
            ids = fea.nodes_at_elems[face, :]
            vertices[4*face-3:4*face, :] = fea.nodes_coords[ids, :]
            faces[2*face-1, :] = 4*face-3:4*face-1
            faces[2*face, :] = [4 * face - 1, 4 * face, 4 * face - 3]
            to_plot[4*face-3:4*face] .= values[face]
        end
    else
        error("Wrong number of values provided")
    end
    lims = extrema(filter(!isnan, to_plot), init=(-0.0001, 0.0001))
    colors = [val !== nothing ? get(colmap, (val - lims[1]) / (lims[2] - lims[1])) : val for val in to_plot]
    fig = Mke.Figure()
    ax, plt = Mke.mesh(fig[1, 1], vertices, faces, color=colors, colorrange=lims, colormap=colmap, shading=false, nan_color=:white)
    if legend
        leg_colors = [get(colmap, (val - lims[1]) / (lims[2] - lims[1])) for val in sort(filter(x -> !isnothing(x), unique(to_plot)))]
        elems = [Mke.PolyElement(color=color, strokecolor=:transparent) for color in leg_colors]
        Mke.Legend(fig[1, 2], elems, legend_names)
    end
    if showfacets
        xs = zeros(8 * nel)
        ys = zeros(8 * nel)
        vertices = fea.nodes_coords
        el_nodes = fea.nodes_at_elems
        for elem in 1:nel
            nodes = el_nodes[elem, :]
            xs[8*elem-7] = vertices[nodes[1], 1]
            xs[8*elem-6] = vertices[nodes[2], 1]
            xs[8*elem-5] = vertices[nodes[2], 1]
            xs[8*elem-4] = vertices[nodes[3], 1]
            xs[8*elem-3] = vertices[nodes[3], 1]
            xs[8*elem-2] = vertices[nodes[4], 1]
            xs[8*elem-1] = vertices[nodes[4], 1]
            xs[8*elem] = vertices[nodes[1], 1]

            ys[8*elem-7] = vertices[nodes[1], 2]
            ys[8*elem-6] = vertices[nodes[2], 2]
            ys[8*elem-5] = vertices[nodes[2], 2]
            ys[8*elem-4] = vertices[nodes[3], 2]
            ys[8*elem-3] = vertices[nodes[3], 2]
            ys[8*elem-2] = vertices[nodes[4], 2]
            ys[8*elem-1] = vertices[nodes[4], 2]
            ys[8*elem] = vertices[nodes[1], 2]

        end
        Mke.linesegments!(ax, xs, ys, color=:black, linewidth=0.5)
    end
    if colorbar
        Mke.Colorbar(fig[1, 2], colormap=colmap, limits=lims, ticks=Mke.LinearTicks(colornum + 3), label=colorbar_name)
    end
    return fig
end