function display_solution(type::Symbol, sol::FEASolution, mmtop::MMTOProblem, x::Matrix{Float64}; s_max::Vector{Float64}=Float64[],mat_names::Vector{String} = String[], facets::Bool=false)
    if type == :Density
        val, w = calc_mat_type(mmtop, x)
        mat_num = length(mmtop.E0)
        mat = mat_num .- argmax.(eachrow(w))
        # mat[w[:,1].<0.55 .&& w[:,2] .< 0.55] .= 0
        cmap = ColorScheme([Colors.RGB(1.0, 1.0, 1.0), Colors.RGB(0.0, 0.0, 0.0)])
        fig = viz(mmtop.fea, mat, legend=true, legend_names=mat_names[end:-1:1], colornum=3, colormap=cmap, showfacets=facets)
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
        mat_num = length(mmtop.E0)
        mat = argmax.(eachrow(w))
        σ = calc_stress(mmtop, sol, x, s_max, :vM)
        σ[mat .== mat_num ] .= NaN
        σ[mmtop.fixed_elements] .= NaN
        fig = viz(mmtop.fea, σ, colorbar=true, colornum=16, showfacets=facets, colorbar_name="МПа")
    elseif type == :X_Stress
        val, w = calc_mat_type(mmtop, x)
        mat_num = length(mmtop.E0)
        mat = argmax.(eachrow(w))
        σ = calc_stress(mmtop, sol, x, s_max, :x)
        σ[mat .== mat_num] .= NaN
        σ[mmtop.fixed_elements] .= NaN
        fig = viz(mmtop.fea, σ, colorbar=true, colornum=16, showfacets=facets, colorbar_name="МПа")
    elseif type == :Y_Stress
        val, w = calc_mat_type(mmtop, x)
        mat_num = length(mmtop.E0)
        mat = argmax.(eachrow(w))
        σ = calc_stress(mmtop, sol, x, s_max, :y)
        σ[mat .== mat_num] .= NaN
        σ[mmtop.fixed_elements] .= NaN
        fig = viz(mmtop.fea, σ, colorbar=true, colornum=16, showfacets=facets, colorbar_name="МПа")
    elseif type == :XY_Stress
        val, w = calc_mat_type(mmtop, x)
        mat_num = length(mmtop.E0)
        mat = argmax.(eachrow(w))
        σ = calc_stress(mmtop, sol, x, s_max, :xy)
        σ[mat .== mat_num] .= NaN
        σ[mmtop.fixed_elements] .= NaN
        fig = viz(mmtop.fea, σ, colorbar=true, colornum=16, showfacets=facets, colorbar_name="МПа")
    elseif type == :MS_Stress
        val, w = calc_mat_type(mmtop, x)
        mat_num = length(mmtop.E0)
        mat = argmax.(eachrow(w))
        σ = calc_stress(mmtop, sol, x, s_max, :MS)
        σ[mat .== mat_num] .= NaN
        σ[mmtop.fixed_elements] .= NaN
        fig = viz(mmtop.fea, σ, colorbar=true, colornum=16, showfacets=facets, colorbar_name="")
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
    dx = maximum(vertices[:, 1]) - minimum(vertices[:, 1])
    dy = maximum(vertices[:, 2]) - minimum(vertices[:, 2])
    if dx / dy > 1.0
        res = (720, 720 * (dy + 16) / dx)
    else
        res = (720 * (dx + 16) / dy, 720)
    end
    fig = Mke.Figure(resolution=res)
    ax, msh = Mke.mesh(fig[1, 1], vertices, faces, color=colors, colorrange=lims, colormap=colmap, shading=false, nan_color=:white)
    ax.aspect = Mke.AxisAspect(dx / dy)
    if legend
        mats = 1:length(legend_names)
        b = (lims[2] - lims[1]) / (mats[end] - mats[1])
        k = lims[1] - b * mats[1]
        leg_colors = [get(colmap, (round(Int, b * val + k) - lims[1]) / (lims[2] - lims[1])) for val in mats]
        elems = [Mke.PolyElement(color=color, strokecolor=:transparent) for color in leg_colors]
        if dx / dy > 1.0
            Mke.Legend(fig[2, 1], elems, legend_names, orientation=:horizontal, tellwidth=false, tellheight=true,framevisible = false)
        else
            Mke.Legend(fig[1, 2], elems, legend_names,framevisible = false)
        end

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
        if dx / dy > 1.0
            Mke.Colorbar(fig[2, 1], colormap=colmap, limits=lims, ticks=Mke.LinearTicks(colornum + 3), label=colorbar_name, vertical=false, flipaxis=false, alignmode=Mke.Outside(20))
        else
            Mke.Colorbar(fig[1, 2], colormap=colmap, limits=lims, ticks=Mke.LinearTicks(colornum + 3), label=colorbar_name, alignmode=Mke.Outside(20))
        end
    end
    Mke.resize_to_layout!(fig)
    return fig
end