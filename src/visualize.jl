using ColorSchemes
import GLMakie as Mke

export viz, display_solution

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
    colornum::Int=12, colorbar::Bool=false, colormap::Union{Symbol,ColorScheme}=:jet, legend::Bool=false, legend_names::Union{Nothing,Vector{String}}=nothing)
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

            ys[8*elem.id-7] = vertices[nodes[1], 2]
            ys[8*elem.id-6] = vertices[nodes[2], 2]
            ys[8*elem.id-5] = vertices[nodes[2], 2]
            ys[8*elem.id-4] = vertices[nodes[3], 2]
            ys[8*elem.id-3] = vertices[nodes[3], 2]
            ys[8*elem.id-2] = vertices[nodes[4], 2]
            ys[8*elem.id-1] = vertices[nodes[4], 2]
            ys[8*elem.id] = vertices[nodes[1], 2]

        end
        Mke.linesegments!(ax, xs, ys, color=:black, linewidth=0.5)
    end
    if colorbar
        Mke.Colorbar(fig[1, 2], colormap=colmap, limits=lims, ticks=Mke.LinearTicks(colornum + 3))
    end
    return fig
end