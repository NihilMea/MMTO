function parse_arg(s::AbstractString, type::Type=Float64)
    if contains("\"", s)
        type = String
        ret = strip(s,['\"',' '])
        return ret
    end
    val = tryparse(type, s)
    if val !== nothing
        ret = val
    else
        ret = eval(Meta.parse(s))
    end
    return ret
end

function parse_line(line::AbstractString)
    comment_begin = findfirst('#', line)
    comment_begin = comment_begin === nothing ? length(line) : comment_begin
    args_begin = findfirst(':', line)
    args = split(line[(args_begin+1):(comment_begin-1)], " ")
    args = filter(i -> i != "", args)
    return args
end

function parse_point(vals::Vector{SubString{String}})
    ret = zeros(2)
    for (i, val) in enumerate(vals)
        t = tryparse(Float64, val)
        if t !== nothing
            ret[i] = t
        else
            ret[i] = eval(Meta.parse(val))
        end
    end
    return Point2D(ret...)
end

struct Material
    name::String
    E::Float64
    rho::Float64
    S::Float64
    V_lim::Float64
end

function parse_file(inp_file::String)
    file = open(inp_file)
    fea = nothing
    filt = nothing
    mmtop = nothing
    sol = nothing
    mats = Material[]
    x = nothing
    iter_num = 0
    prob_type = nothing
    constr = Symbol[]
    init_val = 0.5
    for line in eachline(file)
        if line == ""
            break
        elseif line[1] == '#'
            nothing
        elseif occursin("=", line)
            var, val = split(line, "=")
            var = strip(var)
            val = parse(Float64, val)
            eval(Meta.parse(join([var, "=", val])))
        elseif line[1:3] == "DS:"
            args = parse_line(line)
            dims = zeros(3)
            for (i, arg) in enumerate(split(args[1], "-"))
                dims[i] = parse_arg(arg)
            end
            el = zeros(2)
            for (i, arg) in enumerate(split(args[2], "-"))
                el[i] = parse_arg(arg)
            end
            fea = FEAProblem(dims[1], dims[2], el[1], el[2], dims[3])
        elseif line[1:3] == "BC:"
            args = parse_line(line)
            if args[1] == "L"
                type = :load
            elseif args[1] == "D"
                type = :disp
            end
            val = parse_arg(args[2])
            dir = Symbol[]
            if args[3][1] == '('
                sub_args = split(args[3][2:end-1], ",")
                for arg in sub_args
                    if arg == "u"
                        push!(dir, :u)
                    elseif arg == "v"
                        push!(dir, :v)
                    end
                end
            else
                arg = args[3][1]
                if arg == 'u'
                    push!(dir, :u)
                elseif arg == 'v'
                    push!(dir, :v)
                end
            end
            if length(args) == 4
                p = parse_point(split(args[4][2:end-1], ","))
                set_bc!(fea, type, val, dir, p)
            elseif length(args) == 5
                p1 = parse_point(split(args[4][2:end-1], ","))
                p2 = parse_point(split(args[5][2:end-1], ","))
                set_bc!(fea, type, val, dir, p1, p2)
            end
        elseif line[1:3] == "FR:"
            args = parse_line(line)
            val = parse_arg(args[1])
            filt = Filter(val, fea)
        
        elseif line[1:3] == "MT:"
            args = parse_line(line)
            vals = parse_arg.(args)
            push!(mats, Material(vals...))
        elseif line[1:3] == "SR:"
            if mmtop === nothing
                error("Problem not defined (PP should go before SR)")
            end
            args = parse_line(line)
            if args[1] == "P"
                type = :passive
            elseif args[1] == "F"
                type = :fixed
            end
            p1 = parse_point(split(strip(args[2], ['(', ')']), ","))
            p2 = parse_point(split(strip(args[3], ['(', ')']), ","))
            set_region!(mmtop, type, p1, p2)

        elseif line[1:3] == "PP:"
            args = parse_line(line)
            init_val = parse_arg(args[1], Int64)
            prob_q = parse_arg(args[2], Int64)
            prob_s = parse_arg(args[3], Int64)
            prob_p = parse_arg(args[4], Int64)
            iter_num = parse_arg(args[5], Int64)
            
            if args[6] == "COMPLIANCE"
                prob_type = :Compl_min
            elseif args[6] == "MASS"
                prob_type = :Mass_min
            end
            for arg in args[7:end]
                if arg == "STRESS"
                    push!(constr, :Stress)
                elseif arg == "VOLUME"
                    push!(constr, :Volume)
                end
            end
            mmtop = MMTOProblem(fea, [mat.E for mat in mats], prob_q,prob_s,prob_p)
        elseif line[1:4] == "PLT:"
            sol, x = solve(mmtop, prob_type, constr, filt, false, init_val, [mat.V_lim for mat in mats],
                [mat.rho for mat in mats], [mat.S for mat in mats], iter_num)
            args = parse_line(line)
            if isodd(length(args))
                folder = parse_arg(args[1])
                init_ind = 2
            else
                folder = ""
                init_ind = 1
            end
            for i in init_ind:2:length(args)
                name = parse_arg(args[i+1])
                if args[i] == "DENS"
                    fig = display_solution(:Density, sol, mmtop, x, mat_names=vcat([mat.name for mat in mats], "Пустота"))
                elseif args[i] == "STRESS_VM"
                    fig = display_solution(:VM_Stress, sol, mmtop, x)
                elseif args[i] == "STRESS_X"
                    fig = display_solution(:X_Stress, sol, mmtop, x)
                elseif args[i] == "STRESS_Y"
                    fig = display_solution(:Y_Stress, sol, mmtop, x)
                elseif args[i] == "STRESS_XY"
                    fig = display_solution(:XY_Stress, sol, mmtop, x)
                elseif args[i] == "DISPL_Y"
                    fig = display_solution(:Y_Displ, sol, mmtop, x)
                elseif args[i] == "DISPL_X"
                    fig = display_solution(:X_Displ, sol, mmtop, x)
                else 
                    error("Wrong plot type")
                end
                if !isdir(folder) && !(folder=="")
                    mkdir(folder)
                end
                Mke.save(join(["./",folder,"/",name,".png"]), fig)
            end
        else
            error(join(["Wrong input line:\n", line]))
        end
    end

    close(file)
    
    return fea, filt, mmtop, sol, x
end