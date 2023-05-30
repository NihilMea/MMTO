function parse_arg(s::AbstractString, type::Type=Float64)
    if contains("\"", s)
        ret = strip(s, ['\"', ' '])
        return ret
    end
    if contains("[", s) && contains("}", s)
        type = Float64
        ret = strip(s, ['[', ']', ' '])
        if contains(" ", ret)
            ret = tryparse.(type, split(ret, " "))
            return ret
        elseif contains(",", ret)
            ret = tryparse.(type, split(ret, ","))
            return ret
        else
            error(join(["Something wrong with input line: ", s]))
        end
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
    # args_begin = findfirst(':', line)
    args = split(line[1:(comment_begin-1)], " ")
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
    out_iter = 0
    for line in eachline(file)
        if line == ""
            continue
        elseif line[1] == '#'
            continue
        elseif occursin("=", line)
            var, val = split(line, "=")
            var = strip(var)
            val = parse(Float64, val)
            eval(Meta.parse(join([var, "=", val])))
            continue
        end
        args = parse_line(line)
        if args[1] == "DS:"
            dims = zeros(3)
            for (i, arg) in enumerate(split(args[2], "-"))
                dims[i] = parse_arg(arg)
            end
            el = zeros(2)
            for (i, arg) in enumerate(split(args[3], "-"))
                el[i] = parse_arg(arg)
            end
            fea = FEAProblem(dims[1], dims[2], el[1], el[2], dims[3])
        elseif args[1] == "BC:"
            if args[2] == "L"
                type = :load
            elseif args[2] == "D"
                type = :disp
            end
            val = parse_arg(args[3])
            dir = Symbol[]
            if args[4][1] == '('
                sub_args = split(args[4][2:end-1], ",")
                for arg in sub_args
                    if arg == "u"
                        push!(dir, :u)
                    elseif arg == "v"
                        push!(dir, :v)
                    end
                end
            else
                arg = args[4][1]
                if arg == 'u'
                    push!(dir, :u)
                elseif arg == 'v'
                    push!(dir, :v)
                end
            end
            if length(args) == 5
                p = parse_point(split(args[5][2:end-1], ","))
                set_bc!(fea, type, val, dir, p)
            elseif length(args) == 6
                p1 = parse_point(split(args[5][2:end-1], ","))
                p2 = parse_point(split(args[6][2:end-1], ","))
                set_bc!(fea, type, val, dir, p1, p2)
            end
        elseif args[1] == "FR:"
            val = parse_arg(args[2])
            filt = Filter(val, fea)

        elseif args[1] == "MT:"
            vals = parse_arg.(args[2:end])
            push!(mats, Material(vals...))
        elseif args[1] == "SR:"
            if mmtop === nothing
                error("Problem not defined (PP should go before SR)")
            end
            if args[2] == "P"
                type = :passive
            elseif args[2] == "F"
                type = :fixed
            end
            p1 = parse_point(split(strip(args[3], ['(', ')']), ","))
            p2 = parse_point(split(strip(args[4], ['(', ')']), ","))
            set_region!(mmtop, type, p1, p2)

        elseif args[1] == "PP:"
            init_val = parse_arg(args[2], Int64)
            prob_q = parse_arg(args[3], Int64)
            prob_s = parse_arg(args[4], Int64)
            prob_p = parse_arg(args[5], Int64)
            iter_num = parse_arg(args[6], Int64)

            if args[7] == "COMPLIANCE"
                prob_type = :Compl_min
            elseif args[7] == "MASS"
                prob_type = :Mass_min
            end
            for arg in args[8:end]
                if arg == "STRESS"
                    push!(constr, :Stress)
                elseif arg == "VOLUME"
                    push!(constr, :Volume)
                end
            end
            mmtop = MMTOProblem(fea, [mat.E for mat in mats], prob_q, prob_s, prob_p)
        elseif args[1] == "PLT:"
            sol, x, f0_arr,fi, vol, mass, out_iter = solve(mmtop, prob_type, constr, filt, false, init_val, [mat.V_lim for mat in mats], [mat.dens for mat in mats], [mat.S for mat in mats], iter_num)
            jldsave("results.jld2"; mmtop, fea, filt, sol, x, f0_arr, fi, vol, mass, out_iter)
            if isodd(length(args) - 1)
                folder = parse_arg(args[2])
                init_ind = 3
            else
                folder = ""
                init_ind = 2
            end
            for i in init_ind:2:(length(args))
                name = parse_arg(args[i+1])
                if args[i] == "DENS"
                    fig = display_solution(:Density, sol, mmtop, x, mat_names=vcat([mat.name for mat in mats], ""))
                elseif args[i] == "STRESS_VM"
                    fig = display_solution(:VM_Stress, sol, mmtop, x, s_max=[mat.S for mat in mats])
                elseif args[i] == "STRESS_MS"
                    fig = display_solution(:MS_Stress, sol, mmtop, x, s_max=[mat.S for mat in mats])
                elseif args[i] == "STRESS_X"
                    fig = display_solution(:X_Stress, sol, mmtop, x, s_max=[mat.S for mat in mats])
                elseif args[i] == "STRESS_Y"
                    fig = display_solution(:Y_Stress, sol, mmtop, x, s_max=[mat.S for mat in mats])
                elseif args[i] == "STRESS_XY"
                    fig = display_solution(:XY_Stress, sol, mmtop, x, s_max=[mat.S for mat in mats])
                elseif args[i] == "DISPL_Y"
                    fig = display_solution(:Y_Displ, sol, mmtop, x)
                elseif args[i] == "DISPL_X"
                    fig = display_solution(:X_Displ, sol, mmtop, x)
                else
                    error("Wrong plot type")
                end
                if !isdir(folder) && !(folder == "")
                    mkdir(folder)
                end
                Mke.save(join(["./", folder, "/", name, ".png"]), fig)
            end
        else
            error(join(["Wrong input line:\n", line]))
        end
    end

    close(file)

    return fea, filt, mmtop, sol, x, out_iter
end

function load_solution(load_file::String)
    jldopen(load_file, "w") do file
        # file["bigdata"] = randn(5)
    end
end