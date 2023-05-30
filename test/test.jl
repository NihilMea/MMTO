using MMTO



fea, filt, mmtop, sol, x = parse_file("./test/L_beam.prob");

# display_solution(:Density, sol, mmtop, x; mat_names=[ "титан","алюминий", "пустота"])

fea, filt, mmtop, sol, x = parse_file("./test/MBB_beam.prob");

# display_solution(:Density, sol, mmtop, x; mat_names=["сталь", "алюминий", "пустота"])

