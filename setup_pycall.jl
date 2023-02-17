using Pkg
Pkg.add("PyCall")
ENV["PYTHON"] = raw"/Users/federicoclaudi/miniconda3/envs/gld/bin/python"
Pkg.build("PyCall")