import importlib, colorama
colorama.init()

mcl = importlib.import_module("mcl")

while True:
    cmd = input(f"{colorama.Fore.CYAN}debug {colorama.Fore.YELLOW}>{colorama.Fore.RESET} ")
    if cmd == "exit":
        break
    
    mcl = importlib.import_module("mcl")

    result, error = mcl.compile("<stdin>", cmd)
    
    if error: print(error)
    else: print(result)