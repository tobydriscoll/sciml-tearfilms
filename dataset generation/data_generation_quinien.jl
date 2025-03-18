#opt_ode
function opt_ode(para)
    p0 = 12e-6; vw = 1.8e-5;c0 = 302;sigma_0 = 0.045;mu = 1.3e-3; d = 4.5467e-6;Df = 0.39e-9;D0 = 1.6e-9;DS = 3e-8;t_s = 18.8; 
    Pc = (p0*vw*c0)/(d/t_s)
    v_prime = para[1]*1e-6/60
    b1_prime = para[2]
    b2_prime = para[3]
    #a_prime = para[2]
    #t_s*b1_prime
    #t_s*b2_prime
    #(t_s*v_prime)/d
    function TF(u, p, t)
        h, hc = u
        c = hc/h
        b1, b2, v = p
        #a,v = p
        #g = a
        g = b1*exp(-b2*t)
        dh = -g*h + Pc*(c-1) - v
        dhc = -g*hc
        [dh, dhc]
    end
    
    tspan = (0.0, 1.0)
    u0 = [1.0, 1.0]
    prob = ODEProblem(TF, u0, tspan, [t_s*b1_prime, t_s*b2_prime,(t_s*v_prime)/d ])
    #prob = ODEProblem(TF, u0, tspan, [t_s*a_prime,(t_s*v_prime)/d ])
    sol = solve(prob, Tsit5())
    
    #t = range(0,1,length(D["I"]))
    t = ts
    
    ϵf = 1.75e+7  #Napierian extinction coefficient
    rho = 10e+3    #density of water g/L
    fcr = 0.002
    f0 = 0.00159255
    f0 = f0/fcr
    M_v = 376; # molecular weight of sodium fluorescein (approximately 376g/mol).
    f_M = (rho * fcr) / ((M_v)*100)
    Φ = ϵf * f_M * d
    hc = sol(t)[:,:][2,:]
    h = sol(t)[:,:][1,:]
    c = hc./h
    f = f0*c
    FI = ((-exp.(-Φ * f .* h)) .+ 1) ./ ((f .^ 2) .+ 1)
    I0 = 1 / FI[1]
    
    I1 = I0 * ((-exp.(-Φ * f .* h)) .+ 1) ./ ((f .^ 2) .+ 1)
    
    return sum(abs2, D["I"] .- I1)
    #return sum(abs2, test_I .- I1)
    #return I1
    end
    
    initial_guess = result.u
    initial_guess = [11.6163;-0.029;0.033]
    initial_guess = [11;0.1]
        obj3(x,p) =  opt_ode(x)
    
        ## 
        f = OptimizationFunction(obj3)
        prob = Optimization.OptimizationProblem(f, initial_guess,lb=[0.1,-1.0,0.0],ub=[40.0,5.0,2.0]) 
        result = solve(prob, NLopt.LN_NELDERMEAD(),maxeval=1000,ftol_abs=0.001,callback = my_callback)
    
        prob = Optimization.OptimizationProblem(f, initial_guess,lb=[0.0,-1.0],ub=[40.0,2.0])
        result = solve(prob,NLopt.LN_PRAXIS(),callback = my_callback)
    
    
        function my_callback(parameterI,opt_I0)
            println("Objective value: ", opt_I0)
            println("Current x: ", parameterI)
            return false
        end