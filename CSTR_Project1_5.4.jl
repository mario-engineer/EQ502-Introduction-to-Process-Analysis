#Process Parameters
ρ = 880.0               #kg/m^3
Cp = 1750.0             #J/kg.K
Cpj = 4180.0
Ko = 820000.0           #m^3//mol.min
Ea = 48500.0            #J/mol
ΔHr = -72800.0          #J/mol
U = 680.0               #J/min*m^3*K
A = 5.0                 #m^2
V = 40.0                #m^3
R = 8.314               #J/mol.K
Qj = 0.01               #m^3/min
Q = 3.0                 #m^3/min
ρj = 1000.0             #kg/ m^3
Cain = 200.0            #mol/m^3
Cbin = 200.0            #mol/m^3
Ccin = 0.0              #mol/m^3
Tin = 300.0             #Kelvin
Tjin = 280.0            #Kelvin
#dx = 0.0                #Newton Raphson variable

Kin = Ko*exp(-Ea/(R*Tin))

#System of Equations
f1(Ca, Cb, Cc, T, Tj, K) = Q*(Cain-Ca)-K*Ca*Cb*V
f2(Ca, Cb, Cc, T, Tj, K) = Q*(Cbin-Cb)-K*Ca*Cb*V
f3(Ca, Cb, Cc, T, Tj, K) = Q*(Ccin-Cc)+K*Ca*Cb*V
f4(Ca, Cb, Cc, T, Tj, K) = Q*ρ*Cp*(Tin-T)+U*A*(Tj-T)+K*Ca*Cb*V*(-ΔHr)
f5(Ca, Cb, Cc, T, Tj, K) = Qj*ρj*Cpj*(Tjin-Tj)+U*A*(T-Tj)
f6(Ca, Cb, Cc, T, Tj, K) = Ko*exp(-Ea/(R*T))-K

#Defining the Matrix Function
F(Ca, Cb, Cc, T, Tj, K) = [f1(Ca, Cb, Cc, T, Tj, K);
                           f2(Ca, Cb, Cc, T, Tj, K);
                           f3(Ca, Cb, Cc, T, Tj, K);
                           f4(Ca, Cb, Cc, T, Tj, K);
                           f5(Ca, Cb, Cc, T, Tj, K);
                           f6(Ca, Cb, Cc, T, Tj, K)]

using ForwardDiff, LinearAlgebra

#System of gradients
∇f1(Ca, Cb, Cc, T, Tj, K) = ForwardDiff.gradient(Ca -> f1(Ca...), [Ca; Cb; Cc; T; Tj; K])
∇f2(Ca, Cb, Cc, T, Tj, K) = ForwardDiff.gradient(Ca -> f2(Ca...), [Ca; Cb; Cc; T; Tj; K])
∇f3(Ca, Cb, Cc, T, Tj, K) = ForwardDiff.gradient(Ca -> f3(Ca...), [Ca; Cb; Cc; T; Tj; K])
∇f4(Ca, Cb, Cc, T, Tj, K) = ForwardDiff.gradient(Ca -> f4(Ca...), [Ca; Cb; Cc; T; Tj; K])
∇f5(Ca, Cb, Cc, T, Tj, K) = ForwardDiff.gradient(Ca -> f5(Ca...), [Ca; Cb; Cc; T; Tj; K])
∇f6(Ca, Cb, Cc, T, Tj, K) = ForwardDiff.gradient(Ca -> f6(Ca...), [Ca; Cb; Cc; T; Tj; K])

#Defining the Jacobian
J(Ca, Cb, Cc, T, Tj, K) = [∇f1(Ca, Cb, Cc, T, Tj, K)';
                           ∇f2(Ca, Cb, Cc, T, Tj, K)';
                           ∇f3(Ca, Cb, Cc, T, Tj, K)';
                           ∇f4(Ca, Cb, Cc, T, Tj, K)';
                           ∇f5(Ca, Cb, Cc, T, Tj, K)';
                           ∇f6(Ca, Cb, Cc, T, Tj, K)']

X = [Cain, Cbin, Ccin, Tin, Tjin, Kin]

while norm(F(X[1], X[2], X[3], X[4], X[5], X[6]))>1e-3
    
    δ = J(X[1], X[2], X[3], X[4], X[5], X[6])\-F(X[1], X[2], X[3], X[4], X[5], X[6])
    global X = X + δ

end

println("Ca=", round(X[1], digits=4), " mol/m^3")
println("Cb=", round(X[2], digits=4), " mol/m^3")
println("Cc=", round(X[3], digits=4), " mol/m^3")
println("T=", round(X[4], digits=4), " K")
println("Tj=", round(X[5], digits=4), " K")

#= Output:
Ca=49.4656 mol/m^3
Cb=49.4656 mol/m^3
Cc=150.5344 mol/m^3
T=307.0977 K
Tj=282.0383 K
=#
