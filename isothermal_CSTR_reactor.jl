#= Considers an isothermal CSTR from its startup to a steady state.
Reactants A and B produce C and D according to the irreversible reaction:
A + B > C + D, in which the rate constant k is equal to 0.855 l/mol.s.
The reactor was initially fed with a solution containing product D at a
concentration of 0.8 mol/L (CD0 ¼ 0.8 mol/L).
A solution with reactants A and B was added to the reactor at
a flow rate of 5 L/min and at concentrations of
A and B equal to 0.7 and 0.4 mol/L, respectively
(CA1 ¼ 0.7 and CB1 ¼ 0.4 mol/L).
The outlet volumetric flow rate is also 5 L/L,
and the volume of liquid inside the
reactor remains equal to 40 L over the entire reaction.
Find the ODE system that represents the concentrations of A, B, C, and D
in the CSTR from startup to a steady state.
Define all initial conditions to solve the equations.
Create hypotheses for your model if needed. FOGLER (1999)

Use V = 40L

From mass balance in the steady state we have:

(F/V)*(Ca1 - Ca) - K(Ca*Cb) = 0
(F/V)*(Cb1 - Cb) - K(Ca*Cb) = 0
(F/V)*(Cc1 - Cc) + K(Ca*Cb) = 0
(F/V)*(Cd1 - Cd) + K(Ca*Cb) = 0
=#

using ForwardDiff, LinearAlgebra

#Process Parameters
Feed = 5.0/60.0 #L/s
V = 40.0 #L
k = 0.855 #L/mol.s

#System of Equations
f1(Ca, Cb, Cc, Cd) = (Feed/V)*(0.7-Ca)-(k*Ca*Cb)
f2(Ca, Cb, Cc, Cd) = (Feed/V)*(0.4-Cb)-(k*Ca*Cb)
f3(Ca, Cb, Cc, Cd) = (Feed/V)*(0.0-Cc)+(k*Ca*Cb)
f4(Ca, Cb, Cc, Cd) = (Feed/V)*(0.0-Cd)+(k*Ca*Cb)

#System of gradients
∇f1(Ca, Cb, Cc, Cd) = ForwardDiff.gradient(Ca -> f1(Ca...), [Ca; Cb; Cc; Cd])
∇f2(Ca, Cb, Cc, Cd) = ForwardDiff.gradient(Ca -> f2(Ca...), [Ca; Cb; Cc; Cd])
∇f3(Ca, Cb, Cc, Cd) = ForwardDiff.gradient(Ca -> f3(Ca...), [Ca; Cb; Cc; Cd])
∇f4(Ca, Cb, Cc, Cd) = ForwardDiff.gradient(Ca -> f4(Ca...), [Ca; Cb; Cc; Cd])

#Defining the Jakobian
J(Ca, Cb, Cc, Cd) = [∇f1(Ca, Cb, Cc, Cd)';
                     ∇f2(Ca, Cb, Cc, Cd)';
                     ∇f3(Ca, Cb, Cc, Cd)';
                     ∇f4(Ca, Cb, Cc, Cd)']

#Defining the Matrix Function
F(Ca, Cb, Cc, Cd) = [f1(Ca, Cb, Cc, Cd);
                     f2(Ca, Cb, Cc, Cd);
                     f3(Ca, Cb, Cc, Cd);
                     f4(Ca, Cb, Cc, Cd)]


#Initial Condition
C = [0.7, 0.4, 0.0, 0.0]

#Newton Raphson Method
while norm(F(C[1], C[2], C[3], C[4]))> 1e-5
    d = J(C[1], C[2], C[3], C[4]) \ -F(C[1], C[2], C[3], C[4])
    global C += d
end

#Exhibiting Ca, Cb, Cc and Cd
println("Ca= ", round(C[1], digits=6), " mol/L")
println("Cb= ", round(C[2], digits=6), " mol/L")
println("Cc= ", round(C[3], digits=6), " mol/L")
println("Cd= ", round(C[4], digits=6), " mol/L")

#= Output:

Ca= 0.303189 mol/L
Cb= 0.003189 mol/L
Cc= 0.396811 mol/L
Cd= 0.396811 mol/L

=#