from sympy import cos, sin, sqrt, pi, Matrix, Symbol, exp, print_latex, simplify

class Initializer:
    def __init__(self):
        theta = Symbol('theta',real=True)
        phi = Symbol('phi',real=True)
        gamma = Symbol('gamma',real=True, positive=True)
        p = Symbol('p',real=True, positive=True)

class AD:
    
    def name_changer(self, map_name: any):
        if map_name == 'ad':
            return 'amplitude-damping'
    
    def rho_AB_ad(theta, phi, p):
        state = Matrix([[(cos(theta/2)),
                        (sqrt(p)*exp(1j*phi)*sin(theta/2)),
                        (sqrt(1-p)*exp(1j*phi)*sin(theta/2)),
                        0]])
        return state
    