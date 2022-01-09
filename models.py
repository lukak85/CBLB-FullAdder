# Urrios 2016: multicellular memory + Macia 2016

import numpy as np



def not_cell(state, params):
    L_X, x, y, N_X, N_Y = state
    delta_L, gamma_L_X, n_y, theta_L_X, eta_x, omega_x, m_x, delta_x, rho_x = params

    # presume that the molecules are degraded in the same strain as they are produced
    N_Y = N_X


    f = gamma_L_X * (y ** n_y)/(1 + (theta_L_X*y)**n_y )
    dL_X_dt = N_X * (f - delta_L * L_X)

    dx_dt = N_X * (eta_x * (1/(1+ (omega_x*L_X)**m_x))) - N_Y * (delta_x * x) - rho_x * x

    return dL_X_dt, dx_dt



def yes_cell(state, params):
    x, y, N_X, N_Y = state
    gamma_x, n_y, theta_x, delta_x, rho_x = params

    # presume that the molecules are degraded in the same strain as they are produced
    N_Y = N_X


    dx_dt = N_X * gamma_x * (y ** n_y)/(1 + (theta_x*y)**n_y ) - N_Y * (delta_x * x) - rho_x * x
    
    return dx_dt



# L_A ... intermediate
# a ... out
# b ... in
# N_A ... number of cells
def not_cell_wrapper(state, params):
    L_A, a, b, N_A = state

    state_A = L_A, a, b, N_A, N_A
    params_A = params

    return not_cell(state_A, params_A)


# a ... out
# b ... in
# N_A ... number of cells
def yes_cell_wrapper(state, params):
    a, b, N_A = state

    state_A = a, b, N_A, N_A
    params_A = params

    return yes_cell(state_A, params_A)



def population(state, params):
    N = state
    r = params

    dN = r * N * (1 - N)    

    return dN



def not_gate(A, N_A, L_A, out, r_X, params_not):
    state_not_A =  L_A, out, A, N_A
    dL_A, dd = not_cell_wrapper(state_not_A, params_not)
    dN_A = population(N_A, r_X)
    return dN_A, dL_A, dd


def yes_gate(A, N_A, out, r_X, params_yes):
    state_yes_A =  out, A, N_A
    dd = yes_cell_wrapper(state_yes_A, params_yes)
    dN_A = population(N_A, r_X)
    return dN_A, dd



def full_adder_model(state, T, params):
    delta_L, gamma_L_X, n_y, theta_L_X, eta_x, omega_x, m_x, delta_x, rho_x, gamma_x, theta_x, r_X = params
    params_yes = gamma_x, n_y, theta_x, delta_x, rho_x
    params_not = delta_L, gamma_L_X, n_y, theta_L_X, eta_x, omega_x, m_x, delta_x, rho_x

    A, B, C = state[:3]

    N_0, L_0 = state[3:5]
    N_1 = state[5]
    N_2 = state[6]

    N_3 = state[7]
    N_4, L_4 = state[8:10]
    N_5 = state[10]

    N_6 = state[11]
    N_7 = state[12]
    N_8, L_8 = state[13:15]

    N_9, L_9, = state[15:17]
    N_10, L_10 = state[17:19]
    N_11, L_11 = state[19:21]

    N_12, L_12 = state[21:23]
    N_13, L_13 = state[23:25]
    N_14, L_14 = state[25:27]
    N_15, L_15 = state[27:29]


    N_16, L_16 = state[29:31]
    N_17, L_17 = state[31:33]

    N_18, L_18 = state[33:35]
    N_19, L_19 = state[35:37]

    N_20, L_20 = state[37:39]
    N_21, L_21 = state[39:41]

    N_22, L_22 = state[41:43]
    N_23, L_23 = state[43:45]
    N_24, L_24 = state[45:47]


    x1, x2, x3, x4, s, y1, y2, y3, cout = state[47:56]
    

    # S
    a0_not = not_gate(A, N_0, L_0, x1, r_X, params_not)
    b0_yes = yes_gate(B, N_1, x1, r_X, params_yes)
    c0_yes = yes_gate(C, N_2, x1, r_X, params_yes)

    a1_yes = yes_gate(A, N_3, x2, r_X, params_yes)
    b1_not = not_gate(B, N_4, L_4, x2, r_X, params_not)
    c1_yes = yes_gate(C, N_5, x2, r_X, params_yes)

    a2_yes = yes_gate(A, N_6, x3, r_X, params_yes)
    b2_yes = yes_gate(B, N_7, x3, r_X, params_yes)
    c2_not = not_gate(C, N_8, L_8, x3, r_X, params_not)

    a3_not = not_gate(A, N_9, L_9, x4, r_X, params_not)
    b3_not = not_gate(B, N_10, L_10, x4, r_X, params_not)
    c3_not = not_gate(C, N_11, L_11, x4, r_X, params_not)

    x1_not = not_gate(x1, N_12, L_12, s, r_X, params_not)
    x2_not = not_gate(x2, N_13, L_13, s, r_X, params_not)
    x3_not = not_gate(x3, N_14, L_14, s, r_X, params_not)
    x4_not = not_gate(x4, N_15, L_15, s, r_X, params_not)

    # Cout
    a4_not = not_gate(A, N_16, L_16, y1, r_X, params_not)
    b4_not = not_gate(B, N_17, L_17, y1, r_X, params_not)

    a5_not = not_gate(A, N_18, L_18, y2, r_X, params_not)
    c5_not = not_gate(C, N_19, L_19, y2, r_X, params_not)

    b6_not = not_gate(B, N_20, L_20, y3, r_X, params_not)
    c6_not = not_gate(C, N_21, L_21, y3, r_X, params_not)

    y1_not = not_gate(y1, N_22, L_22, cout, r_X, params_not)
    y2_not = not_gate(y2, N_23, L_23, cout, r_X, params_not)
    y3_not = not_gate(y3, N_24, L_24, cout, r_X, params_not)

    dA, dB, dC = 0,0,0

    dstate = np.array([
        dA, dB, dC,

        a0_not[0], a0_not[1],
        b0_yes[0],
        c0_yes[0],

        a1_yes[0],
        b1_not[0], b1_not[1],
        c1_yes[0],

        a2_yes[0],
        b2_yes[0],
        c2_not[0], c2_not[1],

        a3_not[0], a3_not[1],
        b3_not[0], b3_not[1],
        c3_not[0], c3_not[1],

        x1_not[0], x1_not[1],
        x2_not[0], x2_not[1],
        x3_not[0], x3_not[1],
        x4_not[0], x4_not[1],

        a4_not[0], a4_not[1],
        b4_not[0], b4_not[1],

        a5_not[0], a5_not[1],
        c5_not[0], c5_not[1],
        
        b6_not[0], b6_not[1],
        c6_not[0], c6_not[1],

        y1_not[0], y1_not[1],
        y2_not[0], y2_not[1],
        y3_not[0], y3_not[1],

        a0_not[2] + b0_yes[1] + c0_yes[1],
        a1_yes[1] + b1_not[2] + c1_yes[1],
        a2_yes[1] + b2_yes[1] + c2_not[2],
        a3_not[2] + b3_not[2] + c3_not[2],

        x1_not[2] + x2_not[2] + x3_not[2] + x4_not[2],

        a4_not[2] + b4_not[2],
        a5_not[2] + c5_not[2],
        b6_not[2] + c6_not[2],

        y1_not[2] + y2_not[2] + y3_not[2]
    ])
    
    return dstate


def FA_model(T, state, params):
    delta_L, gamma_L_X, n_y, theta_L_X, eta_x, omega_x, m_x, delta_x, rho_x, gamma_x, theta_x, r_X = params

    params_adder = delta_L, gamma_L_X, n_y, theta_L_X, eta_x, omega_x, m_x, delta_x, rho_x, gamma_x, theta_x, r_X

    dstate_adder = full_adder_model(state, T, params_adder)

    return dstate_adder