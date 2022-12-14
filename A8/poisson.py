import numpy as np
import copy

def CG(A, b, x_guess):
    # Wtf does Num-nodes do?
    numNodes = 3

    # define the vectors into which you will store data
    two_norm = np.zeros(numNodes)
    inf_norm = np.zeros(numNodes)
    
    #set-up of initial parameters
    x = copy.deepcopy(x_guess)
    print('b: ', b)
    r = b - A@x  # residual vector
    p = copy.deepcopy(r) 
    print(p)
    
    for i in range(numNodes):
        #alpha term
        a_n = np.dot(p, r)
        ap = A@p
        a_d = np.dot(p, ap)
        print("a_n", a_n, "a_d", a_d)
        alpha = a_n/a_d
        print('alpha: ', alpha)
            
        # update x
        x = copy.deepcopy(x + alpha*p)
        print('x: ', x)
            
        # update r
        r = b - A*x
        print('r: ',r)
        # calculate the two-norm and inf-norm
        # print('2 norm: ', np.linalg.norm(r,2))
        # two_norm[i] = np.linalg.norm(r,2)
        # inf_norm[i] = max(abs(r))

        # .dropna(inplace=True)

        # beta term
        ar, ap = A@r, A@p
        b_n = -1*np.dot(p, ar)
        b_d = np.dot(p, ap)
        print("b_n: ", b_n, "b_d: ", b_d)
        beta = b_n/b_d
        print('beta: ', beta)
            
        # update p
        p = r + beta*p
    return x, inf_norm, two_norm



def LinearCG(A, b, x0, tol=1e-3):
    xk = x0
    rk = np.dot(A, xk) - b
    pk = -rk
    rk_norm = np.linalg.norm(rk)
    
    num_iter = 0
    curve_x = [xk]
    while rk_norm > tol:
        apk = np.dot(A, pk)
        rkrk = np.dot(rk, rk)
        
        alpha = rkrk / np.dot(pk, apk)
        xk = xk + alpha * pk
        rk = rk + alpha * apk
        beta = np.dot(rk, rk) / rkrk
        pk = -rk + beta * pk
        
        num_iter += 1
        curve_x.append(xk)
        rk_norm = np.linalg.norm(rk)
        print("Iteration: ", num_iter, "rk_norm: ", rk_norm)
        # print('Iteration: {} \t x = {} \t residual = {:.4f}'.
        #       format(num_iter, xk, rk_norm))
    
    print('\nSolution: \t x = {}'.format(xk))
        
    return np.array(curve_x)