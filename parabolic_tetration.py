#=========================================================================================================
# This is a constructive toy implementation for the analytic continuation of parabolic tetration to ℂ.
# The goal is to demonstrate algorithmic feasability and overall intuition for a transcendental functional
# equation problem, not optimization, formal completeness or rigorous proof on a theorem level in any way.
#=========================================================================================================

import numpy as np
from scipy.special import factorial

def exp_gen(z):
    while True:
        yield z
        z = np.exp(z/np.e)

def log_gen(z):
    while True:
        z = np.e*np.log(z+0j)
        yield z

def taylor_series(eval_point, jet_coeffs):
    n = np.arange(jet_coeffs.shape[0])[:, None]
    return np.sum(jet_coeffs*eval_point**n/factorial(n), 0)

def assemble_orbit(init_cond, sum_trunc, grid_step):

    #=========================================================================================================
    # Returns a vector of size sum_trunc with elements on the tetration orbit from a initial condition, e.g.:
    # print(assemble_orbit(1, 100, 0)) -> [1, 1.44466786, 1.7014207, 1.86996122, 1.98957349, 2.07907521, ...]
    #
    # The parameter grid_step serves to shift the entire vector by performing discrete steps along the orbit,
    # used when analytic continuation requires real step > 1 or negative. Saves time and improves performance.
    #==========================================================================================================

    return np.hstack([
        np.fromiter(log_gen(init_cond), np.complex128, count = max(-grid_step, 0))[::-1][:sum_trunc],
        np.fromiter(exp_gen(init_cond), np.complex128, count = max(sum_trunc+grid_step, 0))[max(grid_step, 0):]
    ])

def assemble_ts_jet(init_cond, sum_trunc, grid_step, ts_degree):
    
    #=========================================================================================================
    # Builds the propagator from the orbit, then f'(0) from the asymptotic quadratic decay for large sum_trunc
    # Derivatives of the orbit are proportional to the propagator and the seed f'(0), tetr_prime = tetr_orbit'
    #=========================================================================================================

    tetr_orbit = assemble_orbit(init_cond, sum_trunc, grid_step)
    propagator = np.cumprod(tetr_orbit/np.e)
    tetr_prime = propagator*2*np.e/(propagator[-1]*sum_trunc**2)

    #=========================================================================================================
    # Recursively builds the Taylor jet order by order from a convolution of previously built orders and their
    # cumulative sums up to truncation error. Returns a matrix of dimensions (ts_degree+1, sum_trunc) with the
    # approximated Taylor coefficients on the rows for each discrete step on the columns. e.g., set f(1) = 1:
    #
    # print(assemble_ts_jet(1, 10**4, 0, 4)[:, :3]) - > 
    # [[ 1.        +0.j  1.44466786+0.j  1.7014207 +0.j]
    # [ 0.61167024+0.j  0.32508047+0.j  0.20347362+0.j]
    # [-0.464155  +0.j -0.17353159+0.j -0.084283  +0.j]
    # [ 0.55196446-0.j  0.1432838 -0.j  0.05362544-0.j]
    # [-0.9042921 +0.j -0.16143434+0.j -0.04631397+0.j]]
    #
    # Then, the first column contain the first 5 Taylor coefficients around f(1), the second column the first
    # five Taylor coefficients around f(2) etc up to f(sum_trunc-1). Though computed in parallel, performance
    # drops the higher you probe for you'll be now considering a jet with a truncation smaller than sum_trunc.
    #=========================================================================================================

    jet_coeffs = np.zeros([ts_degree+1, sum_trunc], np.complex128)
    jet_coeffs[:2] = tetr_orbit, tetr_prime
    integrator = np.zeros([ts_degree-1, sum_trunc], np.complex128)

    for k in range(1, ts_degree):
        integrator[k-1] = np.cumsum(jet_coeffs[k][::-1])[::-1]
        bin_coeffs = np.array([1], dtype=np.int64) if k == 1 else np.convolve(bin_coeffs, [1, 1])
        jet_coeffs[k+1] = -np.sum(bin_coeffs[:, None]*jet_coeffs[1:k+1]*integrator[:k][::-1], 0)/np.e
        
    return jet_coeffs

def test_composition(init_cond, sum_trunc, grid_step, ts_degree, iterations):

    #=========================================================================================================
    # This is a simple test for the semigroup property of functional composition. We perform i-many sequential
    # analytic continuations of 1/i steps and prints the results for any given initial condition and iteration
    # list. The property should hold for integer i>0 and initial condition on the basin of attraction. e.g.:
    #
    # test_composition(1, 10**4, 0, 20, [1, 2, 3, 4]) ->
    # [1.44500573+0.j 1.70163221+0.j 1.87010674+0.j ... 2.71773838+0.j 2.71773844+0.j 2.71773849+0.j]
    # [1.44500577+0.j 1.70163221+0.j 1.87010674+0.j ... 2.71773838+0.j 2.71773844+0.j 2.71773849+0.j]
    # [1.44500577+0.j 1.70163221+0.j 1.87010674+0.j ... 2.71773838+0.j 2.71773844+0.j 2.71773849+0.j]
    # [1.44500577+0.j 1.70163221+0.j 1.87010674+0.j ... 2.71773838+0.j 2.71773844+0.j 2.71773849+0.j]
    #
    # test_composition(np.pi/2, 10**4, 9, 20, [1, 2, 3, 4]) ->
    # [2.35911153+0.j 2.38182889+0.j 2.40181783+0.j ... 2.71773895+0.j 2.717739  +0.j 2.71773906+0.j]
    # [2.35911153+0.j 2.38182889+0.j 2.40181783+0.j ... 2.71773895+0.j 2.717739  +0.j 2.71773906+0.j]
    # [2.35911153+0.j 2.38182889+0.j 2.40181783+0.j ... 2.71773895+0.j 2.717739  +0.j 2.71773906+0.j]
    # [2.35911153+0.j 2.38182889+0.j 2.40181783+0.j ... 2.71773895+0.j 2.717739  +0.j 2.71773906+0.j]
    #
    # test_composition(2+3j, 10**4, -5, 20, [1, 2, 3, 4]) ->
    # [3.8890356 +7.44336495e-01j 4.0258165 +1.13077845e+00j 4.02238115+1.77697488e+00j ... ]
    # [3.88903579+7.44336515e-01j 4.02581691+1.13077824e+00j 4.02238248+1.77697392e+00j ... ]
    # [3.88903579+7.44336516e-01j 4.02581691+1.13077824e+00j 4.02238248+1.77697392e+00j ... ]
    # [3.88903579+7.44336516e-01j 4.02581691+1.13077824e+00j 4.02238248+1.77697392e+00j ... ]
    #=========================================================================================================
    
    for i in iterations:
        eval_point = 1/i
        extensions = taylor_series(eval_point, assemble_ts_jet(init_cond, sum_trunc, grid_step, ts_degree))
        for _ in range(i-1):
            extensions = taylor_series(eval_point, assemble_ts_jet(extensions[0], sum_trunc, 0, ts_degree))
        print(extensions)

def compute_tetration(z, sum_trunc=10**4, ts_degree=20):

    #=========================================================================================================
    # Returns the numerical value of parabolic tetration at a complex point. Accounts for known singularities
    # and moves away from them whenever possible at each step so that we have a greater radius of convergence.
    # This particular function is under revision and will most likely change to include adaptive steps in the
    # imaginary direction to improve computation for large Im(z) and stability near singularities. Examples:
    #
    # Monotonoic on the reals for -2 < x:
    # print(compute_tetration(0))     -> 0.9999999999999998
    # print(compute_tetration(0.1))   -> 1.058945980275626
    # print(compute_tetration(0.25))  -> 1.139742437500749
    # print(compute_tetration(0.5))   -> 1.2574110494403492
    # print(compute_tetration(0.75))  -> 1.3579988825314138
    # print(compute_tetration(0.9))   -> 1.4116791656503156
    # print(compute_tetration(1))     -> 1.444667861009766
    # print(compute_tetration(1.5))   -> 1.588159302438947
    # print(compute_tetration(np.e))  -> 1.8287204214620858
    # print(compute_tetration(np.pi)) -> 1.8893194462578213
    #
    # Stable for arbitrary complex z:
    # print(compute_tetration(1+1j))         -> 1.5252538682062966+0.3030298048896416j
    # print(compute_tetration(-5+3j))        -> 3.285472864530328+1.209177697469653j
    # print(compute_tetration(3.4-np.pi*1j)) -> 2.075020418428692-0.30888627796601564j
    # print(compute_tetration(239+358j))     -> 2.7113712761861426+0.010718560004437641j (approaching e)
    #
    # Conjugation respects holomorphicity:
    # print(compute_tetration(np.sqrt(2)+13j)) -> 2.5841425194806416+0.35424038402188895j
    # print(compute_tetration(np.sqrt(2)-13j)) -> 2.5841425194806416-0.35424038402188895j
    #=========================================================================================================

    if isinstance(z, int) and z < -1:
        return np.nan 
    init_cond = np.exp(1/np.e)
    real_part = np.real(z)
    imag_part = np.imag(z)
    real_step = real_part%1
    imag_sign = np.sign(imag_part)*1j
    imag_step = (imag_part-int(imag_part))*1j
    imag_norm = int(np.abs(imag_part))
    grid_step = int(real_part-real_step)-1
    for ts_center in [imag_sign]*imag_norm+[real_step]:
        ts_coeffs = assemble_ts_jet(init_cond, sum_trunc, 0, ts_degree)
        init_cond = taylor_series(ts_center, ts_coeffs)[0]
    ts_coeffs = assemble_ts_jet(init_cond, sum_trunc, 0, ts_degree)
    init_cond = taylor_series(imag_step, ts_coeffs)[0]
    ts_coeffs = assemble_ts_jet(init_cond, 1, grid_step, 1)
    return ts_coeffs[0, 0]

