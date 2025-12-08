#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy.stats import norm, gamma
from scipy.special import gammainc, gammaincc
from scipy.optimize import minimize
import argparse
import sys
import warnings
import copy # For deep copying range dictionaries

# Suppress runtime warnings during normal operation
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- Default Values ---
DEFAULT_Y_UPPER_BROAD = 100.0
DEFAULT_DELTA_UPPER_BROAD = 50.0
TARGET_REDUCTION_STEP = 0.05
MINIMUM_TARGET_THRESHOLD = 0.01

# --- Helper Functions ---

def calculate_constants(n, k, w, dd, lambdas):
    # (Implementation assumed correct from previous versions)
    # Returns constants dict including theoretical mins
    # --- Paste robust implementation of calculate_constants here ---
    # --- START calculate_constants implementation ---
    d = len(lambdas)
    if not (1 <= dd < d): raise ValueError(f"dd must be between 1 and d-1 (d={d}), got dd={dd}")
    if not (1 <= k <= n): raise ValueError(f"k must be between 1 and n (n={n}), got k={k}")
    if not (w >= 1): print(f"Warning: w={w} < 2. Theorem meaningfulness may reduce.", file=sys.stderr)
    if not (w * k <= n): raise ValueError(f"w*k must be <= n (n={n}), got w*k={w*k}")
    if not isinstance(lambdas, np.ndarray): lambdas = np.array(lambdas, dtype=float)
    if not np.all(lambdas > 0):
        print("Internal Warning: calculate_constants received non-positive eigenvalues.", file=sys.stderr)
        lambdas = np.maximum(lambdas, 1e-12)

    lambda_1_dd = lambdas[:dd]
    lambda_dd_d = lambdas[dd:]

    constants = {}
    constants['n'] = float(n); constants['k'] = float(k); constants['w'] = float(w)
    constants['dd'] = dd; constants['d'] = d; constants['eps'] = 1e-10
    constants['valid_sw_r'] = True; constants['valid_sw_z'] = True

    # R related constants
    if len(lambda_dd_d) == 0:
         constants['mu_R'] = 0.0; constants['sigma_R_sq'] = 0.0; constants['sigma_R'] = 0.0
         constants['nu_R'] = np.nan; constants['theta_R'] = np.nan; constants['valid_sw_r'] = False
         print("Warning: No dimensions left for R. SW approx for R invalid.", file=sys.stderr)
    else:
        constants['mu_R'] = np.sum(2 * lambda_dd_d)
        constants['sigma_R_sq'] = np.sum(8 * lambda_dd_d**2)
        if constants['sigma_R_sq'] <= constants['eps']:
             print("Info: sigma_R^2 near zero. R treated as constant.", file=sys.stderr)
             constants['sigma_R'] = 0.0; constants['nu_R'] = np.inf; constants['theta_R'] = 0.0
             constants['valid_sw_r'] = False # Not using SW gamma, but R is valid constant
        else:
            constants['sigma_R'] = np.sqrt(constants['sigma_R_sq'])
            constants['mu_R'] = max(0.0, constants['mu_R'])
            if constants['mu_R'] < constants['eps']:
                 print("Warning: mu_R near zero, sigma_R > 0. SW problematic.", file=sys.stderr)
                 constants['nu_R'] = max(constants['eps'], 2 * constants['mu_R']**2 / constants['sigma_R_sq'])
                 constants['theta_R'] = constants['sigma_R_sq'] / (2 * constants['mu_R']) if constants['mu_R'] > constants['eps'] else np.inf
            else:
                 constants['nu_R'] = 2 * constants['mu_R']**2 / constants['sigma_R_sq']
                 constants['theta_R'] = constants['sigma_R_sq'] / (2 * constants['mu_R'])
            if constants['nu_R'] <= constants['eps'] or constants['theta_R'] <= constants['eps'] or np.isnan(constants['nu_R']) or np.isnan(constants['theta_R']):
                 print(f"Warning: Invalid SW params for R (nu={constants['nu_R']:.2e}, th={constants['theta_R']:.2e}).", file=sys.stderr)
                 constants['valid_sw_r'] = False

    # Z related constants
    constants['Z_offset'] = np.sum(2 * lambda_1_dd**2)
    constants['mu_Z_prime'] = np.sum(4 * lambda_1_dd**2)
    constants['sigma_Z_prime_sq'] = np.sum(32 * lambda_1_dd**4)
    if constants['sigma_Z_prime_sq'] <= constants['eps']:
        print("Warning: sigma_Z_prime^2 near zero. SW approx for Z invalid.", file=sys.stderr)
        constants['nu_Z_prime'] = np.nan; constants['theta_Z_prime'] = np.nan; constants['valid_sw_z'] = False
    else:
        constants['mu_Z_prime'] = max(0.0, constants['mu_Z_prime'])
        if constants['mu_Z_prime'] < constants['eps']:
             print("Warning: mu_Z_prime near zero, sigma_Z_prime > 0. SW problematic.", file=sys.stderr)
             constants['nu_Z_prime'] = max(constants['eps'], 2 * constants['mu_Z_prime']**2 / constants['sigma_Z_prime_sq'])
             constants['theta_Z_prime'] = constants['sigma_Z_prime_sq'] / (2 * constants['mu_Z_prime']) if constants['mu_Z_prime'] > constants['eps'] else np.inf
        else:
             constants['nu_Z_prime'] = 2 * constants['mu_Z_prime']**2 / constants['sigma_Z_prime_sq']
             constants['theta_Z_prime'] = constants['sigma_Z_prime_sq'] / (2 * constants['mu_Z_prime'])
        if constants['nu_Z_prime'] <= constants['eps'] or constants['theta_Z_prime'] <= constants['eps'] or np.isnan(constants['nu_Z_prime']) or np.isnan(constants['theta_Z_prime']):
            print(f"Warning: Invalid SW params for Z' (nu={constants['nu_Z_prime']:.2e}, th={constants['theta_Z_prime']:.2e}).", file=sys.stderr)
            constants['valid_sw_z'] = False

    # Order Statistic constants
    constants['alpha_k'] = k / (n + 1.0); constants['beta_wk'] = w * k / (n + 1.0)
    q_eps = 1e-10
    if not (q_eps < constants['alpha_k'] < 1.0 - q_eps): raise ValueError(f"alpha_k={constants['alpha_k']:.4g} too close to 0/1.")
    if not (q_eps < constants['beta_wk'] < 1.0 - q_eps): raise ValueError(f"beta_wk={constants['beta_wk']:.4g} too close to 0/1.")
    if constants['beta_wk'] <= constants['alpha_k']: raise ValueError(f"beta_wk <= alpha_k. Check w > 1.")
    ppf_alpha = norm.ppf(constants['alpha_k']); ppf_beta = norm.ppf(constants['beta_wk'])
    pdf_alpha = norm.pdf(ppf_alpha); pdf_beta = norm.pdf(ppf_beta)
    if pdf_alpha <= constants['eps'] or pdf_beta <= constants['eps']: raise ValueError("PDF at quantile near zero.")
    constants['C_alpha_beta'] = ppf_beta - ppf_alpha
    term_sig_1 = constants['alpha_k']*(1-constants['alpha_k'])/pdf_alpha**2
    term_sig_2 = constants['beta_wk']*(1-constants['beta_wk'])/pdf_beta**2
    term_sig_3 = (2*constants['alpha_k']*(1-constants['beta_wk'])) / (pdf_alpha*pdf_beta) # Clarified parentheses
    constants['C_sigma_sq'] = term_sig_1 + term_sig_2 - term_sig_3
    if constants['C_sigma_sq'] < -constants['eps']: raise ValueError(f"C_sigma^2={constants['C_sigma_sq']:.4g} negative.")
    elif constants['C_sigma_sq'] < constants['eps']:
         print(f"Warning: C_sigma^2={constants['C_sigma_sq']:.4g} near zero.", file=sys.stderr)
         constants['C_sigma'] = constants['eps']
    else: constants['C_sigma'] = np.sqrt(constants['C_sigma_sq'])

    # THEORETICAL minimum bounds
    constants['y_min_theoretical'] = -constants['mu_R'] / constants['sigma_R'] if constants['sigma_R'] > constants['eps'] else -np.inf
    if constants['C_sigma'] < constants['eps']: constants['delta_min_theoretical'] = -np.inf
    else: constants['delta_min_theoretical'] = -constants['C_alpha_beta'] * np.sqrt(constants['n']) / constants['C_sigma']
    # --- END calculate_constants implementation ---
    return constants

def calculate_lower_bound(y, delta, const, verbose=False):
    # (Implementation assumed correct from previous version with exception fix)
    # Returns the bound value (float >= 0).
    # --- Paste robust implementation of calculate_lower_bound here ---
    # --- START calculate_lower_bound implementation ---
    bound_val = 0.0
    eps = const.get('eps', 1e-10)
    if y < const['y_min_theoretical'] - eps:
        if verbose: print(f"DEBUG: y={y:.4g} < theoretical y_min={const['y_min_theoretical']:.4g}", file=sys.stderr)
        return bound_val
    if delta <= const['delta_min_theoretical'] + eps:
         if verbose: print(f"DEBUG: delta={delta:.4g} <= theoretical delta_min={const['delta_min_theoretical']:.4g}", file=sys.stderr)
         return bound_val

    d_1 = const['mu_R'] + y * const['sigma_R']; d_1 = max(0.0, d_1)
    C_tot = const['C_alpha_beta'] + delta * (const['n']**(-0.5)) * const['C_sigma']
    if C_tot <= eps:
        if verbose: print(f"DEBUG: C_tot={C_tot:.4g} <= eps", file=sys.stderr)
        return bound_val

    # Term 1
    F_R_d1 = 0.0; term1 = 0.0
    if const['sigma_R'] < eps: F_R_d1 = 1.0 if d_1 >= const['mu_R'] - eps else 0.0
    elif not const['valid_sw_r']:
         if verbose: print(f"DEBUG: Invalid SW for R.", file=sys.stderr); return bound_val
    else:
         a_R = const['nu_R'] / 2
         arg_gamma_R = d_1 / (2 * const['theta_R']) if const['theta_R'] > eps else (np.inf if d_1 > eps else 0.0)
         if not (a_R > 0 and arg_gamma_R >= 0 and not np.isnan(a_R) and not np.isnan(arg_gamma_R)):
              if verbose: print(f"DEBUG: Invalid args gammainc R: a={a_R:.3g}, x={arg_gamma_R:.3g}", file=sys.stderr); return bound_val
         try:
             with np.errstate(invalid='ignore', divide='ignore'):
                 F_R_d1 = gammainc(a_R, arg_gamma_R); F_R_d1 = 0.0 if np.isnan(F_R_d1) else F_R_d1
         except Exception as e:
             if verbose: print(f"DEBUG: Err gammainc R: {e}", file=sys.stderr)
             return bound_val
         F_R_d1 = np.clip(F_R_d1, 0.0, 1.0)
    try: term1 = F_R_d1 ** const['k']; term1 = 0.0 if np.isnan(term1) else term1
    except OverflowError: term1 = 0.0

    # Term 2
    term2 = norm.sf(delta); term2 = np.clip(term2, 0.0, 1.0)

    # Term 3
    term3 = 0.0
    try:
        if C_tot < eps: arg_Z = np.inf
        else:
            d1_sq = d_1**2; Ctot_sq = C_tot**2 # Semicolon removed
            arg_Z = (d1_sq / Ctot_sq) if Ctot_sq > eps else (np.inf if d1_sq > eps else 0.0)
        if np.isinf(arg_Z) or np.isnan(arg_Z): bar_F_Z = 0.0
        elif not const['valid_sw_z']:
             if verbose: print(f"DEBUG: Invalid SW for Z.", file=sys.stderr); return bound_val
        else:
             a_Z = const['nu_Z_prime'] / 2
             arg_gamma_Z_num = max(0, arg_Z - const['Z_offset'])
             arg_gamma_Z = arg_gamma_Z_num / (2*const['theta_Z_prime']) if const['theta_Z_prime'] > eps else (np.inf if arg_gamma_Z_num > eps else 0.0)
             if not (a_Z > 0 and arg_gamma_Z >= 0 and not np.isnan(a_Z) and not np.isnan(arg_gamma_Z)):
                  if verbose: print(f"DEBUG: Invalid args gammaincc Z: a={a_Z:.3g}, x={arg_gamma_Z:.3g}", file=sys.stderr); return bound_val
             try:
                 with np.errstate(invalid='ignore', divide='ignore'):
                     bar_F_Z = gammaincc(a_Z, arg_gamma_Z); bar_F_Z = 0.0 if np.isnan(bar_F_Z) else bar_F_Z
             except Exception as e:
                 if verbose: print(f"DEBUG: Err gammaincc Z: {e}", file=sys.stderr)
                 return bound_val
             term3 = np.clip(bar_F_Z, 0.0, 1.0)
    except OverflowError: term3 = 0.0

    lower_bound = term1 * term2 * term3
    lower_bound = max(0.0, lower_bound)
    if np.isnan(lower_bound): lower_bound = 0.0
    if verbose: print(f"DEBUG: y={y:.3g}, d={delta:.3g} -> T1={term1:.3g}, T2={term2:.3g}, T3={term3:.3g} -> B={lower_bound:.3g}", file=sys.stderr)
    # --- END calculate_lower_bound implementation ---
    return lower_bound

def objective_function(params, constants):
    # (Implementation assumed correct from previous version)
    # Returns -lower_bound or penalty
    # --- Paste robust implementation of objective_function here ---
    # --- START objective_function implementation ---
    y, delta = params
    lower_bound = calculate_lower_bound(y, delta, constants, verbose=False)
    if lower_bound <= 1e-30: return 1e10 # Penalty
    return -lower_bound
    # --- END objective_function implementation ---

def calculate_heuristic_bounds(constants, current_target_prob):
    """
    Calculates tighter bounds based on the heuristic: each term >= current_target_prob.
    Correctly derives delta lower bound from Term 3 using y_min_T1.
    """
    bounds_info = {
        'y_min_T1': -np.inf, 'delta_max_T2': np.inf,
        'y_max_T3': np.inf, 'delta_min_T3': -np.inf, # Correctly calculated delta_min_T3
        'possible': True, 'messages': []
    }
    k = constants['k']
    eps = constants['eps']
    sqrt_n = np.sqrt(constants['n'])
    # Avoid division by zero if C_sigma is effectively zero
    C_sig_term = constants['C_sigma'] * (sqrt_n ** -1.0) if constants['C_sigma'] > eps else eps

    if not (eps < current_target_prob < 1.0 - eps):
         bounds_info['messages'].append(f"Target probability {current_target_prob:.3f} invalid for heuristic.")
         bounds_info['possible'] = False
         return bounds_info

    # --- Term 1 -> y_min_T1 ---
    p_target_R = current_target_prob**(1.0 / k)
    bounds_info['messages'].append(f"Term1 Heuristic: F_R(d1) >= {p_target_R:.6f}")
    # (Calculation for y_min_T1 remains the same as before)
    if constants['sigma_R'] < eps: # R is constant mu_R
        if 1.0 < p_target_R - eps: bounds_info['possible'] = False; bounds_info['messages'].append("  (Term1: Target F_R > 1 impossible.)")
        else: bounds_info['y_min_T1'] = -np.inf; bounds_info['messages'].append("  (Term1: sigma_R=0. No y lower bound.)")
    elif not constants['valid_sw_r']: bounds_info['possible'] = False; bounds_info['messages'].append("  (Term1: SW invalid.)")
    elif p_target_R >= 1.0 - eps: bounds_info['y_min_T1'] = np.inf; bounds_info['possible'] = False; bounds_info['messages'].append("  (Term1: Target F_R >= ~1. Infeasible.)")
    else:
        try:
            a_R = constants['nu_R'] / 2; scale_R = 2 * constants['theta_R']
            if not (a_R > 0 and scale_R > 0): raise ValueError("Invalid R Gamma params")
            d1_min_T1 = gamma.ppf(p_target_R, a=a_R, scale=scale_R)
            if np.isinf(d1_min_T1) or np.isnan(d1_min_T1): bounds_info['y_min_T1'] = np.inf; bounds_info['possible'] = False; bounds_info['messages'].append(f"  (Term1: PPF F_R gave {d1_min_T1}.)")
            else:
                bounds_info['y_min_T1'] = (d1_min_T1 - constants['mu_R']) / constants['sigma_R']
                bounds_info['messages'].append(f"  => y >= {bounds_info['y_min_T1']:.4g}")
        except Exception as e: bounds_info['messages'].append(f"  (Term1: Error: {e})"); bounds_info['possible'] = False
    # --- Store d1 corresponding to y_min_T1 for later use ---
    d1_at_ymin = constants['mu_R'] + bounds_info['y_min_T1'] * constants['sigma_R'] if np.isfinite(bounds_info['y_min_T1']) else np.inf
    if constants['sigma_R'] < eps : d1_at_ymin = constants['mu_R'] # If sigma_R=0, use mu_R


    # --- Term 2 -> delta_max_T2 ---
    delta_target_cdf = 1.0 - current_target_prob
    bounds_info['messages'].append(f"Term2 Heuristic: CDF(delta) <= {delta_target_cdf:.3f}")
    # (Calculation for delta_max_T2 remains the same as before)
    if delta_target_cdf <= eps: bounds_info['delta_max_T2'] = -np.inf; bounds_info['possible'] = False; bounds_info['messages'].append("  (Term2: Requires delta -> -inf.)")
    else:
        bounds_info['delta_max_T2'] = norm.ppf(delta_target_cdf)
        bounds_info['messages'].append(f"  => delta <= {bounds_info['delta_max_T2']:.4g}")

    # --- Term 3 -> y_max_T3 AND delta_min_T3 ---
    z_target_quantile = 1.0 - current_target_prob
    bounds_info['messages'].append(f"Term3 Heuristic: F_Z(argZ) <= {z_target_quantile:.3f}")
    z_q = np.nan # Initialize z_q
    if not constants['valid_sw_z']: bounds_info['possible'] = False; bounds_info['messages'].append("  (Term3: SW invalid.)")
    elif z_target_quantile <= eps: bounds_info['y_max_T3'] = -np.inf; bounds_info['possible'] = False; bounds_info['messages'].append("  (Term3: Target F_Z <= ~0. Infeasible.)")
    else:
        try:
            a_Z = constants['nu_Z_prime'] / 2; scale_Z = 2 * constants['theta_Z_prime']
            if not (a_Z > 0 and scale_Z > 0): raise ValueError("Invalid Z Gamma params")
            z_prime_q = gamma.ppf(z_target_quantile, a=a_Z, scale=scale_Z)
            z_q = constants['Z_offset'] + z_prime_q # This is the required upper bound for argZ

            if np.isinf(z_q) or np.isnan(z_q): bounds_info['possible'] = False; bounds_info['messages'].append(f"  (Term3: PPF F_Z gave {z_q}.)")
            elif z_q < 0: bounds_info['y_max_T3'] = -np.inf; bounds_info['possible'] = False; bounds_info['messages'].append(f"  (Term3: Required z_q={z_q:.4g} < 0 impossible.)")
            else:
                bounds_info['messages'].append(f"  Requires argZ <= {z_q:.4g}")
                sqrt_z_q = np.sqrt(z_q)

                # Calculate y_max_T3 (depends on delta_max_T2) - Unchanged logic
                delta_for_y_max = bounds_info['delta_max_T2']
                if delta_for_y_max <= constants['delta_min_theoretical'] + eps: bounds_info['possible'] = False; bounds_info['messages'].append("  (Term3: delta_max_T2 conflicts theoretical min.)")
                else:
                    C_tot_at_delta_max = constants['C_alpha_beta'] + delta_for_y_max * C_sig_term
                    if C_tot_at_delta_max <= eps: bounds_info['possible'] = False; bounds_info['messages'].append(f"  (Term3: C_tot({delta_for_y_max:.4g}) <= 0.)")
                    else:
                        d1_max_overall = sqrt_z_q * C_tot_at_delta_max
                        if constants['sigma_R'] < eps:
                            bounds_info['y_max_T3'] = np.inf
                            if constants['mu_R'] > d1_max_overall + eps: bounds_info['possible'] = False; bounds_info['messages'].append("  (Term3: sigma_R=0 inconsistency.)")
                        else:
                            bounds_info['y_max_T3'] = (d1_max_overall - constants['mu_R']) / constants['sigma_R']
                            bounds_info['messages'].append(f"  => y <= {bounds_info['y_max_T3']:.4g} (from delta_max)")

                # *** CORRECTED: Calculate delta_min_T3 (depends on y_min_T1 / d1_at_ymin) ***
                if np.isinf(d1_at_ymin) or d1_at_ymin < 0:
                    bounds_info['messages'].append(f"  (Cannot derive delta_min_T3 as d1_at_ymin={d1_at_ymin:.4g} is invalid/infinite)")
                elif C_sig_term <= eps: # Denominator C_sig_term is zero or negative
                     bounds_info['messages'].append(f"  (Cannot derive delta_min_T3 as C_sigma or C_sig_term is near zero)")
                elif sqrt_z_q <= eps: # Denominator sqrt_z_q is zero
                     bounds_info['messages'].append(f"  (Cannot derive delta_min_T3 as sqrt(z_q) is near zero)")
                else:
                     # Need C_tot(delta) >= d1_at_ymin / sqrt(z_q)
                     required_C_tot = d1_at_ymin / sqrt_z_q
                     # Solve C_alpha_beta + delta * C_sig_term >= required_C_tot for delta
                     delta_min_term3_val = (required_C_tot - constants['C_alpha_beta']) / C_sig_term
                     bounds_info['delta_min_T3'] = delta_min_term3_val
                     bounds_info['messages'].append(f"  => delta >= {bounds_info['delta_min_T3']:.4g} (from y_min)")

        except Exception as e:
            bounds_info['messages'].append(f"  (Term3: Error: {e})")
            bounds_info['possible'] = False

    # Final Consistency Checks (Ensure derived ranges overlap and respect theoretical bounds)
    if bounds_info['y_min_T1'] >= bounds_info['y_max_T3'] - eps:
        bounds_info['messages'].append(f"  (Check FAIL: y_min_T1 >= y_max_T3)")
        bounds_info['possible'] = False
    # Ensure delta_min_T3 respects theoretical min (it should if theory is right, but check)
    if bounds_info['delta_min_T3'] < constants['delta_min_theoretical'] - eps:
         bounds_info['messages'].append(f"  (Check Note: delta_min_T3 < theoretical delta_min. Will use theoretical.)")
         bounds_info['delta_min_T3'] = constants['delta_min_theoretical']
    # Check if the new delta lower bound conflicts with delta upper bound
    if bounds_info['delta_min_T3'] >= bounds_info['delta_max_T2'] - eps:
        bounds_info['messages'].append(f"  (Check FAIL: delta_min_T3 >= delta_max_T2)")
        bounds_info['possible'] = False

    return bounds_info


# --- Main Execution Logic ---

def main():
    parser = argparse.ArgumentParser(
        description="Optimize lower bound P(T1 <= T2) with adaptive target-based range refinement.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Core Inputs
    parser.add_argument("n", type=int, help="N")
    parser.add_argument("k", type=int, help="k")
    parser.add_argument("w", type=int, help="w")
    parser.add_argument("dd", type=int, help="d'")
    parser.add_argument("csv_file", type=str, help="Path to CSV file with eigenvalues")
    parser.add_argument("--eig_col", type=str, default="Eigenvalue", help="Eigenvalue column name")
    parser.add_argument("--target_bound", type=float, required=True, help="Desired minimum lower bound")

    # Optimization Control
    parser.add_argument("--optim_method", type=str, default="L-BFGS-B", help="Optimization method")
    parser.add_argument("--y_start", type=float, default=None, help="Optional starting guess for y")
    parser.add_argument("--delta_start", type=float, default=None, help="Optional starting guess for delta")
    parser.add_argument("--ftol", type=float, default=1e-12, help="Optimizer ftol")
    parser.add_argument("--gtol", type=float, default=1e-7, help="Optimizer gtol")

    # Search Range Specification / Override
    parser.add_argument("--y_search_lower_override", type=float, default=None, help="Manual y lower bound")
    parser.add_argument("--y_search_upper_override", type=float, default=None, help="Manual y upper bound")
    parser.add_argument("--delta_search_lower_override", type=float, default=None, help="Manual delta lower bound")
    parser.add_argument("--delta_search_upper_override", type=float, default=None, help="Manual delta upper bound")

    args = parser.parse_args()

    if not (0 < args.target_bound < 1):
        print(f"Error: --target_bound ({args.target_bound}) must be between 0 and 1.", file=sys.stderr)
        sys.exit(1)

    # --- Load Eigenvalues ---
    try:
        df = pd.read_csv(args.csv_file)
        if args.eig_col not in df.columns: raise ValueError(f"Column '{args.eig_col}' not found")
        lambdas = df[args.eig_col].dropna().astype(float).values
        if len(lambdas) == 0: raise ValueError(f"No valid eigenvalues found in '{args.eig_col}'")
        lambdas = np.sort(lambdas)[::-1] # Sort descending
        num_nonpos = np.sum(lambdas <= 0)
        if num_nonpos > 0:
             min_pos_val = 1e-12
             print(f"Warning: Found {num_nonpos} non-positive eigenvalues. Clamping to {min_pos_val}.", file=sys.stderr)
             lambdas = np.maximum(lambdas, min_pos_val)
    except FileNotFoundError: print(f"Error: CSV file not found: '{args.csv_file}'", file=sys.stderr); sys.exit(1)
    except ValueError as e: print(f"Error processing eigenvalues: {e}", file=sys.stderr); sys.exit(1)
    except Exception as e: print(f"Error loading eigenvalues: {e}", file=sys.stderr); sys.exit(1)

    # --- Calculate Base Constants ---
    try:
        constants = calculate_constants(args.n, args.k, args.w, args.dd, lambdas)
    except ValueError as e: print(f"Error processing inputs: {e}", file=sys.stderr); sys.exit(1)
    except Exception as e: print(f"Error calculating constants: {e}", file=sys.stderr); sys.exit(1)

    # Check SW validity needed for heuristic bounds and optimization
    can_proceed = True
    if not constants['valid_sw_r'] and constants['sigma_R'] > constants['eps']:
         print("\nError: SW approx for R invalid (and sigma_R > 0). Cannot use heuristic or optimize reliably.", file=sys.stderr)
         can_proceed = False
    if not constants['valid_sw_z']:
         print("\nError: SW approx for Z' invalid. Cannot use heuristic or optimize reliably.", file=sys.stderr)
         can_proceed = False
    if not can_proceed: sys.exit(1)

    # --- Print Base Info ---
    print("--- Base Constants & Setup ---")
    print(f"n={args.n}, k={args.k}, w={args.w}, dd={args.dd}, d={constants['d']}")
    print(f"Target Lower Bound Input: {args.target_bound:.3f}")
    print("Theoretical Minimum Ranges:")
    print(f"  y >= {constants['y_min_theoretical']:.4e}")
    print(f"  δ > {constants['delta_min_theoretical']:.4e}")
    print("-" * 30)

    # --- Adaptive Range Refinement Loop ---
    print("--- Range Refinement Process ---")
    eps_bound = 1e-9 # Small nudge for strict inequalities
    current_target = args.target_bound
    final_range = None
    final_target_used = None

    while current_target > MINIMUM_TARGET_THRESHOLD:
        print(f"\nAttempting refinement with target >= {current_target:.3f}")

        # 1. Start with initial broad range (using theoretical min, broad default max)
        current_range = {
            'y_lower': constants['y_min_theoretical'],
            'y_upper': DEFAULT_Y_UPPER_BROAD,
            'delta_lower': constants['delta_min_theoretical'] + eps_bound,
            'delta_upper': DEFAULT_DELTA_UPPER_BROAD
        }
        print("Initial Broad Range for this iteration:")
        print(f"  y: [{current_range['y_lower']:.4g}, {current_range['y_upper']:.4g}]")
        print(f"  δ: [{current_range['delta_lower']:.4g}, {current_range['delta_upper']:.4g}]")

        # 2. Apply Heuristic based on current_target
        print(f"Applying Heuristic (each term >= {current_target:.3f}):")
        heuristic_bounds = calculate_heuristic_bounds(constants, current_target)
        for msg in heuristic_bounds['messages']: print(f"  {msg}")

        range_after_heuristic = copy.deepcopy(current_range)
        heuristic_possible = heuristic_bounds['possible']
        if heuristic_possible:
            # Update using max for lower bounds, min for upper bounds
            range_after_heuristic['y_lower'] = max(range_after_heuristic['y_lower'], heuristic_bounds['y_min_T1'])
            range_after_heuristic['y_upper'] = min(range_after_heuristic['y_upper'], heuristic_bounds['y_max_T3'])
            range_after_heuristic['delta_lower'] = max(range_after_heuristic['delta_lower'], heuristic_bounds['delta_min_T3']) # ADDED
            range_after_heuristic['delta_upper'] = min(range_after_heuristic['delta_upper'], heuristic_bounds['delta_max_T2'])
            print("Range after Heuristic:")
            print(f"  y: [{range_after_heuristic['y_lower']:.4g}, {range_after_heuristic['y_upper']:.4g}]")
            print(f"  δ: [{range_after_heuristic['delta_lower']:.4g}, {range_after_heuristic['delta_upper']:.4g}]")
        else:
            print("Heuristic calculation failed or indicated infeasibility.")

        # 3. Apply Manual Overrides
        range_after_manual = copy.deepcopy(range_after_heuristic)
        print("Applying Manual Overrides (if specified):")
        override_applied = False
        if args.y_search_lower_override is not None:
            range_after_manual['y_lower'] = max(constants['y_min_theoretical'], args.y_search_lower_override)
            override_applied = True
        if args.y_search_upper_override is not None:
            range_after_manual['y_upper'] = args.y_search_upper_override
            override_applied = True
        if args.delta_search_lower_override is not None:
            # Ensure override respects theoretical minimum + epsilon
            override_delta_lower = max(constants['delta_min_theoretical'] + eps_bound, args.delta_search_lower_override)
            # Apply the override by taking the max with the current lower bound
            range_after_manual['delta_lower'] = max(range_after_manual['delta_lower'], override_delta_lower)
            override_applied = True
        if args.delta_search_upper_override is not None:
            range_after_manual['delta_upper'] = args.delta_search_upper_override
            override_applied = True

        if override_applied:
             print("Range after Manual Overrides:")
             print(f"  y: [{range_after_manual['y_lower']:.4g}, {range_after_manual['y_upper']:.4g}]")
             print(f"  δ: [{range_after_manual['delta_lower']:.4g}, {range_after_manual['delta_upper']:.4g}]")
        else:
             print("No manual overrides specified.")

        # 4. Check Validity
        is_valid = True
        if not heuristic_possible: is_valid = False # Heuristic already failed
        if range_after_manual['y_lower'] >= range_after_manual['y_upper'] - eps_bound:
            print(f"Range Check FAIL: y lower >= y upper")
            is_valid = False
        if range_after_manual['delta_lower'] >= range_after_manual['delta_upper'] - eps_bound:
            print(f"Range Check FAIL: delta lower >= delta upper")
            is_valid = False

        # 5. Decide
        if is_valid:
            final_range = range_after_manual
            final_target_used = current_target
            print(f"\n--- Found Valid Search Range for Target >= {final_target_used:.3f} ---")
            print("Final Search Range:")
            print(f"  y: [{final_range['y_lower']:.4g}, {final_range['y_upper']:.4g}]")
            print(f"  δ: [{final_range['delta_lower']:.4g}, {final_range['delta_upper']:.4g}]")
            print("-" * 30)
            break # Exit while loop
        else:
            print(f"Search range became invalid for target >= {current_target:.3f}.")
            current_target -= TARGET_REDUCTION_STEP
            if current_target <= MINIMUM_TARGET_THRESHOLD:
                print(f"\nTarget ({current_target:.3f}) fell below threshold ({MINIMUM_TARGET_THRESHOLD:.3f}). Cannot find valid range. Aborting.")
                sys.exit(1)
            else:
                print(f"Reducing target to {current_target:.3f} and retrying...")

    if final_range is None:
         print("\nError: Loop finished without finding valid range. Aborting.", file=sys.stderr)
         sys.exit(1)

    # --- Optimization ---
    print(f"--- Optimizing y and δ within Final Range ---")
    bounds_final = [ (final_range['y_lower'], final_range['y_upper']),
                     (final_range['delta_lower'], final_range['delta_upper']) ]

    # Define initial guess
    if args.y_start is not None and bounds_final[0][0] <= args.y_start <= bounds_final[0][1]: y0 = args.y_start
    else: y0 = max(bounds_final[0][0], min(0.0, bounds_final[0][1]))
    if args.delta_start is not None and bounds_final[1][0] <= args.delta_start <= bounds_final[1][1]: delta0 = args.delta_start
    else: delta0 = max(bounds_final[1][0], min(0.0, bounds_final[1][1]))

    initial_guess = [y0, delta0]
    initial_guess[0] = np.clip(initial_guess[0], bounds_final[0][0] + eps_bound, bounds_final[0][1] - eps_bound)
    initial_guess[1] = np.clip(initial_guess[1], bounds_final[1][0] + eps_bound, bounds_final[1][1] - eps_bound)
    print(f"Initial Guess (adjusted): y={initial_guess[0]:.4g}, δ={initial_guess[1]:.4g}")
    print(f"Using Method: {args.optim_method}, Tolerances: ftol={args.ftol:.1e}, gtol={args.gtol:.1e}")

    optim_options = {'disp': False, 'ftol': args.ftol, 'gtol': args.gtol}
    try:
        optim_result = minimize( objective_function, initial_guess, args=(constants,),
                                 method=args.optim_method, bounds=bounds_final, options=optim_options )

        print("\n--- Optimization Result ---")
        print(f"Success: {optim_result.success}")
        if hasattr(optim_result, 'message'): print(f"Message: {optim_result.message}")
        if not optim_result.success: print("Warning: Optimization may not have converged optimally.")

        optimal_y, optimal_delta = optim_result.x
        max_lower_bound_neg = optim_result.fun

        if max_lower_bound_neg >= 1e9: # Check against penalty
             print("\nError: Optimizer finished in penalized region (bound effectively zero).")
             optimal_y, optimal_delta = initial_guess # Report initial guess
             max_lower_bound = 0.0
        else:
             max_lower_bound = -max_lower_bound_neg

        print("\n--- Final Optimized Bound ---")
        print(f"Effective Target Used for Range: {final_target_used:.3f}")
        print(f"Optimal y found: {optimal_y:.6f}")
        print(f"Optimal δ found: {optimal_delta:.6f}")
        print(f"Maximum Lower Bound P(T1 ⊆ T2) ≳ {max(0.0, max_lower_bound):.6e}")

        if max_lower_bound < final_target_used - 1e-3 : # Allow tolerance
              print(f"\nNote: Final optimized bound ({max_lower_bound:.4f}) is below the effective target ({final_target_used:.3f}).")

    except Exception as e:
        print(f"\n--- Optimization Failed ---")
        print(f"Error during optimization: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()