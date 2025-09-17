import pandas as pd
import numpy as np
import warnings
import traceback 
import matplotlib.pyplot as plt 

# ---  Pymatgen  ---
try:
    from pymatgen.core import Element
    _pymatgen_found = True
    print("Pymatgen found. Will use it for atomic volume.")
except ImportError:
    _pymatgen_found = False
    print("Pymatgen not found. Using hardcoded approximate values for V.")


# --- constant ---
kB = 8.6173303e-5  # Boltzmann's constant (eV/K)
alpha = 1/12     # Line tension prefactor
eps_dot_0 = 1e4   # Reference strain rate (s^-1)
TF = 3.067         # Taylor factor
T_test = 300.0     # Test temperature (K) 
eps_dot = 1e-5     # Applied strain rate (s^-1) 

# --- 3 ---
elements = ['V', 'Nb', 'Ta', 'Ti', 'Zr', 'Hf', 'Cr', 'Mo', 'W'] 

#  BCC  (A^3/atom)
element_V_bcc = {
    'V': 14.020, 'Nb': 17.952, 'Ta': 17.985, 'Ti': 17.387, 'Zr': 23.020, 
    'Hf': 22.528, 'Cr': 12.321, 'Mo': 15.524, 'W': 15.807
} 

#  Cij
element_C11 = { 'V': 232.4, 'Nb': 252.7, 'Ta': 266.32, 'Ti': 134.0, 'Zr': 104.0, 'Hf': 131.0, 'Cr': 339.8, 'Mo': 450.02, 'W': 532.55 }
element_C12 = { 'V': 119.36, 'Nb': 133.2, 'Ta': 158.16, 'Ti': 110.0, 'Zr': 93.0, 'Hf': 103.0, 'Cr': 58.6, 'Mo': 172.92, 'W': 204.95 }
element_C44 = { 'V': 45.95, 'Nb': 30.97, 'Ta': 87.36, 'Ti': 36.0, 'Zr': 38.0, 'Hf': 45.0, 'Cr': 99.0, 'Mo': 125.03, 'W': 163.13 }

# --- a_bcc , K_bcc ---
V_elem_bcc = np.array([element_V_bcc[el] for el in elements])
C11_elem = np.array([element_C11[el] for el in elements])
C12_elem = np.array([element_C12[el] for el in elements])

# a = (2*V_atomic)^(1/3) for BCC
a_elem_bcc = (2 * V_elem_bcc)**(1/3) 
# K = (C11 + 2*C12) / 3
K_elem_bcc = (C11_elem + 2 * C12_elem) / 3

print("Pre-calculated pure element properties (BCC):")
print(pd.DataFrame({
    'Element': elements,
    'V_bcc (A^3)': V_elem_bcc,
    'a_bcc (A)': a_elem_bcc,
    'K_bcc (GPa)': K_elem_bcc
}))
excel_filename = 'elastic_constant_data1.xlsx'
composition_cols = elements 
alloy_c11_col = 'C11 (GPa)' # DFT C11
alloy_c12_col = 'C12 (GPa)' # DFT C12
alloy_c44_col = 'C44 (GPa)' # DFT C44

alloy_a_dft_col = 'a_alloy_dft (A)'


print(f"\nLoading alloy data from: {excel_filename}")
try:
    df_alloys = pd.read_excel(excel_filename)
    print(f"Successfully loaded alloy data. Shape: {df_alloys.shape}")
    ### MODIFIED ###
    required_cols = composition_cols + [alloy_c11_col, alloy_c12_col, alloy_c44_col, alloy_a_dft_col]
    missing_data_cols = [col for col in required_cols if col not in df_alloys.columns]
    if missing_data_cols: raise ValueError(f"Missing required columns: {missing_data_cols}")
except FileNotFoundError: raise SystemExit(f"Error: File not found at {excel_filename}")
except Exception as e: traceback.print_exc(); raise SystemExit(f"Error loading alloy data: {e}")

try:
    C = df_alloys[composition_cols].values
    C11_dft = df_alloys[alloy_c11_col].values 
    C12_dft = df_alloys[alloy_c12_col].values
    C44_dft = df_alloys[alloy_c44_col].values
    a_alloy_dft_values = df_alloys[alloy_a_dft_col].values

except KeyError as e: raise SystemExit(f"Error accessing columns: {e}.")
except Exception as e: raise SystemExit(f"Error processing data columns: {e}")


valid_rows_idx = (np.abs(np.sum(C, axis=1) - 1.0) < 1e-3) | (np.abs(np.sum(C, axis=1) - 100.0) < 1e-1)
if not np.all(valid_rows_idx):
    warnings.warn(f"Composition data in {np.sum(~valid_rows_idx)} rows does not sum to 1 or 100. Excluding these rows.")
    original_index = df_alloys.index[valid_rows_idx] 
    C = C[valid_rows_idx]
    C11_dft = C11_dft[valid_rows_idx]
    C12_dft = C12_dft[valid_rows_idx]
    C44_dft = C44_dft[valid_rows_idx]
    df_alloys_filtered = df_alloys.loc[original_index].copy() 
else:
    df_alloys_filtered = df_alloys.copy()
    original_index = df_alloys.index 

if np.all(np.abs(np.sum(C, axis=1) - 100.0) < 1e-1):
    print("Converting composition from percent to fractions...")
    C = C / 100.0

print(f"Processed composition matrix C shape after filtering/conversion: {C.shape}")
if C.shape[0] == 0: raise SystemExit("Stopping: No valid alloy compositions found after filtering.")

# --- 6. 计算 RoM Cij 和 两种预测屈服强度 ---
num_alloys = C.shape[0]
predicted_YS_T_dft = np.zeros(num_alloys) #  DFT Cij  YS
predicted_YS_T_rom = np.zeros(num_alloys) #  RoM Cij YS
C11_rom_results = np.zeros(num_alloys)
C12_rom_results = np.zeros(num_alloys)
C44_rom_results = np.zeros(num_alloys)

def calculate_properties_lubardo(c_row, a_alloy_dft, C11_in, C12_in, C44_in, a_elem_bcc, K_elem_bcc):
    
    # --- 1. μ_alloy, K_alloy ---
    if C11_in <= 0 or C44_in <= 0 or C11_in <= abs(C12_in) or (C11_in + 2*C12_in) <= 0:
        return {'YS_T': np.nan} # Cij 
        
    K_alloy = (C11_in + 2 * C12_in) / 3
    Cprime_alloy = (C11_in - C12_in) / 2
    if Cprime_alloy <= 1e-9 or K_alloy <= 1e-9: return {'YS_T': np.nan}
        
    mu_alloy = np.sqrt(C44_in * Cprime_alloy)

    # --- Lubardo ---
    # γ_alloy = (1 + 4μ_alloy / 3K_alloy)
    gamma_alloy = 1.0 + (4.0 * mu_alloy) / (3.0 * K_alloy)
    # γᵢ = (1 + 4μ_alloy / 3Kᵢ)
    gamma_i_vec = 1.0 + (4.0 * mu_alloy) / (3.0 * K_elem_bcc)
    
    # ΔVᵢ = 0.5 * (aᵢ³ - a_alloy³) * (γ_alloy / γᵢ)
    # a_alloy_dft is the true lattice parameter of the alloy
    vol_diff_uncorrected = 0.5 * (a_elem_bcc**3 - a_alloy_dft**3)
    lubardo_correction_factor = gamma_alloy / gamma_i_vec
    
    DV_lubardo_vec = vol_diff_uncorrected * lubardo_correction_factor
    
    # --- 3.  Maresca-Curtin ---
    sigma_DV_sq = np.sum(c_row * (DV_lubardo_vec**2))
    if sigma_DV_sq < 0: sigma_DV_sq = 0.0

    b_alloy = (np.sqrt(3) / 2) * a_alloy_dft
    if b_alloy <= 1e-9: return {'YS_T': np.nan}
    
    # --- 4. YS  ---
    denominator_nu = 2 * (3 * K_alloy + mu_alloy)
    if abs(denominator_nu) < 1e-9: return {'YS_T': np.nan}
    nu_alloy = (3 * K_alloy - 2 * mu_alloy) / denominator_nu
    if nu_alloy >= 0.5 or nu_alloy <= -1.0 or abs(1.0 - nu_alloy) < 1e-9: return {'YS_T': np.nan}
    
    nu_term_tau_base = (1 + nu_alloy) / (1 - nu_alloy)
    if nu_term_tau_base <= 0: return {'YS_T': np.nan}
    
    nu_term_tau = nu_term_tau_base**(4/3)
    nu_term_DEb = nu_term_tau_base**(2/3)
    
    misfit_term_base = sigma_DV_sq / (b_alloy**6)
    if misfit_term_base < 0: misfit_term_base = 0.0
    misfit_term_tau = misfit_term_base**(2/3)
    misfit_term_DEb = misfit_term_base**(1/3)

    tau_0 = 0.04 * alpha**(-1/3) * mu_alloy * nu_term_tau * misfit_term_tau
    
    GPa_A3_to_eV = 1 / 160.21766208 
    DEb_GPa_A3 = 2 * alpha**(1/3) * mu_alloy * b_alloy**3 * nu_term_DEb * misfit_term_DEb
    DEb = DEb_GPa_A3 * GPa_A3_to_eV 
    
    if DEb <= 1e-9 or T_test <=0:
        tau_T = tau_0
    else:
        log_term = np.log(eps_dot_0 / eps_dot)
        if log_term < 0:
            tau_T = tau_0
        else:
            thermal_term_base = (kB * T_test / DEb) * log_term
            exponent_arg = (1.0 / 0.55) * (thermal_term_base**0.91)
            try:
                tau_T = tau_0 * np.exp(-exponent_arg)
            except OverflowError:
                tau_T = 0.0

    YS_T = TF * tau_T
    

    return {'YS_T': YS_T, 'sigma_DV_sq_lubardo': sigma_DV_sq}


predicted_YS_T_dft = np.zeros(num_alloys)
predicted_YS_T_rom = np.zeros(num_alloys)

print("\nStarting calculations with Lubardo Correction...")
for i in range(num_alloys):
    c_row = C[i, :] 
    a_alloy_dft_current = a_alloy_dft_values[i]

    C11_rom = np.sum(c_row * C11_elem)
    C12_rom = np.sum(c_row * C12_elem)
    C44_rom = np.sum(c_row * C44_elem)

    ### MODIFIED ###
    # ---  DFT Cij ,DFT a_alloy , YS ---
    dft_props = calculate_properties_lubardo(c_row, a_alloy_dft_current, C11_dft[i], C12_dft[i], C44_dft[i], a_elem_bcc, K_elem_bcc)
    predicted_YS_T_dft[i] = dft_props['YS_T']
    
    # --- RoM Cij , DFT a_alloy , YS ---

    rom_props = calculate_properties_lubardo(c_row, a_alloy_dft_current, C11_rom, C12_rom, C44_rom, a_elem_bcc, K_elem_bcc)
    predicted_YS_T_rom[i] = rom_props['YS_T']

    if (i + 1) % 400 == 0: 
        print(f"Processed {i+1}/{num_alloys} alloys...")

print("\nCalculations complete.")


df_alloys_filtered['YS_Predicted_DFT_Lubardo (GPa)'] = pd.Series(predicted_YS_T_dft, index=original_index)
df_alloys_filtered['YS_Predicted_RoM_Lubardo (GPa)'] = pd.Series(predicted_YS_T_rom, index=original_index)

print("\nComparing RoM vs DFT calculated elastic constants:")
compare_df = df_alloys_filtered.dropna(subset=[
    alloy_c11_col, alloy_c12_col, alloy_c44_col, 
    'C11_RoM (GPa)', 'C12_RoM (GPa)', 'C44_RoM (GPa)' 
]) 

print("\nDifference statistics (DFT - RoM) for Cij:")
if not compare_df.empty:
    print("C11 Diff:", (compare_df[alloy_c11_col] - compare_df['C11_RoM (GPa)']).describe())
    print("\nC12 Diff:", (compare_df[alloy_c12_col] - compare_df['C12_RoM (GPa)']).describe())
    print("\nC44 Diff:", (compare_df[alloy_c44_col] - compare_df['C44_RoM (GPa)']).describe())
else:
     print("Not enough valid data points for Cij comparison statistics.")

if not compare_df.empty:
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.scatter(compare_df['C11_RoM (GPa)'], compare_df[alloy_c11_col], alpha=0.5)
    min_c11 = min(compare_df['C11_RoM (GPa)'].min(), compare_df[alloy_c11_col].min())
    max_c11 = max(compare_df['C11_RoM (GPa)'].max(), compare_df[alloy_c11_col].max())
    plt.plot([min_c11, max_c11], [min_c11, max_c11], 'r--', label='Ideal y=x')
    plt.xlabel("C11 RoM (GPa)"); plt.ylabel("C11 DFT (GPa)")
    plt.title("C11: DFT vs Rule of Mixtures"); plt.grid(True, alpha=0.5); plt.legend()
    plt.subplot(1, 3, 2)
    plt.scatter(compare_df['C12_RoM (GPa)'], compare_df[alloy_c12_col], alpha=0.5)
    min_c12 = min(compare_df['C12_RoM (GPa)'].min(), compare_df[alloy_c12_col].min())
    max_c12 = max(compare_df['C12_RoM (GPa)'].max(), compare_df[alloy_c12_col].max())
    plt.plot([min_c12, max_c12], [min_c12, max_c12], 'r--', label='Ideal y=x')
    plt.xlabel("C12 RoM (GPa)"); plt.ylabel("C12 DFT (GPa)")
    plt.title("C12: DFT vs Rule of Mixtures"); plt.grid(True, alpha=0.5); plt.legend()
    plt.subplot(1, 3, 3)
    plt.scatter(compare_df['C44_RoM (GPa)'], compare_df[alloy_c44_col], alpha=0.5)
    min_c44 = min(compare_df['C44_RoM (GPa)'].min(), compare_df[alloy_c44_col].min())
    max_c44 = max(compare_df['C44_RoM (GPa)'].max(), compare_df[alloy_c44_col].max())
    plt.plot([min_c44, max_c44], [min_c44, max_c44], 'r--', label='Ideal y=x')
    plt.xlabel("C44 RoM (GPa)"); plt.ylabel("C44 DFT (GPa)")
    plt.title("C44: DFT vs Rule of Mixtures"); plt.grid(True, alpha=0.5); plt.legend()

    plt.tight_layout(); plt.show()
else:
     print("\nSkipping Cij comparison plots due to lack of valid data.")

print("\nComparing Yield Strength predicted using DFT Cij vs RoM Cij:")
compare_ys_df = df_alloys_filtered.dropna(subset=['YS_Predicted_DFT (GPa)', 'YS_Predicted_RoM (GPa)'])

print("\nDifference statistics (YS DFT - YS RoM):")
if not compare_ys_df.empty:
    print((compare_ys_df['YS_Predicted_DFT (GPa)'] - compare_ys_df['YS_Predicted_RoM (GPa)']).describe())
else:
     print("Not enough valid data points for YS comparison statistics.")

if not compare_ys_df.empty:
    plt.figure(figsize=(7, 7))
    plt.scatter(compare_ys_df['YS_Predicted_RoM (GPa)'], compare_ys_df['YS_Predicted_DFT (GPa)'], alpha=0.5)
    min_ys = min(compare_ys_df['YS_Predicted_RoM (GPa)'].min(), compare_ys_df['YS_Predicted_DFT (GPa)'].min())
    max_ys = max(compare_ys_df['YS_Predicted_RoM (GPa)'].max(), compare_ys_df['YS_Predicted_DFT (GPa)'].max())
    plt.plot([min_ys, max_ys], [min_ys, max_ys], 'r--', label='Ideal y=x')
    plt.xlabel("Predicted YS using RoM Cij (GPa)")
    plt.ylabel("Predicted YS using DFT Cij (GPa)")
    plt.title("Yield Strength Prediction: DFT Cij vs RoM Cij")
    plt.grid(True, alpha=0.5)
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
else:
     print("\nSkipping YS comparison plot due to lack of valid data.")

print("\nIdentifying alloys with potentially exaggerated YS predictions...")
high_YS_threshold = 2.0

df_to_check = df_alloys_filtered if 'df_alloys_filtered' in locals() else df_alloys

ys_col_name = 'YS_Predicted_DFT (GPa)' if 'YS_Predicted_DFT (GPa)' in df_to_check.columns else 'Predicted_YS_T_Maresca (GPa)'
cols_to_print = composition_cols + [alloy_c11_col, alloy_c12_col, alloy_c44_col, ys_col_name]

high_YS_alloys = df_to_check[df_to_check[ys_col_name] > high_YS_threshold]

if not high_YS_alloys.empty:
    print(f"\nFound {len(high_YS_alloys)} alloys with Predicted YS > {high_YS_threshold} GPa:")
    print(high_YS_alloys[cols_to_print].to_string()) 

else:
    print(f"\nNo alloys found with Predicted YS > {high_YS_threshold} GPa.")
print("\nFinal DataFrame with RoM Cij and Both Predicted Yield Strengths (first 10 rows):")
display_cols = composition_cols + [alloy_c11_col, 'C11_RoM (GPa)', 
                                   alloy_c12_col, 'C12_RoM (GPa)', 
                                   alloy_c44_col, 'C44_RoM (GPa)', 
                                   'YS_Predicted_DFT (GPa)', 'YS_Predicted_RoM (GPa)']
# Display from the potentially filtered dataframe
print(df_alloys_filtered[display_cols].head(10))

print("\nStatistics for Predicted Yield Strength (based on DFT Cij, excluding NaNs):")
print(df_alloys_filtered['YS_Predicted_DFT (GPa)'].describe())
print("\nStatistics for Predicted Yield Strength (based on RoM Cij, excluding NaNs):")
print(df_alloys_filtered['YS_Predicted_RoM (GPa)'].describe())

try:
    output_filename = 'elastic_data_with_RoM_DFT_YS_compare.xlsx'
    df_alloys_filtered.to_excel(output_filename, index=False) 
    print(f"\nResults saved to {output_filename}")
except Exception as e:
    print(f"\nError saving results to Excel: {e}")