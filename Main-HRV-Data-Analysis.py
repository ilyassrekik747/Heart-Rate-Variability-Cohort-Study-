
import os, re, warnings, time
import numpy as np
import pandas as pd
from scipy import signal, interpolate, stats
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
SUBJ_PATH = r"C:\Users\ilyas\Desktop\holter\subject-info.csv"
RECORD_DIR = r"C:\Users\ilyas\Desktop\holter\records"
OUTPUT_DIR = r"C:\Users\ilyas\Desktop\holter\in"
MAX_SUBJECTS = 2000
os.makedirs(OUTPUT_DIR, exist_ok=True)

def try_read_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    parsers = [{"sep": None, "engine": "python"}, {"sep": ",", "engine": "python"}, {"sep": ";", "engine": "python"},
               {"sep": "\t", "engine": "python"}]
    for p in parsers:
        try:
            df = pd.read_csv(path, **p)
            print(f"✓ Loaded {path} (sep={p['sep']}) shape={df.shape}")
            return df
        except Exception:
            continue
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip() != ""]
    parts = [ln.split(",") for ln in lines]
    header = parts[0]
    rows = parts[1:]
    df = pd.DataFrame(rows, columns=header)
    print(f"✓ Loaded {path} with fallback splitter shape={df.shape}")
    return df


def extract_first_int(x):
    if pd.isna(x): return np.nan
    s = str(x)
    m = re.search(r"(-?\d+)", s)
    return int(m.group(1)) if m else np.nan
def decode_outcome_row(row, col_D, col_E):
    D = extract_first_int(row.get(col_D, np.nan)) if col_D is not None else np.nan
    E = extract_first_int(row.get(col_E, np.nan)) if col_E is not None else np.nan

    if not np.isnan(E):
        if int(E) == 3:
            return "SCD"
        if int(E) in (6, 7):
            return "PumpFailure"
        if int(E) == 1:
            return "NonCardiacDeath"
        if int(E) == 0:
            return "Survivor"
        return f"Other_E_{int(E)}"
    if not np.isnan(D):
        if int(D) == 3:
            return "Death_unknown_cause"
        if int(D) == 2:
            return "Transplant"
        if int(D) == 1:
            return "LostToFollowUp"
        if int(D) == 0:
            return "Survivor"
    return "Unknown"

def samples_to_rr(rpeaks_samples, fs):
    rr_samples = np.diff(rpeaks_samples)
    rr_sec = rr_samples / float(fs)
    return rr_sec


def rr_postprocess(rr_sec, min_rr=0.3, max_rr=2.0):
    rr = np.array(rr_sec, dtype=float)
    if rr.size == 0: return rr, np.array([])
    mask = (rr >= min_rr) & (rr <= max_rr)
    removed = np.where(~mask)[0]
    if removed.size == 0: return rr, removed
    valid_idx = np.where(mask)[0]
    if valid_idx.size < 2:
        rr_clipped = np.clip(rr, min_rr, max_rr)
        return rr_clipped, removed
    interp_func = interpolate.interp1d(valid_idx, rr[valid_idx], kind='linear', fill_value='extrapolate')
    rr_interp = rr.copy()
    rr_interp[removed] = interp_func(removed)
    return rr_interp, removed


def compute_time_domain(rr_sec):
    if rr_sec.size == 0:
        return {'mean_rr_ms': np.nan, 'sdnn_ms': np.nan, 'rmssd_ms': np.nan, 'nn50': np.nan, 'pnn50_pct': np.nan,
                'sdsd_ms': np.nan}

    rr_ms = rr_sec * 1000.0
    mean_rr = np.mean(rr_ms)
    sdnn = np.std(rr_ms, ddof=1)
    diff_ms = np.diff(rr_ms)
    rmssd = np.sqrt(np.mean(diff_ms ** 2)) if diff_ms.size > 0 else np.nan

    nn50 = np.sum(np.abs(diff_ms) > 50) if diff_ms.size > 0 else np.nan
    pnn50 = (nn50 / len(diff_ms)) * 100 if diff_ms.size > 0 else np.nan

    sdsd = np.std(diff_ms, ddof=1) if diff_ms.size > 0 else np.nan

    return {
        'mean_rr_ms': mean_rr,
        'sdnn_ms': sdnn,
        'rmssd_ms': rmssd,
        'nn50': nn50,
        'pnn50_pct': pnn50,
        'sdsd_ms': sdsd
    }


def compute_frequency_domain(rr_sec, resample_rate=4.0):
    if rr_sec.size < 4:
        return {'lf_ms2': np.nan, 'hf_ms2': np.nan, 'total_ms2': np.nan, 'lf_nu_pct': np.nan}
    times = np.cumsum(rr_sec)
    times = np.insert(times, 0, 0.0)[:-1]
    total_time = times[-1] + rr_sec[-1]
    fs_interp = float(resample_rate)
    t_interp = np.arange(0, total_time, 1.0 / fs_interp)
    interp_func = interpolate.interp1d(times, rr_sec, kind='cubic', fill_value='extrapolate')
    rr_interp = interp_func(t_interp)
    rr_detrended = signal.detrend(rr_interp)
    nperseg = min(256, len(rr_detrended))
    f, pxx = signal.welch(rr_detrended, fs=fs_interp, nperseg=nperseg)

    def band_power(b):
        idx = np.logical_and(f >= b[0], f < b[1])
        if idx.sum() == 0: return np.nan
        return np.trapz(pxx[idx], f[idx])

    vlf = band_power((0.0033, 0.04));
    lf = band_power((0.04, 0.15));
    hf = band_power((0.15, 0.4))
    total = np.nansum([vlf, lf, hf])
    lf_nu = 100.0 * lf / (lf + hf) if (not np.isnan(lf) and not np.isnan(hf) and (lf + hf) > 0) else np.nan
    return {'lf_ms2': lf, 'hf_ms2': hf, 'total_ms2': total, 'lf_nu_pct': lf_nu}


def compute_geometric_measures(rr_sec):
    """Calculate geometric HRV measures: SDNN index, TINN"""
    if rr_sec.size < 10:
        return {'sdnn_index': np.nan, 'tinn_ms': np.nan}

    rr_ms = rr_sec * 1000.0
    segment_duration = 120  
    total_duration = np.sum(rr_sec)

    if total_duration >= segment_duration:
        segment_sdnn_values = []
        current_start = 0
        cumulative_time = np.cumsum(rr_sec)

        while current_start < len(rr_sec):
            segment_end_idx = np.searchsorted(cumulative_time, cumulative_time[current_start] + segment_duration)
            if segment_end_idx > current_start + 1:  # Need at least 2 beats
                segment_rr = rr_ms[current_start:segment_end_idx]
                segment_sdnn_values.append(np.std(segment_rr, ddof=1))
            current_start = segment_end_idx

        sdnn_index = np.mean(segment_sdnn_values) if segment_sdnn_values else np.nan
    else:
        sdnn_index = np.nan
    hist, bin_edges = np.histogram(rr_ms, bins=min(20, len(rr_ms) // 10), density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    if len(hist) >= 3:

        max_bin_idx = np.argmax(hist)
        tinn = bin_edges[-1] - bin_edges[0]
    else:
        tinn = np.nan

    return {'sdnn_index': sdnn_index, 'tinn_ms': tinn}


def compute_dfa(rr_sec):
    """Calculate Detrended Fluctuation Analysis (DFA) - alpha1 and alpha2"""
    if rr_sec.size < 100:
        return {'dfa_alpha1': np.nan, 'dfa_alpha2': np.nan}

    rr_ms = rr_sec * 1000.0
    y = np.cumsum(rr_ms - np.mean(rr_ms))

    scales1 = np.arange(4, 17) 
    scales2 = np.arange(16, 65)  

    def calculate_fluctuation(scales):
        fluctuations = []
        for n in scales:
            if n > len(y):
                continue
            segments = len(y) // n
            if segments < 2:
                continue
            f_n = 0
            for i in range(segments):
                segment = y[i * n:(i + 1) * n]
                if len(segment) < 2:
                    continue
                x_seg = np.arange(len(segment))
                coeffs = np.polyfit(x_seg, segment, 1)
                trend = np.polyval(coeffs, x_seg)
                detrended = segment - trend

                f_n += np.mean(detrended ** 2)

            if segments > 0:
                f_n = np.sqrt(f_n / segments)
                fluctuations.append(f_n)

        if len(fluctuations) < 2:
            return np.nan, np.nan

        log_scales = np.log(scales[:len(fluctuations)])
        log_fluctuations = np.log(fluctuations)

        if len(log_scales) >= 2:
            slope, intercept = np.polyfit(log_scales, log_fluctuations, 1)
            return slope
        else:
            return np.nan
    dfa_alpha1 = calculate_fluctuation(scales1)
    dfa_alpha2 = calculate_fluctuation(scales2)
    return {'dfa_alpha1': dfa_alpha1, 'dfa_alpha2': dfa_alpha2}

def sample_entropy_fast(rr, m=2, maxlen=200):
    rr = np.asarray(rr)
    if rr.size < m + 1: return np.nan
    if rr.size > maxlen:
        idx = np.linspace(0, rr.size - 1, maxlen).astype(int)
        rr_ds = rr[idx]
    else:
        rr_ds = rr
    r = 0.2 * np.std(rr_ds)
    N = rr_ds.size

    def _count(m_):
        x = np.array([rr_ds[i:i + m_] for i in range(N - m_ + 1)])
        count = 0
        for i in range(len(x)):
            d = np.max(np.abs(x - x[i]), axis=1)
            count += np.sum(d <= r) - 1
        return count

    try:
        A = _count(m + 1);
        B = _count(m)
        if A == 0 or B == 0: return np.nan
        return -np.log(A / B)
    except Exception:
        return np.nan


def compute_hourly_lf_instability(rpeaks, fs):
    peak_times = rpeaks / float(fs)
    hourly_lfs = []
    start = 0.0
    if len(peak_times) < 3:
        return np.nan
    while start + 3600 <= peak_times[-1] + 1e-6:
        idx = np.where((peak_times >= start) & (peak_times < start + 3600))[0]
        if idx.size >= 3:
            rr = samples_to_rr(rpeaks[idx], fs)
            rr, _ = rr_postprocess(rr)
            fd = compute_frequency_domain(rr)
            hourly_lfs.append(fd['lf_ms2'])
        start += 3600
    arr = np.array(hourly_lfs, dtype=float)
    valid = ~np.isnan(arr)
    if valid.sum() >= 2 and np.nanmean(arr[valid]) != 0:
        return np.nanstd(arr[valid], ddof=1) / np.nanmean(arr[valid])
    return np.nan
def main():
    total_start = time.time()
    print("=" * 60)
    print(" Starting PhysioNet HRV Pipeline")
    print("=" * 60)

    step_start = time.time()
    print("\n[1/10] Loading subject-info.csv...")
    subj = try_read_csv(SUBJ_PATH)
    subj.columns = [c.strip() for c in subj.columns]
    print(f" Subject-info loaded: {subj.shape[0]} rows, {subj.shape[1]} columns")
    print("Columns:", list(subj.columns)[:20])
    print(f"  Step 1 completed in {time.time() - step_start:.2f} seconds")

    step_start = time.time()
    print("\n[2/10] Identifying outcome code columns (D=Exit, E=Cause)...")
    cols = subj.columns.tolist()
    col_D = col_E = None
    for c in cols:
        cl = c.lower()
        if "exit" in cl or c.strip().upper() == "D":
            col_D = c
        if "cause" in cl or "cause of death" in cl or c.strip().upper() == "E":
            col_E = c
    if col_D is None and "D" in cols: col_D = "D"
    if col_E is None and "E" in cols: col_E = "E"
    print(f" Using columns -> D: '{col_D}', E: '{col_E}'")
    print(f"  Step 2 completed in {time.time() - step_start:.2f} seconds")

    step_start = time.time()
    print("\n[3/10] Decoding outcome labels for all subjects...")
    subj['outcome_label'] = subj.apply(lambda r: decode_outcome_row(r, col_D, col_E), axis=1)
    print(f" Outcome labels assigned. Unique outcomes: {subj['outcome_label'].nunique()}")
    print(f"  Step 3 completed in {time.time() - step_start:.2f} seconds")

    step_start = time.time()
    print("\n[4/10] Initializing HRV result columns...")
    subj = subj.head(MAX_SUBJECTS).copy()
    subj['has_ecg'] = False

    hrv_columns = [
        'sdnn_ms', 'rmssd_ms', 'lf_ms2', 'hf_ms2', 'total_ms2', 'lf_nu_pct', 'lf_instability',
        'sample_entropy', 'nn50', 'pnn50_pct', 'sdsd_ms', 'sdnn_index', 'tinn_ms', 'dfa_alpha1', 'dfa_alpha2'
    ]

    for col in hrv_columns:
        subj[col] = np.nan

    print(f" Initialized HRV columns for {len(subj)} subjects")
    print(f"  Step 4 completed in {time.time() - step_start:.2f} seconds")

    step_start = time.time()
    print("\n[5/10] Scanning for WFDB .dat records...")
    if os.path.isdir(RECORD_DIR):
        record_files = {os.path.splitext(f)[0] for f in os.listdir(RECORD_DIR) if f.endswith(".dat")}
    else:
        record_files = set()
    print(f" Found {len(record_files)} WFDB records in '{RECORD_DIR}'")
    print(f"  Step 5 completed in {time.time() - step_start:.2f} seconds")

    step_start = time.time()
    total_to_process = len(subj)
    print(f"\n[6/10] Processing ECGs for subjects with records (up to {total_to_process} subjects)...")
    try:
        import wfdb
        wfdb_available = True
    except Exception as e:
        wfdb_available = False
        print(f"  wfdb not available: {e}. Skipping ECG processing.")

    processed_count = 0
    error_count = 0

    for idx, row in subj.iterrows():
        rec_id = str(row["record_id"]) if "record_id" in subj.columns else str(row[subj.columns[0]])

        if rec_id in record_files and wfdb_available:
            record_path = os.path.join(RECORD_DIR, rec_id)
            try:
                record = wfdb.rdrecord(record_path)
                sig = record.p_signal[:, 0]
                fs = int(record.fs)
                try:
                    ann = wfdb.rdann(record_path, 'atr')
                    rpeaks = np.array(ann.sample)
                except Exception:
                    b, a = signal.butter(3, [5 / (fs / 2), 15 / (fs / 2)], btype='band')
                    filt = signal.filtfilt(b, a, sig)
                    diff = np.ediff1d(filt, to_end=0)
                    sq = diff ** 2
                    win = max(1, int(0.150 * fs))
                    ma = np.convolve(sq, np.ones(win) / win, mode='same')
                    height = np.percentile(ma, 75)
                    peaks, _ = signal.find_peaks(ma, distance=int(0.3 * fs), height=height)
                    rpeaks = peaks
                rr = samples_to_rr(rpeaks, fs)
                rr, _ = rr_postprocess(rr)

                td = compute_time_domain(rr)
                fd = compute_frequency_domain(rr)
                geo = compute_geometric_measures(rr)
                dfa = compute_dfa(rr)
                instability = compute_hourly_lf_instability(rpeaks, fs)
                ent = sample_entropy_fast(rr)

                subj.at[idx, 'has_ecg'] = True
                subj.at[idx, 'sdnn_ms'] = td['sdnn_ms']
                subj.at[idx, 'rmssd_ms'] = td['rmssd_ms']
                subj.at[idx, 'lf_ms2'] = fd['lf_ms2']
                subj.at[idx, 'hf_ms2'] = fd['hf_ms2']
                subj.at[idx, 'total_ms2'] = fd['total_ms2']
                subj.at[idx, 'lf_nu_pct'] = fd['lf_nu_pct']
                subj.at[idx, 'lf_instability'] = instability
                subj.at[idx, 'sample_entropy'] = ent

                subj.at[idx, 'nn50'] = td['nn50']
                subj.at[idx, 'pnn50_pct'] = td['pnn50_pct']
                subj.at[idx, 'sdsd_ms'] = td['sdsd_ms']
                subj.at[idx, 'sdnn_index'] = geo['sdnn_index']
                subj.at[idx, 'tinn_ms'] = geo['tinn_ms']
                subj.at[idx, 'dfa_alpha1'] = dfa['dfa_alpha1']
                subj.at[idx, 'dfa_alpha2'] = dfa['dfa_alpha2']

                processed_count += 1
                print(f" Processed {rec_id}")

            except Exception as e:
                error_count += 1
                err_msg = str(e).replace('\n', ' ')[:200]
                print(f" Failed {rec_id}: {err_msg}")
        else:
 
            pass

    print(f" ECG processing done: {processed_count} successful, {error_count} failed")
    print(f"  Step 6 completed in {time.time() - step_start:.2f} seconds")


    step_start = time.time()
    print("\n[7/10] Saving merged subjects table (all subjects)...")
    merged_out = os.path.join(OUTPUT_DIR, "merged_subjects_with_hrv.csv")
    subj.to_csv(merged_out, index=False)
    print(f" Saved: {merged_out}")
    print(f"  Step 7 completed in {time.time() - step_start:.2f} seconds")

    step_start = time.time()
    print("\n[8/10] Saving HRV summary for processed subjects...")
    hrv_df = subj[subj['has_ecg']].copy()
    hrv_out = os.path.join(OUTPUT_DIR, "physionet_hrv_summary.csv")
    hrv_df.to_csv(hrv_out, index=False)
    print(f" Saved: {hrv_out} ({len(hrv_df)} subjects)")
    print(f"  Step 8 completed in {time.time() - step_start:.2f} seconds")

    step_start = time.time()
    print("\n[9/10] Performing group comparisons (SCD vs others, etc.)...")
    if not hrv_df.empty:
        groups = hrv_df.groupby('outcome_label')
        metrics = ['sdnn_ms', 'rmssd_ms', 'lf_ms2', 'lf_instability', 'sample_entropy',
                   'nn50', 'pnn50_pct', 'sdsd_ms', 'sdnn_index', 'tinn_ms', 'dfa_alpha1', 'dfa_alpha2']
        comp_rows = []
        labels = hrv_df['outcome_label'].unique().tolist()
        for m in metrics:
            for i in range(len(labels)):
                for j in range(i + 1, len(labels)):
                    a = hrv_df[hrv_df['outcome_label'] == labels[i]][m].dropna()
                    b = hrv_df[hrv_df['outcome_label'] == labels[j]][m].dropna()
                    if len(a) >= 3 and len(b) >= 3:
                        tstat, p = stats.ttest_ind(a, b, equal_var=False)
                    else:
                        tstat, p = np.nan, np.nan
                    comp_rows.append({
                        'metric': m, 'group1': labels[i], 'group2': labels[j],
                        'n1': len(a), 'n2': len(b),
                        'mean1': np.nanmean(a) if len(a) > 0 else np.nan,
                        'mean2': np.nanmean(b) if len(b) > 0 else np.nan,
                        'p': p
                    })
        comp_df = pd.DataFrame(comp_rows)
        comp_df.to_csv(os.path.join(OUTPUT_DIR, "group_comparisons.csv"), index=False)
        print(f" Saved group comparisons: {os.path.join(OUTPUT_DIR, 'group_comparisons.csv')}")
    else:
        print("  No processed subjects → skipping comparisons")
    print(f"  Step 9 completed in {time.time() - step_start:.2f} seconds")

    step_start = time.time()
    print("\n[10/10] Attempting logistic regression (SCD vs others)...")
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score, roc_curve
        df_model = hrv_df.dropna(subset=['sdnn_ms', 'lf_instability']).copy()
        if len(df_model) >= 10:
            df_model['y'] = df_model['outcome_label'].apply(lambda x: 1 if x == 'SCD' else 0)
            X = df_model[['sdnn_ms', 'lf_instability']].fillna(df_model.mean())
            y = df_model['y'].values
            clf = LogisticRegression(solver='liblinear').fit(X, y)
            probs = clf.predict_proba(X)[:, 1]
            auc = roc_auc_score(y, probs)
            fpr, tpr, _ = roc_curve(y, probs)
            plt.figure(figsize=(6, 6))
            plt.plot(fpr, tpr, label=f"Combined AUC={auc:.3f}")
            plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            plt.xlabel("False positive rate");
            plt.ylabel("True positive rate")
            plt.title("ROC: SDNN + LF Instability")
            plt.legend()
            plt.savefig(os.path.join(OUTPUT_DIR, "roc_combined.png"))
            plt.close()
            print(f"✓ Saved ROC plot: roc_combined.png | AUC = {auc:.3f}")
        else:
            print("  Not enough data (need ≥10 subjects with features)")
    except Exception as e:
        print(f"  Skipping logistic/ROC: {e}")
    print(f" Step 10 completed in {time.time() - step_start:.2f} seconds")

    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print(f" Pipeline completed successfully!")
    print(f" Total subjects: {len(subj)}")
    print(f" Subjects with ECG processed: {processed_count}")
    print(f" Output saved to: {OUTPUT_DIR}")
    print(f"  Total runtime: {total_time:.2f} seconds ({total_time / 60:.1f} minutes)")
    print("=" * 60)


if __name__ == "__main__":
    main()
