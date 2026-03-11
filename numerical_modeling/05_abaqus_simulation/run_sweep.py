# run_inp_sweep_post.py
# Run all .inp in a folder, then post-process RF3 vs U3 from RP_TOP (Step-2).
#
# Usage (Abaqus Command):
#   abaqus python run_inp_sweep_post.py
#   abaqus python run_inp_sweep_post.py "C:\path\to\inp_folder"
#
# Outputs:
#   results/summary.csv
#   results/<job>_RF3_U3.csv

import os
import sys
import glob
import time
import csv
import subprocess

# ----------------- Settings -----------------
CPUS = 14
STEP_NAME = "Step-2"
RP_TAG = "RP_TOP"
OUT_DIR = "results"
ONLY_PATTERN = "*.inp"   # change to "triax_*.inp" if you want to filter

# ----------------- Helpers -----------------
def run(cmd):
    print(">> " + " ".join(cmd))
    subprocess.check_call(cmd)

def open_csv(path):
    """Python2/3 safe CSV open."""
    if sys.version_info[0] < 3:
        return open(path, "wb")
    return open(path, "w", newline="")

def wait_for_job(job_name, timeout_sec=None):
    """
    Wait for Abaqus job to finish.
    We wait for the .lck to disappear. Optionally timeout.
    """
    lck = job_name + ".lck"
    odb = job_name + ".odb"
    sta = job_name + ".sta"

    t0 = time.time()

    # Wait a bit for lock to appear (job start)
    for _ in range(60):
        if os.path.exists(lck) or os.path.exists(sta) or os.path.exists(odb):
            break
        time.sleep(1.0)

    # Wait until lock disappears
    while os.path.exists(lck):
        time.sleep(5.0)
        if timeout_sec is not None and (time.time() - t0) > timeout_sec:
            return False

    # Give ODB a few seconds to flush
    for _ in range(30):
        if os.path.exists(odb):
            return True
        time.sleep(1.0)
        if timeout_sec is not None and (time.time() - t0) > timeout_sec:
            return False

    return os.path.exists(odb)

def postprocess_rf3_u3(odb_path, out_csv):
    """
    Extract U3 and RF3 from history output at RP_TOP in Step-2.
    Writes CSV with columns: U3, RF3.
    """
    from odbAccess import openOdb

    odb = openOdb(odb_path, readOnly=True)
    if STEP_NAME not in odb.steps:
        odb.close()
        raise RuntimeError("Step '%s' not found in %s" % (STEP_NAME, odb_path))

    step = odb.steps[STEP_NAME]

    # Find RP_TOP history region
    rp_region = None
    rp_region_name = None
    for name, reg in step.historyRegions.items():
        if RP_TAG in name.upper():
            rp_region = reg
            rp_region_name = name
            break

    if rp_region is None:
        odb.close()
        raise RuntimeError("History region containing '%s' not found in %s" % (RP_TAG, odb_path))

    # Find keys robustly (sometimes padded/spaces)
    def find_key(reg, target):
        t = target.strip().upper()
        for k in reg.historyOutputs.keys():
            if k.strip().upper() == t:
                return k
        return None

    k_rf3 = find_key(rp_region, "RF3")
    k_u3  = find_key(rp_region, "U3")
    if k_rf3 is None or k_u3 is None:
        odb.close()
        raise RuntimeError("RF3/U3 not found in historyOutputs (region=%s)" % rp_region_name)

    rf3 = rp_region.historyOutputs[k_rf3].data  # list of (time, value)
    u3  = rp_region.historyOutputs[k_u3].data

    with open_csv(out_csv) as f:
        w = csv.writer(f)
        w.writerow(["U3", "RF3"])
        n = min(len(rf3), len(u3))
        for i in range(n):
            w.writerow([u3[i][1], rf3[i][1]])

    odb.close()

def main():
    # Input folder
    if len(sys.argv) >= 2:
        inp_dir = sys.argv[1]
    else:
        inp_dir = os.getcwd()

    if not os.path.isdir(inp_dir):
        raise RuntimeError("Folder not found: %s" % inp_dir)

    # Results folder (inside inp_dir)
    out_dir = os.path.join(inp_dir, OUT_DIR)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    inps = sorted(glob.glob(os.path.join(inp_dir, ONLY_PATTERN)))
    if not inps:
        raise RuntimeError("No .inp files found in %s (pattern=%s)" % (inp_dir, ONLY_PATTERN))

    summary_path = os.path.join(out_dir, "summary.csv")
    with open_csv(summary_path) as fsum:
        wsum = csv.writer(fsum)
        wsum.writerow(["job", "inp", "status", "odb", "csv"])

        for inp in inps:
            job = os.path.splitext(os.path.basename(inp))[0]
            print("\n=== CASE: %s ===" % job)

            # Run in inp_dir (so Abaqus writes files there)
            cwd0 = os.getcwd()
            os.chdir(inp_dir)
            try:
                # 1) Run solver
                try:
                    run(["abaqus", "job=%s" % job, "input=%s" % os.path.basename(inp),
                         "cpus=%d" % CPUS, "interactive"])
                except Exception as e:
                    print("!! Launch failed:", e)
                    wsum.writerow([job, inp, "LAUNCH_FAIL", "", ""])
                    continue

                # 2) Wait completion + check ODB
                ok = wait_for_job(job)
                odb_path = os.path.join(inp_dir, job + ".odb")
                if (not ok) or (not os.path.exists(odb_path)):
                    print("!! No ODB for", job)
                    wsum.writerow([job, inp, "NO_ODB", "", ""])
                    continue

                # 3) Postprocess
                out_csv_path = os.path.join(out_dir, job + "_RF3_U3.csv")
                try:
                    postprocess_rf3_u3(odb_path, out_csv_path)
                    print("OK post ->", out_csv_path)
                    wsum.writerow([job, inp, "OK", odb_path, out_csv_path])
                except Exception as e:
                    print("!! Postprocess failed:", e)
                    wsum.writerow([job, inp, "POST_FAIL", odb_path, ""])
                    continue

            finally:
                os.chdir(cwd0)

    print("\nDone. Summary:", summary_path)

if __name__ == "__main__":
    main()