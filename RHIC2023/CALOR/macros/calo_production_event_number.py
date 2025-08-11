import sys
import re
import pyodbc
import pandas as pd
import matplotlib.pyplot as plt

#####################################################################################################################
#   Usage: python3 calo_production_event_number.py <triggered_tag> <calofitting_tag> [start_run] [end_run]
#
#   triggered_tag     : dataset tag containing DST_TRIGGERED_EVENT_seb00..seb17 (or legacy DST_TRIGGERED_EVENT)
#   calofitting_tag   : dataset tag containing DST_CALOFITTING (or legacy DST_CALOFITTING_run2pp)
#   start/end runs    : optional; if omitted, all runs in the provided datasets are used to determine the range
#   Notes             : for calorimeter datasets only; RunDB requires all 18 SEBs
#####################################################################################################################

def get_dst_query(tag, like_pattern, start_run, end_run):
    query = (
        f"SELECT runnumber, dsttype, SUM(events) "
        f"FROM datasets WHERE tag = '{tag}' AND dsttype LIKE '{like_pattern}' "
    )
    if str(start_run).strip() and str(end_run).strip():
        query += f"AND runnumber >= {start_run} AND runnumber <= {end_run} "
    query += "GROUP BY runnumber, dsttype ORDER BY runnumber, dsttype;"
    return query

def get_rundb_query(start_run, end_run):
    return f"""
    SELECT run.runnumber, run.eventsinrun
    FROM run
    JOIN event_numbers ON run.runnumber = event_numbers.runnumber
    WHERE run.runtype = 'physics'
      AND run.runnumber >= {start_run} AND run.runnumber <= {end_run}
      AND event_numbers.hostname IN ('seb00','seb01','seb02','seb03','seb04','seb05','seb06','seb07',
                                     'seb08','seb09','seb10','seb11','seb12','seb13','seb14','seb15','seb16','seb17')
    GROUP BY run.runnumber, run.eventsinrun
    HAVING count(DISTINCT event_numbers.hostname) = 18
    ORDER BY run.runnumber;
    """

def get_data(cursor, query):
    cursor.execute(query)
    return [row for row in cursor.fetchall()]

def pivot_and_standardize(rows, value_name='sum_events'):
    """rows -> pivot(runnumber x dsttype), normalize column names."""
    if not rows:
        return pd.DataFrame(columns=['runnumber'])
    df = pd.DataFrame([tuple(r) for r in rows], columns=['runnumber', 'dsttype', value_name])
    piv = df.pivot_table(index='runnumber', columns='dsttype', values=value_name, aggfunc='sum').reset_index()

    # Normalize known names (keep legacy compatibility)
    rename_map = {
        'DST_CALOFITTING_run2pp': 'dst_calofitting',
        'DST_CALOFITTING': 'dst_calofitting',
        'DST_CALO_run2pp': 'dst_calo',
        'DST_CALO': 'dst_calo',
        'DST_TRIGGERED_EVENT_run2pp': 'dst_triggered_event',
        'DST_TRIGGERED_EVENT': 'dst_triggered_event',
    }
    piv = piv.rename(columns=rename_map)

    # Normalize per-SEB triggered columns
    seb_cols = []
    for col in list(piv.columns):
        m = re.fullmatch(r'DST_TRIGGERED_EVENT_seb(\d{2})', col)
        if m:
            new_col = f'dst_triggered_event_seb{m.group(1)}'
            piv = piv.rename(columns={col: new_col})
            seb_cols.append(new_col)

    seb_cols.sort()
    return piv, seb_cols

def main():
    # --- args ---
    if len(sys.argv) < 3:
        print("Usage: python3 calo_production_event_number.py <triggered_tag> <calofitting_tag> [start_run] [end_run]")
        sys.exit(1)
    triggered_tag   = sys.argv[1]
    calofitting_tag = sys.argv[2]
    start_run = sys.argv[3] if len(sys.argv) > 3 else ""
    end_run   = sys.argv[4] if len(sys.argv) > 4 else ""
    print("start run and end run", start_run, end_run)

    # --- File Catalog connections ---
    fc_conn = pyodbc.connect("DSN=FileCatalog;UID=phnxrc;READONLY=True")
    cur = fc_conn.cursor()

    # Pull triggered per-SEB (or legacy)
    q_trig = get_dst_query(triggered_tag, "DST_TRIGGERED_EVENT%", start_run, end_run)
    rows_trig = get_data(cur, q_trig)
    if rows_trig:
        print("Found triggered entries in the File Catalog")
    trig_pivot, seb_cols = pivot_and_standardize(rows_trig)

    # Pull calofitting (and legacy)
    q_fit = get_dst_query(calofitting_tag, "DST_CALOFITTING%", start_run, end_run)
    rows_fit = get_data(cur, q_fit)
    if rows_fit:
        print("Found calofitting entries in the File Catalog")
    fit_pivot, _ = pivot_and_standardize(rows_fit)

    fc_conn.close()

    # Initialize run range if not provided (use union of seen runs from both tags)
    run_mins = []
    run_maxs = []
    if 'runnumber' in trig_pivot and len(trig_pivot):
        run_mins.append(trig_pivot['runnumber'].min())
        run_maxs.append(trig_pivot['runnumber'].max())
    if isinstance(fit_pivot, tuple):
        fit_pivot = fit_pivot[0]  # safety if pivot_and_standardize returned a tuple (it won't here)
    if 'runnumber' in fit_pivot and len(fit_pivot):
        run_mins.append(fit_pivot['runnumber'].min())
        run_maxs.append(fit_pivot['runnumber'].max())

    if not (str(start_run).strip() and str(end_run).strip()):
        if run_mins and run_maxs:
            start_run = int(min(run_mins))
            end_run   = int(max(run_maxs))
        else:
            print("No entries found in either dataset tag, exiting...")
            sys.exit(1)

    print("start run and end run", start_run, end_run)

    # --- RunDB ---
    daq_conn = pyodbc.connect("DSN=daq;READONLY=True")
    daq_cur = daq_conn.cursor()
    rows_rundb = get_data(daq_cur, get_rundb_query(start_run, end_run))
    if rows_rundb:
        print("Found entries in the RunDB")
    daq_conn.close()

    df_rundb = pd.DataFrame([tuple(r) for r in rows_rundb], columns=['runnumber', 'rundb'])

    # --- Merge everything by runnumber ---
    # Ensure pivots are DataFrames (pivot_and_standardize returns (piv, seb_cols))
    if isinstance(trig_pivot, tuple):
        trig_pivot = trig_pivot[0]
    df = pd.merge(df_rundb, trig_pivot, on='runnumber', how='outer')
    df = pd.merge(df, fit_pivot, on='runnumber', how='outer')

    # Build total triggered column for convenience
    if seb_cols:
        df['dst_triggered_event_total'] = df[seb_cols].sum(axis=1, skipna=True)
    elif 'dst_triggered_event' in df.columns:
        df['dst_triggered_event_total'] = df['dst_triggered_event']
    else:
        df['dst_triggered_event_total'] = 0.0

    has_fit = 'dst_calofitting' in df.columns
    has_any_triggered = ('dst_triggered_event_total' in df.columns)

    # --- Filters ---
    fdf = df[(df['rundb'] > 5) & (df['rundb'].notna()) &
             (df['dst_triggered_event_total'].fillna(0) == 0)]
    produced = df[(df['rundb'] > 10_000) & (df['rundb'].notna()) &
                  (df['dst_triggered_event_total'].fillna(0) > 0)]
    allrundf = df[(df['rundb'] > 10_000) & (df['rundb'].notna())]

    # --- Basic counts ---
    NumNotProduced = len(fdf)
    EventsNotProduced = "{:.3e}".format(fdf['rundb'].sum())
    print()
    print(f"Number of calo physics runs in RunDB not passed to production: {NumNotProduced}")
    print(EventsNotProduced, "events not passed to production")
    print()

    # --- Requested ratios over sum(rundb) (using allrundf: rundb > 10k) ---
    total_rundb = allrundf['rundb'].sum()
    if total_rundb and not pd.isna(total_rundb) and total_rundb > 0:
        print("Ratios over sum(RunDB) for runs with rundb > 10k:")

        if has_fit:
            fit_ratio = allrundf['dst_calofitting'].sum() / total_rundb
            print(f"  DST_CALOFITTING / RunDB: {fit_ratio:.3f}")
        else:
            print("  DST_CALOFITTING not found in the calofitting tag.")

        if seb_cols:
            print("  DST_TRIGGERED_EVENT per SEB / RunDB:")
            for col in seb_cols:
                ratio = (allrundf[col].sum() if col in allrundf.columns else 0.0) / total_rundb
                print(f"    {col.replace('dst_triggered_event_', '').upper()}: {ratio:.3f}")
        elif 'dst_triggered_event' in df.columns:
            trig_ratio = allrundf['dst_triggered_event'].sum() / total_rundb
            print(f"  DST_TRIGGERED_EVENT / RunDB: {trig_ratio:.3f}")
        else:
            print("  No DST_TRIGGERED_EVENT columns found.")
    else:
        print("Total RunDB sum is zero or undefined; ratios not computed.")

    # --- Legacy-style summary using *_total (optional but handy) ---
    if has_any_triggered and len(produced) > 0:
        print("Fraction of triggered_total / RunDB (produced runs): "
              f"{produced['dst_triggered_event_total'].sum() / produced['rundb'].sum():.3f}")
        print("Fraction of triggered_total / RunDB (including not produced): "
              f"{allrundf['dst_triggered_event_total'].sum() / allrundf['rundb'].sum():.3f}")
        if has_fit and produced['dst_triggered_event_total'].sum() > 0:
            print("Fraction of calofitting / triggered_total (produced runs): "
                  f"{produced['dst_calofitting'].sum() / produced['dst_triggered_event_total'].sum():.3f}")

    # --- Plots (rundb, triggered_total, calofitting if present) ---
    value_cols = ['rundb']
    if has_any_triggered: value_cols.append('dst_triggered_event_total')
    if has_fit: value_cols.append('dst_calofitting')

    column_sums    = produced[value_cols].sum()
    allcolumn_sums = allrundf[value_cols].sum()
    long_runs      = produced[produced['rundb'] > 10_000_000]

    # Plot 1: totals (produced runs)
    plt.figure(figsize=(8, 6))
    ax = column_sums.plot(kind='bar')
    for i, v in enumerate(column_sums):
        ax.text(i, v + 0.01 * max(1, v), f'{int(v)}', ha='center', va='bottom')
    plt.title(f'{triggered_tag} / {calofitting_tag} Events for Runs {start_run}-{end_run} (produced)')
    plt.ylabel('Events'); plt.xticks(rotation=0, ha='center')

    # Plot 2: totals (including not-produced)
    plt.figure(figsize=(8, 6))
    ax = allcolumn_sums.plot(kind='bar')
    for i, v in enumerate(allcolumn_sums):
        ax.text(i, v + 0.01 * max(1, v), f'{int(v)}', ha='center', va='bottom')
    plt.title(f'All Calo Physics Runs {start_run}-{end_run}')
    plt.ylabel('Events'); plt.xticks(rotation=0, ha='center')

    # Plot 3: RunDB event number for no-production runs
    plt.figure(figsize=(10, 6))
    plt.hist(fdf['rundb'].dropna(), bins=30, edgecolor='black')
    plt.title(f'RunDB events for runs with no production ({start_run}-{end_run})')
    plt.xlabel('Event number'); plt.ylabel('Frequency'); plt.grid(True)

    # Plot 4: production event number for long runs
    plt.figure(figsize=(10, 6))
    plt.hist(long_runs['dst_triggered_event_total'].dropna(), bins=30, edgecolor='black')
    plt.title(f'Production event number (>10M) ({start_run}-{end_run})')
    plt.xlabel('Prod. event number (triggered total)'); plt.ylabel('Frequency'); plt.grid(True)

    plt.show()

if __name__ == "__main__":
    main()
