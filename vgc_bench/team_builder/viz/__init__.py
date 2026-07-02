"""
Interactive visualization for the VGC-Bench team builder.

A read-only consumer of the files a team-builder run writes under ``results/<run>/``.
It reconstructs the *evolution* of the best team over a run (a timeline of snapshots)
alongside evaluation stats (win-rate curves, the PSRO payoff/matchup heatmap, Nash
weights, and species churn) and renders them as a Streamlit dashboard.

Run it with::

    streamlit run vgc_bench/team_builder/viz/app.py -- results/<run>

Nothing here modifies training/optimizer state.
"""
