"""Prototype: 12 monthly land-cover maps from Sentinel-1 for any location.

    streamlit run app/streamlit_app.py
"""
import calendar
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src import config as C, gee_data as G, plots, prototype as P  # noqa: E402

st.set_page_config(page_title="SAR Land Cover Monitor", page_icon="🛰️", layout="wide")

PRESETS = {
    "Munich, Germany": (48.137, 11.575),
    "Cairo, Egypt": (30.044, 31.236),
    "Bengaluru, India": (12.972, 77.594),
    "Nairobi, Kenya": (-1.286, 36.817),
    "Lima, Peru": (-12.046, -77.043),
    "Custom coordinates": None,
}
SIZES = {"2.56 km": 256, "5.12 km": 512, "10.24 km": 1024}


@st.cache_resource(show_spinner="Loading model and connecting to Earth Engine ...")
def setup():
    G.init()
    return P.load_model(), P.load_stats()


@st.cache_data(show_spinner=False)
def analyse(lat, lon, start, size_px, smooth):
    model, stats = setup()
    return P.run(lat, lon, start, size_px=size_px, model=model, stats=stats, smooth=smooth)


def last_complete_month():
    first_of_month = dt.date.today().replace(day=1)
    previous = first_of_month - dt.timedelta(days=1)
    return previous.year, previous.month


# --- Sidebar: inputs ------------------------------------------------------------------------

st.sidebar.header("Location and period")
preset = st.sidebar.selectbox("Location", list(PRESETS))
default = PRESETS[preset] or (48.137, 11.575)
lat = st.sidebar.number_input("Latitude", -60.0, 75.0, default[0], step=0.001, format="%.4f", key=f"lat_{preset}")
lon = st.sidebar.number_input("Longitude", -180.0, 180.0, default[1], step=0.001, format="%.4f", key=f"lon_{preset}")

end_year, end_month = last_complete_month()
years = list(range(2017, end_year + 1))
c1, c2 = st.sidebar.columns(2)
year = c1.selectbox("Start year", years, index=years.index(2025) if 2025 in years else len(years) - 1)
month = c2.selectbox("Start month", range(1, 13), format_func=lambda m: calendar.month_abbr[m])
start = f"{year}-{month:02d}"
months = P.month_list(start)

size_label = st.sidebar.radio("Area (square, centred on the point)", list(SIZES), index=1, horizontal=True)
smooth = st.sidebar.checkbox("Temporal smoothing (3-month majority)", value=False,
                             help="Removes one-month class flicker caused by classification noise. "
                                  "Compare with and without to see how noise inflates change percentages.")

last = f"{end_year}-{end_month:02d}"
valid_period = months[-1] <= last
if not valid_period:
    st.sidebar.error(f"The 12 months must end by {plots.month_label(last)} (the last complete month).")
st.sidebar.caption(f"Period: **{plots.month_label(months[0])} – {plots.month_label(months[-1])}**")
run = st.sidebar.button("Run", type="primary", disabled=not valid_period, width="stretch")

st.sidebar.divider()
st.sidebar.caption("Input: Sentinel-1 VV + VH only (monthly median, Lee filter). "
                   "Model: trained on 2023–24, 8 classes. Reference: Esri 10 m Annual LULC.")

# --- Main ---------------------------------------------------------------------------------

st.title("🛰️ SAR Land Cover Monitor")
st.caption("Monthly land use / land cover from Sentinel-1 radar alone: works through clouds and at night.")

if run:
    st.session_state["params"] = (round(lat, 4), round(lon, 4), start, SIZES[size_label], smooth)

if "params" not in st.session_state:
    st.info("Choose a location and a start month in the sidebar, then press **Run**. "
            "A run takes about 30–90 seconds: 12 Sentinel-1 composites are built in Google Earth Engine "
            "and classified by the trained model.")
    st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}), zoom=9, height=380)
    st.stop()

params = st.session_state["params"]
if not P.MODEL_PATH.exists():
    st.error(f"Model weights not found: `{P.MODEL_PATH}`. Train the model (notebook 02) or copy "
             "`best_model.keras` into the `weights/` folder.")
    st.stop()

with st.spinner("Building 12 monthly Sentinel-1 composites in Earth Engine and classifying them ..."):
    try:
        r = analyse(*params)
    except Exception as e:
        st.error(f"Run failed: {e}")
        st.stop()

lat, lon, start, size_px, smooth = params
n_data = sum(r["maps"][m] is not None for m in r["months"])
st.subheader(f"{lat:.4f}, {lon:.4f} · {plots.month_label(r['months'][0])} – {plots.month_label(r['months'][-1])}")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Area (square)", f"{r['size_km']:.2f} km")
m2.metric("Months with data", f"{n_data} / 12")
m3.metric("Orbit pass", r["orbit"].title())
m4.metric("Sentinel-1 scenes used", sum(r["scenes"].values()))
if n_data < 12:
    st.warning("Some months have no Sentinel-1 scene in this orbit pass; they are shown as empty and "
               "excluded from the change calculation.")

tab_maps, tab_pct, tab_change, tab_ref, tab_input = st.tabs(
    ["🗺️ Monthly maps", "📊 Class percentages", "📈 Changes", "✅ Independent check", "📡 Sentinel-1 input"])

with tab_maps:
    title = "Land cover per month" + (" (3-month majority smoothing)" if r["smooth"] else "")
    st.pyplot(plots.maps_grid(r["maps"], r["months"], title=title))

with tab_pct:
    st.markdown("Share of the mapped area per class (pixels without radar data excluded).")
    st.pyplot(plots.pct_chart(r["pct"]))
    table = r["pct"].rename(index=plots.month_label)
    st.dataframe(table.style.format("{:.1f}"), width="stretch")
    st.download_button("Download percentages (CSV)", r["pct"].to_csv().encode(), "class_percentages.csv")

with tab_change:
    st.markdown("All changes in **percentage points** (pp): the difference between two class shares, "
                "e.g. built-up going from 40 % to 42 % is +2.0 pp.")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("**Month-to-month change**")
        mtm = r["month_to_month"].iloc[1:].rename(index=plots.month_label)
        st.dataframe(mtm.style.format("{:+.1f}").background_gradient(cmap="RdBu", vmin=-10, vmax=10),
                     width="stretch")
    with c2:
        st.markdown(f"**Total change ({r['total'].name})**")
        st.pyplot(plots.total_change_chart(r["total"]))
    st.info("Radar responds to soil moisture, crop growth and flooding, so part of the month-to-month "
            "change is seasonal or classification noise rather than real land-use change. Turn on "
            "temporal smoothing to see how much of it disappears.")
    changes = pd.concat({"month_to_month_pp": r["month_to_month"], "total_pp": r["total"].to_frame().T}).round(3)
    st.download_button("Download changes (CSV)", changes.to_csv().encode(), "class_changes.csv")

with tab_ref:
    if "ref_map" not in r:
        st.warning("No Esri reference map available for this location.")
    else:
        st.markdown(f"The **Esri 10 m Annual LULC** map ({r['ref_year']}) is produced independently from "
                    "Sentinel-2 with a different model and was **not used for training**. It is compared with "
                    "the most frequent class per pixel over the 12 months.")
        st.pyplot(plots.reference_pair(P.annual_majority(r["maps"], r["months"]), r["ref_map"],
                                       r["ref_year"], r["months"]))
        st.metric("Pixel agreement with Esri", f"{100 * r['ref_agreement']:.1f} %")
        st.dataframe(r["ref_table"].style.format("{:+.1f}", subset=["difference_pp"])
                     .format("{:.1f}", subset=["model_%", "reference_%"]), width="stretch")

with tab_input:
    st.markdown(f"Monthly median composites, {r['orbit'].lower()} pass "
                f"(scenes per month: ascending {r['orbit_counts']['ASCENDING']}, "
                f"descending {r['orbit_counts']['DESCENDING']} over the 12 months).")
    band = st.radio("Band", C.S1_BANDS, horizontal=True)
    st.pyplot(plots.s1_grid(r["s1"], r["months"], band=C.S1_BANDS.index(band)))
    st.dataframe(pd.DataFrame({"Sentinel-1 scenes": r["scenes"]}).rename(index=plots.month_label).T,
                 width="stretch")
