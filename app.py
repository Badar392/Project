"""Sugar Mill Plot Mapper - draw an area on the satellite map, detect every plot inside it."""
import colorsys
import json
import zlib

import folium
import pandas as pd
import streamlit as st
from folium.plugins import Draw
from streamlit_folium import st_folium

import core

st.set_page_config(page_title="Sugar Mill Plot Mapper", page_icon="🌾", layout="wide")

SAT_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
LABELS_URL = ("https://server.arcgisonline.com/ArcGIS/rest/services/Reference/"
              "World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}")
START_CENTER, START_ZOOM = [31.40, 72.70], 8          # Punjab sugar belt
MAX_LABELS = 300


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def base_map(center, zoom):
    m = folium.Map(location=center, zoom_start=zoom, tiles=None, control_scale=True, max_zoom=20)
    folium.TileLayer(SAT_URL, attr="Imagery © Esri, Maxar, Earthstar Geographics",
                     name="Satellite", max_native_zoom=18, max_zoom=20).add_to(m)
    # village / town / district names drawn on top of the satellite imagery
    folium.TileLayer(LABELS_URL, attr="Labels © Esri", name="Place names", overlay=True,
                     max_native_zoom=17, max_zoom=20, opacity=1.0).add_to(m)
    return m


def plot_colour(pid: str) -> str:
    h = (zlib.crc32(pid.encode()) % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.75, 1.0)
    return "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))


def ring_from_drawing(drawings):
    """Last drawn polygon/rectangle -> list of (lon, lat)."""
    for d in reversed(drawings or []):
        geom = d.get("geometry", {})
        if geom.get("type") == "Polygon":
            return [tuple(p[:2]) for p in geom["coordinates"][0]]
    return None


def result_map(res):
    ring, feats = res["ring"], res["features"]
    lats = [p[1] for p in ring]
    lons = [p[0] for p in ring]
    m = base_map([sum(lats) / len(lats), sum(lons) / len(lons)], 16)
    folium.PolyLine([(lat, lon) for lon, lat in ring], color="#ffffff", weight=2, dash_array="8 6").add_to(m)
    if res.get("location"):
        folium.Marker(
            [max(lats), sum(lons) / len(lons)],
            icon=folium.DivIcon(
                icon_size=(0, 0), icon_anchor=(0, 28),
                html=f'<div style="transform:translateX(-50%);white-space:nowrap;background:#111c;color:#fff;'
                     f'font:600 13px sans-serif;padding:4px 10px;border-radius:12px;border:1px solid #fff8">'
                     f'📍 {res["location"]}</div>')).add_to(m)
    if feats:
        folium.GeoJson(
            {"type": "FeatureCollection", "features": feats},
            style_function=lambda f: {
                "color": plot_colour(f["properties"]["plot_id"]), "weight": 2.5,
                "fillColor": plot_colour(f["properties"]["plot_id"]), "fillOpacity": 0.28},
            highlight_function=lambda f: {"weight": 5, "fillOpacity": 0.55},
            tooltip=folium.GeoJsonTooltip(
                fields=["plot_id", "area_acre", "area_kanal", "land_cover"],
                aliases=["Plot ID", "Acres", "Kanal", "Land cover"]),
            name="Plots",
        ).add_to(m)
        if len(feats) <= MAX_LABELS:
            for f in feats:
                p = f["properties"]
                label = p["plot_id"].rsplit("-", 1)[-1]
                folium.Marker(
                    [p["centroid_lat"], p["centroid_lon"]],
                    icon=folium.DivIcon(
                        icon_size=(40, 16), icon_anchor=(20, 8),
                        html=f'<div style="font:700 11px sans-serif;color:#fff;text-align:center;'
                             f'text-shadow:0 0 3px #000,0 0 3px #000">{label}</div>')).add_to(m)
    m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
    return m


def features_table(feats):
    return pd.DataFrame([f["properties"] for f in feats])


# --------------------------------------------------------------------------- #
# sidebar
# --------------------------------------------------------------------------- #
st.sidebar.title("🌾 Plot Mapper")
mill = st.sidebar.text_input("Sugar mill / zone", value="")
auditor = st.sidebar.text_input("Auditor name", value="")
prefix = st.sidebar.text_input("Plot ID prefix", value="SM", max_chars=6).strip().upper() or "SM"

with st.sidebar.expander("Detection settings"):
    min_acre = st.slider("Smallest plot (acres)", 0.1, 5.0, 0.5, 0.1)
    max_acre = st.slider("Largest plot (acres)", 5.0, 300.0, 100.0, 5.0)
    strictness = st.slider(
        "Boundary strictness", 6, 30, 12,
        help="Lower = more, smaller plots (use if neighbouring plots get merged). "
             "Higher = fewer, larger plots (use if one plot gets split).")
    include_bare = st.checkbox("Include bare / fallow fields", value=True,
                               help="Untick to detect only fields with green crop.")

# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
tab_map, tab_saved = st.tabs(["🗺️ Map & detect", "💾 Saved audits"])

with tab_map:
    res = st.session_state.get("result")

    # ------------------------------ step 1: draw ------------------------------
    if res is None:
        st.markdown("**Step 1.** Search for a village or area (optional), zoom in, then draw a polygon "
                    "(or rectangle) around the plots.  **Step 2.** Press *Detect plots*.")

        # ---- search bar ----
        with st.form("search_form", clear_on_submit=False):
            sc1, sc2 = st.columns([5, 1])
            query = sc1.text_input("Search a village, tehsil, city or mill", placeholder="e.g. Bhalwal, Shorkot, Chak 45 SB",
                                   label_visibility="collapsed")
            do_search = sc2.form_submit_button("🔎 Search", use_container_width=True)
        if do_search and query.strip():
            with st.spinner("Searching..."):
                hits = core.search_places(query)
            st.session_state["hits"] = hits
            st.session_state["map_ver"] = st.session_state.get("map_ver", 0) + 1
            st.session_state["hit_idx"] = 0
            if not hits:
                st.warning("No place found. Try a nearby town or a different spelling.")
        hits = st.session_state.get("hits", [])
        hit = None
        if hits:
            if len(hits) > 1:
                idx = st.selectbox("Matches", range(len(hits)), key="hit_idx",
                                   format_func=lambda i: hits[i]["name"][:110])
            else:
                idx = 0
            hit = hits[idx]
            view_key = f"{st.session_state.get('map_ver', 0)}_{idx}"
        else:
            view_key = "start"

        if hit:
            m = base_map([hit["lat"], hit["lon"]], 15)
            if hit["bbox"]:
                s_, n_, w_, e_ = hit["bbox"]
                m.fit_bounds([[s_, w_], [n_, e_]], max_zoom=16)
        else:
            m = base_map(START_CENTER, START_ZOOM)
        Draw(
            export=False,
            draw_options={"polygon": {"allowIntersection": False,
                                      "shapeOptions": {"color": "#ffd400", "weight": 3, "fillOpacity": 0.05}},
                          "rectangle": {"shapeOptions": {"color": "#ffd400", "weight": 3, "fillOpacity": 0.05}},
                          "polyline": False, "circle": False, "marker": False, "circlemarker": False},
            edit_options={"edit": False},
        ).add_to(m)
        out = st_folium(m, height=620, use_container_width=True,
                        returned_objects=["all_drawings"], key=f"draw_map_{view_key}")
        ring = ring_from_drawing(out.get("all_drawings") if out else None)

        if ring:
            st.success(f"Area selected: about {core.ring_area_m2(ring) / core.M2_PER_ACRE:,.0f} acres "
                       f"({core.ring_area_m2(ring) / 1e6:.2f} km²).")
        go = st.button("🔍 Detect plots", type="primary", disabled=ring is None)
        if ring is None:
            st.caption("Draw an area first - the button unlocks once a polygon exists.")

        if go and ring:
            params = core.Params(min_area_acre=min_acre, max_area_acre=max_acre,
                                 strictness=float(strictness), include_bare=include_bare)
            bar = st.progress(0.0, text="Starting...")
            try:
                imagery = core.load_imagery(ring, progress=lambda f, t: bar.progress(f, text=t))
                feats, info = core.detect_plots(imagery, ring, params,
                                                progress=lambda f, t: bar.progress(f, text=t))
            except core.AreaTooLarge as e:
                bar.empty()
                st.error(str(e))
                st.stop()
            except Exception as e:                       # network / imagery problems
                bar.empty()
                st.error(f"Detection failed: {e}")
                st.stop()
            bar.progress(0.97, text="Looking up the area name...")
            lat_c = sum(p[1] for p in ring) / len(ring)
            lon_c = sum(p[0] for p in ring) / len(ring)
            place = core.place_name(lat_c, lon_c)
            bar.empty()
            run_code = core.new_run_code()
            st.session_state["result"] = {
                "audit_id": run_code, "ring": ring, "info": info,
                "location": place.get("label", ""), "place": place,
                "features": core.assign_ids(feats, prefix, run_code)}
            st.rerun()

    # ------------------------------ step 2: results ----------------------------
    else:
        feats, ring, info = res["features"], res["ring"], res.get("info", {})
        if res.get("location"):
            st.markdown(f"### 📍 {res['location']}")
        top = st.columns([1, 1, 1, 1, 1.4])
        top[0].metric("Plots", len(feats))
        top[1].metric("Total acres", f"{sum(f['properties']['area_acre'] for f in feats):,.1f}")
        top[2].metric("Total hectares", f"{sum(f['properties']['area_ha'] for f in feats):,.1f}")
        top[3].metric("Audit code", res["audit_id"])
        if top[4].button("⬅ Draw a new area", use_container_width=True):
            st.session_state.pop("result")
            st.rerun()

        if not feats:
            st.warning("No plots were detected inside this area. Try a smaller *Boundary strictness*, "
                       "tick *Include bare / fallow fields*, lower *Smallest plot*, or draw over farmland.")

        st_folium(result_map(res), height=600, use_container_width=True,
                  returned_objects=[], key=f"result_map_{res['audit_id']}_{len(feats)}")

        if info:
            with st.expander("Detection details"):
                st.write(
                    f"Imagery zoom {info.get('zoom')} (~{info.get('mpp')} m per pixel). "
                    f"Regions examined: {info.get('regions')}. Rejected - not farmland: "
                    f"{info.get('not_agricultural')}, too small: {info.get('too_small')}, "
                    f"too large: {info.get('too_large')}, odd shape/narrow: {info.get('bad_shape')}.")
                st.caption("Outlines are derived from satellite imagery, not from land-record (khasra) "
                           "boundaries - verify important plots in the field.")

        if feats:
            st.subheader("Detected plots")
            ids = [f["properties"]["plot_id"] for f in feats]
            drop = st.multiselect("Remove wrongly detected plots (optional)", ids)
            if drop and st.button("Remove selected"):
                res["features"] = [f for f in feats if f["properties"]["plot_id"] not in drop]
                st.rerun()
            st.dataframe(features_table(feats), use_container_width=True, hide_index=True)

            st.subheader("Save")
            c1, c2 = st.columns(2)
            notes = c1.text_area("Notes (optional)", height=80)
            with c2:
                if st.button("💾 Save audit", type="primary", use_container_width=True):
                    core.save_audit(res["audit_id"], mill, auditor, notes, feats, ring, res.get("location", ""))
                    st.success(f"Saved as audit {res['audit_id']} (see the *Saved audits* tab).")
                meta = {"audit_id": res["audit_id"], "mill": mill, "auditor": auditor,
                        "location": res.get("location", "")}
                st.download_button(
                    "⬇ GeoJSON", json.dumps(core.to_geojson(feats, ring, meta)),
                    file_name=f"{prefix}_{res['audit_id']}_plots.geojson",
                    mime="application/geo+json", use_container_width=True)
                st.download_button(
                    "⬇ CSV", pd.DataFrame(core.to_rows(feats)).to_csv(index=False),
                    file_name=f"{prefix}_{res['audit_id']}_plots.csv",
                    mime="text/csv", use_container_width=True)

with tab_saved:
    audits = core.list_audits()
    if not audits:
        st.info("No saved audits yet.")
    else:
        st.dataframe(pd.DataFrame(audits), use_container_width=True, hide_index=True)
        pick = st.selectbox("Open an audit", [a["audit_id"] for a in audits])
        b1, b2 = st.columns(2)
        if b1.button("Open on map", use_container_width=True):
            data = core.load_audit(pick)
            if data:
                st.session_state["result"] = {"audit_id": pick, "ring": data["ring"],
                                              "features": data["features"], "info": {},
                                              "location": data.get("location", "")}
                st.success("Loaded - switch to the *Map & detect* tab.")
                st.rerun()
        if b2.button("Delete", use_container_width=True):
            core.delete_audit(pick)
            st.rerun()
    st.caption("Saved audits live on the app server's disk. On Streamlit Community Cloud this is wiped "
               "when the app restarts, so also download the GeoJSON/CSV for permanent records.")
