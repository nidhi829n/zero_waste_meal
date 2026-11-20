# ZeroWaste Meal - Streamlit app (donor, NGO and admin dashboards)
import os
import datetime
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

NGO_FILE = "Rajasthan_Combined_NGOs_Colab_Dataset.csv"
DONATIONS_FILE = "donations.csv"
OVERRIDES_FILE = "capacity_overrides.csv"
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.set_page_config(page_title="ZeroWaste Meal", page_icon="🍲", layout="centered")


# ---------------------------------------------------------------- data
@st.cache_data
def load_ngos():
    df = pd.read_csv(NGO_FILE, dtype=str)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    for c in df.columns:
        df[c] = df[c].str.strip()
    for c in ["latitude", "longitude", "max_capacity", "current_occupancy",
              "available_capacity", "accepting_donations"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["first_phone"] = df["phone"].str.split(",").str[0].str.strip()
    df["first_email"] = df["email"].str.split(",").str[0].str.strip()
    return df


def get_pool():
    """Usable NGOs, with any capacity updates made by NGOs applied."""
    df = load_ngos().copy()
    if os.path.exists(OVERRIDES_FILE):
        ov = pd.read_csv(OVERRIDES_FILE).drop_duplicates("ngo_id", keep="last")
        m = dict(zip(ov["ngo_id"], ov["available_capacity"]))
        df["available_capacity"] = df.apply(
            lambda r: m.get(r["ngo_id"], r["available_capacity"]), axis=1)
    pool = df[(df["accepting_donations"] == 1)
              & df["latitude"].notna() & df["longitude"].notna()
              & (df["available_capacity"] > 0)
              & ~df["district"].str.startswith("All Districts")].copy()
    return pool


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371 * np.arcsin(np.sqrt(a))


def match_ngos(pool, district, servings, pincode=None, k=3, w_dist=0.6, w_cap=0.4):
    district_loc = pool.groupby("district")[["latitude", "longitude"]].mean()
    pincode_loc = pool.groupby("pincode")[["latitude", "longitude"]].mean()
    cand = pool[pool["district"] == district].copy()
    note = ""
    if cand.empty:
        cand = pool.copy()
        note = f"No usable NGO in {district}. Showing nearest NGOs in Rajasthan."
        loc = district_loc.mean()
    elif pincode and pincode in pincode_loc.index:
        loc = pincode_loc.loc[pincode]
    else:
        loc = district_loc.loc[district]
    cand["distance_km"] = haversine(loc["latitude"], loc["longitude"],
                                    cand["latitude"], cand["longitude"]).round(1)
    cand["capacity_gap"] = (cand["available_capacity"] - servings).abs()
    if len(cand) == 1:
        best = cand.copy()
        best["match_score"] = 100.0
    else:
        X = MinMaxScaler().fit_transform(cand[["distance_km", "capacity_gap"]]) * [w_dist, w_cap]
        nn = NearestNeighbors(n_neighbors=min(k, len(cand))).fit(X)
        d, idx = nn.kneighbors([[0, 0]])
        best = cand.iloc[idx[0]].copy()
        best["match_score"] = (100 * (1 - d[0] / np.hypot(w_dist, w_cap))).clip(0, 100).round(1)
    best.insert(0, "rank", range(1, len(best) + 1))
    return best, note


def split_plan(best, servings):
    left, plan = servings, []
    for _, r in best.iterrows():
        if left <= 0:
            break
        give = int(min(left, r["available_capacity"]))
        plan.append((r["ngo_name"], give))
        left -= give
    return plan, left


def read_donations():
    if os.path.exists(DONATIONS_FILE):
        return pd.read_csv(DONATIONS_FILE, dtype=str).fillna("")
    return pd.DataFrame()


def show_results(best, note, servings):
    if note:
        st.warning(note)
    for _, r in best.iterrows():
        with st.container(border=True):
            st.markdown(f"**#{int(r['rank'])}  {r['ngo_name']}**  ·  match score {r['match_score']}")
            st.caption(f"{r['category']}  ·  {r['distance_km']} km away  ·  "
                       f"free capacity {int(r['available_capacity'])}")
            st.write(f"📍 {r['address']}")
            st.write(f"📞 {r['first_phone']}")
            st.write(f"✉️ {r['first_email']}")
            st.write(f"🙋 Volunteer: {r['volunteer_contact_info']}")
    if servings > best["available_capacity"].max():
        plan, left = split_plan(best, servings)
        st.info("No single NGO can take all of it. Suggested split:\n\n"
                + "\n".join(f"- {n}: {s} servings" for n, s in plan)
                + (f"\n\nStill unallocated: {left} servings" if left > 0 else ""))


# ---------------------------------------------------------------- pages
def donor_page():
    st.header("Donate surplus food")
    pool = get_pool()
    districts = sorted(pool["district"].unique())

    with st.form("donor_form"):
        name = st.text_input("Your name")
        phone = st.text_input("Contact number (10 digits)")
        venue = st.text_area("Venue address", height=80)
        c1, c2 = st.columns(2)
        district = c1.selectbox("District", districts,
                                index=districts.index("Jaipur") if "Jaipur" in districts else 0)
        pincode = c2.text_input("Pincode (optional)")
        servings = st.number_input("Estimated servings", min_value=1, value=50, step=5)
        items = st.text_input("Food items (e.g. dal, rice, roti)")
        vnv = st.radio("Food type", ["Veg", "Non-veg"], horizontal=True)
        delivery = st.radio("Delivery", ["I can deliver", "I need volunteer support"], horizontal=True)
        photo = st.file_uploader("Photo of the food (for quality check)", type=["jpg", "jpeg", "png"])
        fresh = st.checkbox("I confirm the food was cooked recently and stored hygienically")
        submit = st.form_submit_button("Find NGOs")

    if submit:
        digits = "".join(ch for ch in phone if ch.isdigit())[-10:]
        errors = []
        if not name.strip(): errors.append("Enter your name.")
        if len(digits) != 10: errors.append("Enter a valid 10-digit contact number.")
        if not venue.strip(): errors.append("Enter the venue address.")
        if not items.strip(): errors.append("Enter the food items.")
        if photo is None: errors.append("Upload a photo of the food.")
        if not fresh: errors.append("Please confirm the food safety checkbox.")
        if errors:
            for e in errors:
                st.error(e)
            return

        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        photo_name = f"{stamp}_{digits}.{photo.name.split('.')[-1]}"
        with open(os.path.join(UPLOAD_DIR, photo_name), "wb") as f:
            f.write(photo.getbuffer())

        best, note = match_ngos(pool, district, int(servings), pincode.strip() or None)
        record = {
            "donation_id": "D" + stamp,
            "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "name": name.strip(), "phone": digits, "venue_address": venue.strip(),
            "district": district, "pincode": pincode.strip(), "servings": int(servings),
            "food_items": items.strip(), "veg_nonveg": vnv, "delivery_option": delivery,
            "photo": photo_name,
            "matched_ngo_ids": ", ".join(best["ngo_id"]),
            "matched_ngo_names": " | ".join(best["ngo_name"]),
            "match_scores": " | ".join(best["match_score"].astype(str)),
            "distances_km": " | ".join(best["distance_km"].astype(str)),
            "status": "Pending",
        }
        pd.DataFrame([record]).to_csv(DONATIONS_FILE, mode="a",
                                      header=not os.path.exists(DONATIONS_FILE), index=False)
        st.session_state["last_result"] = (best, note, int(servings))

    if "last_result" in st.session_state:
        best, note, servings = st.session_state["last_result"]
        st.success("Thank you! Your donation is saved. Contact these NGOs directly:")
        show_results(best, note, servings)


def ngo_page():
    st.header("NGO dashboard")
    ngos = load_ngos()
    ngos = ngos[~ngos["district"].str.startswith("All Districts")]
    st.caption("Demo login: your access code is your NGO ID (for example NGO-001).")

    if not st.session_state.get("ngo_id"):
        label = ngos["ngo_name"] + "  (" + ngos["ngo_id"] + ")"
        choice = st.selectbox("Select your organisation", label)
        code = st.text_input("Access code", type="password", key="ngo_login_code").strip()
        if st.button("Log in"):
            ngo_id = choice.split("(")[-1].strip(")")
            if code.lower() == ngo_id.lower():
                st.session_state["ngo_id"] = ngo_id
                st.rerun()
            else:
                st.error("Wrong access code.")
        return

    ngo_id = st.session_state["ngo_id"]
    row = get_pool_row(ngo_id)
    st.subheader(row["ngo_name"])
    st.caption(f"{row['category']}  ·  {row['district']}")
    st.write(f"📍 {row['address']}")
    st.write(f"📞 {row['first_phone']}   ✉️ {row['first_email']}")
    st.write(f"Registration: {row['registration_id']}  ({row['registration_date']})")

    st.markdown("#### Update free capacity")
    new_cap = st.number_input("Servings you can take today", min_value=0,
                              value=int(row["available_capacity"]), step=5)
    if st.button("Save capacity"):
        pd.DataFrame([{"ngo_id": ngo_id, "available_capacity": int(new_cap)}]).to_csv(
            OVERRIDES_FILE, mode="a", header=not os.path.exists(OVERRIDES_FILE), index=False)
        st.success("Capacity updated.")

    st.markdown("#### Incoming donations")
    d = read_donations()
    if d.empty:
        st.info("No donations yet.")
    else:
        mine = d[d["matched_ngo_ids"].apply(lambda s: ngo_id in [x.strip() for x in s.split(",")])]
        if mine.empty:
            st.info("No donations matched to you yet.")
        for _, r in mine.iloc[::-1].iterrows():
            with st.container(border=True):
                st.markdown(f"**{r['servings']} servings** · {r['food_items']} ({r['veg_nonveg']})")
                st.write(f"Donor: {r['name']}  ·  📞 {r['phone']}")
                st.write(f"Venue: {r['venue_address']}")
                st.write(f"Delivery: {r['delivery_option']}  ·  {r['time']}")
    if st.button("Log out"):
        st.session_state.pop("ngo_id")
        st.rerun()


def get_pool_row(ngo_id):
    df = load_ngos()
    row = df[df["ngo_id"] == ngo_id].iloc[0].copy()
    if os.path.exists(OVERRIDES_FILE):
        ov = pd.read_csv(OVERRIDES_FILE)
        ov = ov[ov["ngo_id"] == ngo_id]
        if not ov.empty:
            row["available_capacity"] = ov.iloc[-1]["available_capacity"]
    return row


STATUSES = ["Pending", "NGO contacted", "Completed", "Cancelled"]
LOG_COLUMNS = ["donation_id", "time", "name", "phone", "venue_address", "district", "pincode",
               "servings", "food_items", "veg_nonveg", "delivery_option", "photo",
               "matched_ngo_ids", "matched_ngo_names", "match_scores", "distances_km", "status"]


def normalise(d):
    """Make sure every expected column exists and has the right type."""
    d = d.copy()
    for c in LOG_COLUMNS:
        if c not in d.columns:
            d[c] = ""
    d["status"] = d["status"].replace("", "Pending")
    d["servings"] = pd.to_numeric(d["servings"], errors="coerce").fillna(0).astype(int)
    d["dt"] = pd.to_datetime(d["time"], errors="coerce")
    d["day"] = d["dt"].dt.normalize()
    d["best_score"] = pd.to_numeric(d["match_scores"].astype(str).str.split("|").str[0], errors="coerce")
    return d


@st.cache_data
def make_sample_data(n=60, seed=7):
    """Sample donations produced by the real matching model (for the initial report)."""
    rng = np.random.default_rng(seed)
    pool = get_pool()
    names = ["Rohit Mehta", "Anita Jain", "Karan Bhatia", "Sana Qureshi", "Manish Goyal",
             "Pooja Sharma", "Vikas Rathore", "Neha Agarwal", "Imran Khan", "Deepak Soni",
             "Kavita Singh", "Arun Meena"]
    want = {"Jaipur": .55, "Udaipur": .12, "Ajmer": .10, "Jodhpur": .08, "Kota": .08, "Bikaner": .07}
    dists = [x for x in want if x in set(pool["district"])]
    p = np.array([want[x] for x in dists]); p = p / p.sum()
    sizes = [20, 30, 40, 50, 60, 80, 100, 150, 200, 300]
    sp = np.array([4, 6, 8, 10, 10, 8, 6, 4, 2, 1], dtype=float); sp /= sp.sum()
    now = datetime.datetime.now()
    rows = []
    for i in range(n):
        age = int(rng.integers(0, 30))
        when = now - datetime.timedelta(days=age, hours=int(rng.integers(0, 12)), minutes=int(rng.integers(0, 60)))
        district = str(rng.choice(dists, p=p))
        servings = int(rng.choice(sizes, p=sp))
        best, _ = match_ngos(pool, district, servings)
        if age > 7:
            sp_status = [.07, .15, .70, .08]
        elif age > 2:
            sp_status = [.20, .35, .40, .05]
        else:
            sp_status = [.50, .35, .10, .05]
        rows.append({
            "donation_id": f"S{i + 1:03d}", "time": when.strftime("%Y-%m-%d %H:%M"),
            "name": str(rng.choice(names)), "phone": str(9800000000 + int(rng.integers(0, 99999999))),
            "venue_address": f"Sample venue, {district}", "district": district, "pincode": "",
            "servings": servings, "food_items": str(rng.choice(["Dal, rice, roti", "Paneer, roti, sweets",
                                                                  "Biryani, raita", "Mixed meals"])),
            "veg_nonveg": "Veg" if rng.random() < .8 else "Non-veg",
            "delivery_option": str(rng.choice(["I can deliver", "I need volunteer support"], p=[.45, .55])),
            "photo": "", "matched_ngo_ids": ", ".join(best["ngo_id"]),
            "matched_ngo_names": " | ".join(best["ngo_name"]),
            "match_scores": " | ".join(best["match_score"].astype(str)),
            "distances_km": " | ".join(best["distance_km"].astype(str)),
            "status": str(rng.choice(STATUSES, p=sp_status)),
        })
    return pd.DataFrame(rows).sort_values("time", ascending=False).reset_index(drop=True)


def update_status(donation_id, new_status):
    raw = pd.read_csv(DONATIONS_FILE, dtype=str).fillna("")
    if "status" not in raw.columns:
        raw["status"] = "Pending"
    raw.loc[raw["donation_id"] == donation_id, "status"] = new_status
    raw.to_csv(DONATIONS_FILE, index=False)


def dashboard(d):
    if d.empty:
        st.info("No donations yet. Charts will appear after the first donation.")
        return
    done = d[d["status"] == "Completed"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Donations", len(d))
    c2.metric("Servings offered", int(d["servings"].sum()))
    c3.metric("Servings delivered", int(done["servings"].sum()))
    c4.metric("Completion rate", f"{100 * len(done) / len(d):.0f}%")
    c5, c6 = st.columns(2)
    c5.metric("Avg. best match score", f"{d['best_score'].mean():.1f}")
    ids = d["matched_ngo_ids"].astype(str).str.split(",").explode().str.strip()
    c6.metric("Different NGOs matched", ids[ids != ""].nunique())

    st.markdown("##### Donations per day")
    per_day = d.groupby("day").size()
    per_day = per_day.reindex(pd.date_range(per_day.index.min(), per_day.index.max()), fill_value=0)
    st.bar_chart(per_day)

    a, b = st.columns(2)
    with a:
        st.markdown("##### Servings by district")
        st.bar_chart(d.groupby("district")["servings"].sum().sort_values(ascending=False))
    with b:
        st.markdown("##### Donation status")
        st.bar_chart(d["status"].value_counts().reindex(STATUSES, fill_value=0))

    a, b = st.columns(2)
    with a:
        st.markdown("##### Veg and non-veg")
        st.bar_chart(d["veg_nonveg"].value_counts())
    with b:
        st.markdown("##### Delivery option")
        st.bar_chart(d["delivery_option"].value_counts())

    st.markdown("##### Most suggested NGOs")
    names = d["matched_ngo_names"].astype(str).str.split("|").explode().str.strip()
    top = names[names != ""].value_counts().head(10)
    st.bar_chart(top, horizontal=True)


def tracking_log(d, live):
    if d.empty:
        st.info("No donations yet.")
        return
    view = d[["donation_id", "time", "name", "district", "servings", "status",
              "matched_ngo_names", "match_scores", "distances_km"]]
    st.dataframe(view, width="stretch")
    st.download_button("Download full log (CSV)", d[LOG_COLUMNS].to_csv(index=False),
                       "donation_log.csv")

    st.markdown("##### Donation detail and model output")
    pick = st.selectbox("Choose a donation", d.index,
                        format_func=lambda i: f"{d.loc[i, 'donation_id']} · {d.loc[i, 'time']} · "
                                              f"{d.loc[i, 'name']} · {d.loc[i, 'servings']} servings")
    r = d.loc[pick]
    with st.container(border=True):
        st.markdown(f"**Donor:** {r['name']}  ·  📞 {r['phone']}")
        st.write(f"Venue: {r['venue_address']}  ({r['district']} {r['pincode']})")
        st.write(f"{r['servings']} servings · {r['food_items']} ({r['veg_nonveg']}) · {r['delivery_option']}")
        st.write(f"Current status: **{r['status']}**")

    ids = [x.strip() for x in str(r["matched_ngo_ids"]).split(",") if x.strip()]
    nm = [x.strip() for x in str(r["matched_ngo_names"]).split("|")]
    sc = [x.strip() for x in str(r["match_scores"]).split("|")]
    ds = [x.strip() for x in str(r["distances_km"]).split("|")]
    ng = load_ngos().set_index("ngo_id")
    rows = []
    for i, nid in enumerate(ids):
        rows.append({
            "rank": i + 1,
            "ngo": nm[i] if i < len(nm) else "",
            "distance_km": ds[i] if i < len(ds) else "",
            "match_score": sc[i] if i < len(sc) else "",
            "phone": ng.loc[nid, "first_phone"] if nid in ng.index else "",
            "email": ng.loc[nid, "first_email"] if nid in ng.index else "",
            "volunteer": ng.loc[nid, "volunteer_contact_info"] if nid in ng.index else "",
        })
    st.markdown("Model suggestions for this donation:")
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    if live:
        new = st.selectbox("Update status", STATUSES, index=STATUSES.index(r["status"])
                           if r["status"] in STATUSES else 0, key=f"st_{r['donation_id']}")
        if st.button("Save status"):
            update_status(r["donation_id"], new)
            st.success("Status updated.")
            st.rerun()
        img = os.path.join(UPLOAD_DIR, str(r["photo"]))
        if r["photo"] and os.path.exists(img):
            st.markdown("Food photo:")
            try:
                st.image(img, width=320)
            except Exception:
                st.caption("Photo could not be displayed.")
        else:
            st.caption("Photo not available (the server may have restarted).")


def admin_page():
    st.header("Admin dashboard")
    if not st.session_state.get("is_admin"):
        pw = st.text_input("Admin password", type="password", key="admin_pw").strip()
        if st.button("Log in"):
            try:
                real = str(st.secrets["ADMIN_PASSWORD"]).strip()
            except Exception:
                real = "admin123"
            if pw == real:
                st.session_state["is_admin"] = True
                st.rerun()
            else:
                st.error("Wrong password.")
        return

    source = st.radio("Data source", ["Live data", "Sample data (for report)"], horizontal=True)
    live = source == "Live data"
    if live:
        d = normalise(read_donations())
    else:
        st.info("Showing SAMPLE data generated by the matching model. Not real donations.")
        d = normalise(make_sample_data())

    tab1, tab2, tab3 = st.tabs(["Progress dashboard", "Tracking log", "NGO database"])
    with tab1:
        dashboard(d)
    with tab2:
        tracking_log(d, live)
    with tab3:
        ngos = load_ngos()
        st.dataframe(ngos[["ngo_id", "ngo_name", "category", "district",
                           "available_capacity", "first_phone", "first_email"]], width="stretch")
    if st.button("Log out"):
        st.session_state.pop("is_admin")
        st.rerun()


# ---------------------------------------------------------------- main
st.title("🍲 ZeroWaste Meal")
st.caption("Connecting surplus food with NGOs, shelter homes, orphanages and old age homes.")
page = st.sidebar.radio("Go to", ["Donor", "NGO", "Admin"])
{"Donor": donor_page, "NGO": ngo_page, "Admin": admin_page}[page]()
