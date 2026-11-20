# ZeroWaste Meal

A web app that connects people who have **surplus food** (weddings, gatherings, events)
with **NGOs, shelter homes, orphanages and old age homes**. A simple machine learning model
suggests the best NGOs for each donation based on **distance** and **available capacity**,
and shares contact details between donor and NGO.

Minor Engineering Project.

## Features

**Donor page**
- One short form: name, phone, venue, district, pincode, servings, food items, veg / non-veg, delivery option
- Food photo upload (for quality check) and a food-safety confirmation
- Instantly shows the top 3 matching NGOs with address, phone, email and volunteer contact
- Suggests a split across NGOs when the donation is larger than any single NGO can take

**NGO page**
- Login using the organisation name and access code (demo: access code = NGO ID, e.g. `NGO-001`)
- View organisation details, update free capacity, and see incoming donations with donor contacts

**Admin page**
- Password login
- Progress dashboard: donations per day, servings by district, status, veg / non-veg, delivery option, most suggested NGOs
- Tracking log of every donation with the model's suggested NGOs, scores and distances
- Update donation status (Pending, NGO contacted, Completed, Cancelled) and download the log as CSV
- Photo review for quality check
- "Sample data" switch to show a demo dashboard for reports

## How the matching model works

1. **Filter:** keep NGOs that are accepting donations, have free capacity, and are in the donor's district.
2. **Features:** for each NGO, calculate
   - `distance_km` using the haversine formula on the stored latitude and longitude
   - `capacity_gap` = |available capacity - donated servings|
3. **Rank:** scale both features to 0-1, weight them (60% distance, 40% capacity), and use
   scikit-learn `NearestNeighbors` to find the NGOs closest to the ideal point (zero distance, zero gap).
4. **Output:** top 3 NGOs with a match score out of 100.

Only stored data from the CSV is used. There are no live location or map services.

## Tech used

Python, pandas, NumPy, scikit-learn, Streamlit, Google Colab (model building and testing),
GitHub and Streamlit Community Cloud (deployment).

## Files

| File | Purpose |
|---|---|
| `app.py` | The Streamlit app (donor, NGO and admin pages and the matching model) |
| `Rajasthan_Combined_NGOs_Colab_Dataset.csv` | NGO dataset (54 NGOs across Rajasthan) |
| `requirements.txt` | Python packages needed to run the app |
| `README.md` | This file |
| `.gitignore` | Keeps generated files (donations, uploads, secrets) out of GitHub |

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Admin password

The default admin password is `admin123`. When deployed, change it in Streamlit
**Settings > Secrets** by adding:

```toml
ADMIN_PASSWORD = "your-new-password"
```

## Known limitations

- Streamlit Cloud has a temporary file system. Donations, uploaded photos and NGO capacity
  updates are lost when the app restarts or is redeployed. Download the log from the Admin
  page before that happens.
- The donor's location is estimated from the district or pincode (average position of NGOs
  there), not an exact address.
- The NGO login is a demo (access code = NGO ID). A real system would use proper passwords.
- The dataset has no veg / non-veg field for NGOs, so food type is recorded but not used for matching.
- Distance is a straight-line estimate, not road distance.

## Future scope

- Permanent database (Google Sheets or Supabase) and proper NGO passwords
- Exact address location with map and road distance
- Learning the weights from real donation outcomes (accepted or rejected matches)
- SMS or email alerts to NGOs and in-app messaging
