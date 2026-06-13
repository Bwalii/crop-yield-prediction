import pandas as pd

crop_df = pd.read_csv("crop_production.csv")
rain_df = pd.read_csv("rainfall in india 1901-2015.csv")


print("Crop dataset loaded successfully!")
print("Rainfall dataset loaded successfully!")
print("Crop dataset shape:", crop_df.shape)
print("Rainfall dataset shape:", rain_df.shape)
print("Crop Columns:")
print(crop_df.columns)

print("\nRainfall Columns:")
print(rain_df.columns)
print(crop_df.info())
print(rain_df.info())
print(crop_df.isnull().sum())
print(rain_df.isnull().sum())
crop_df = crop_df.dropna(subset=["Production"])
rain_df.fillna(rain_df.mean(numeric_only=True), inplace=True)
crop_df["Yield"] = crop_df["Production"] / crop_df["Area"]
rain_df.rename(columns={"SUBDIVISION": "State_Name"}, inplace=True)
rain_df["YEAR"] = rain_df["YEAR"].astype(int)
rain_df = rain_df[["State_Name", "YEAR", "ANNUAL", "Jun-Sep"]]
merged_df = pd.merge(
    crop_df,
    rain_df,
    left_on=["State_Name", "Crop_Year"],
    right_on=["State_Name", "YEAR"],
    how="left"
)
print("CROP PREVIEW")
print(crop_df.head())

print("\nRAINFALL PREVIEW")
print(rain_df.head())

print("\nCROP STATE NAMES SAMPLE")
print(crop_df["State_Name"].unique()[:10])

print("\nRAINFALL STATE_NAME SAMPLE")
print(rain_df["State_Name"].unique()[:10])

print("\nMISSING VALUES")
print(crop_df.isnull().sum())
print(rain_df.isnull().sum())
print("Crop States:", crop_df["State_Name"].nunique())
print("Rainfall States:", rain_df["State_Name"].nunique())

crop_states = set(crop_df["State_Name"].str.upper())
rain_states = set(rain_df["State_Name"].str.upper())

common_states = crop_states.intersection(rain_states)

print("\nCommon states found:", len(common_states))
print("\nSample common states:")
print(list(common_states)[:20])
print("Number of Crop States:")
print(crop_df["State_Name"].nunique())

print("\nCrop States:")
print(sorted(crop_df["State_Name"].unique()))
print("Number of Rainfall States:")
print(rain_df["State_Name"].nunique())

print("\nRainfall States:")
print(sorted(rain_df["State_Name"].unique()))
state_mapping = {
    "ANDAMAN AND NICOBAR ISLANDS": "ANDAMAN & NICOBAR ISLANDS",
    "ARUNACHAL PRADESH": "ARUNACHAL PRADESH",
    "ASSAM": "ASSAM & MEGHALAYA",
    "BIHAR": "BIHAR",
    "CHHATTISGARH": "CHHATTISGARH",
    "HARYANA": "HARYANA DELHI & CHANDIGARH",
    "CHANDIGARH": "HARYANA DELHI & CHANDIGARH",
    "HIMACHAL PRADESH": "HIMACHAL PRADESH",
    "JAMMU AND KASHMIR": "JAMMU & KASHMIR",
    "JHARKHAND": "JHARKHAND",
    "KERALA": "KERALA",
    "ODISHA": "ORISSA",
    "PUNJAB": "PUNJAB",
    "TAMIL NADU": "TAMIL NADU",
    "TELANGANA": "TELANGANA",
    "UTTARAKHAND": "UTTARAKHAND",
    "UTTAR PRADESH": "EAST UTTAR PRADESH",
    "WEST BENGAL": "GANGETIC WEST BENGAL",
    "SIKKIM": "SUB HIMALAYAN WEST BENGAL & SIKKIM",
    "GOA": "KONKAN & GOA"
}
crop_df["State_Upper"] = (
    crop_df["State_Name"]
    .str.upper()
    .str.strip()
)
crop_df["Mapped_State"] = crop_df["State_Upper"].replace(state_mapping)
print(
    crop_df[
        ["State_Name", "State_Upper", "Mapped_State"]
    ].drop_duplicates().head(20)
)
crop_df["Mapped_State"].nunique()