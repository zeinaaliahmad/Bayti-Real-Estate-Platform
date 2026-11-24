import os
from pathlib import Path
from datetime import datetime

import pandas as pd
from werkzeug.security import generate_password_hash

from app import app, db, Agent, User, Property, PropertyImage, Favorite


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def csv_path(filename: str) -> Path:
    return DATA_DIR / filename


def exists_col(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns


def clean_value(value, default=None):
    if pd.isna(value):
        return default
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return default
    return value


def to_bool(value, default=False):
    if pd.isna(value) or value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y", "t"}


def to_int(value, default=None):
    if pd.isna(value) or value is None or str(value).strip() == "":
        return default
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default


def to_float(value, default=None):
    if pd.isna(value) or value is None or str(value).strip() == "":
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def to_datetime_safe(value, default=None):
    if pd.isna(value) or value is None or str(value).strip() == "":
        return default
    try:
        return pd.to_datetime(value).to_pydatetime()
    except Exception:
        return default


def safe_get(row, key, default=None):
    if key not in row.index:
        return default
    return clean_value(row[key], default)


def ensure_required_files():
    required = [
        "agents.csv",
        "users.csv",
        "properties.csv",
        "property_images.csv",
        "favorites.csv",
    ]
    missing = [name for name in required if not csv_path(name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing CSV files in {DATA_DIR}: {', '.join(missing)}"
        )


def seed_agents():
    path = csv_path("agents.csv")
    df = pd.read_csv(path)

    # normalize headers
    df.columns = [str(col).strip().lower() for col in df.columns]

    print(f"Seeding agents from {path.name}...")
    print("Normalized columns found in agents.csv:")
    print(df.columns.tolist())

    for _, row in df.iterrows():
        agent_id = to_int(safe_get(row, "agent_id"))
        full_name = safe_get(row, "full_name")
        email = safe_get(row, "email")

        if not full_name or not email:
            print(f"Skipping agent row due to missing full_name/email: {row.to_dict()}")
            continue

        agent = Agent(
            id=agent_id,
            full_name=full_name,
            agency_name=safe_get(row, "agency_name"),
            phone=safe_get(row, "phone"),
            email=email.lower(),
            whatsapp=safe_get(row, "whatsapp"),
            rating=to_float(safe_get(row, "rating"), 0.0) or 0.0,
            years_experience=to_int(safe_get(row, "years_experience"), 0) or 0,
            languages=safe_get(row, "languages"),
            photo_url=safe_get(row, "photo_url"),
            total_listings=to_int(safe_get(row, "total_listings"), 0) or 0,
            city=safe_get(row, "city") or safe_get(row, "location"),
            district=safe_get(row, "district"),
            services=safe_get(row, "services"),
            is_verified=to_bool(safe_get(row, "is_verified"), False),
            bio=safe_get(row, "bio")
        )

        db.session.add(agent)

    db.session.commit()
    print("Seeded agents.")


def seed_users():
    path = csv_path("users.csv")
    df = pd.read_csv(path)

    print(f"Seeding users from {path.name}...")

    for _, row in df.iterrows():
        user_id = to_int(safe_get(row, "user_id"))
        full_name = safe_get(row, "full_name") or safe_get(row, "name") or "User"
        email = safe_get(row, "email")

        if not email:
            print(f"Skipping user row due to missing email: {row.to_dict()}")
            continue

        raw_password_hash = safe_get(row, "password_hash")
        raw_password = safe_get(row, "password")

        if raw_password_hash:
            password_hash = raw_password_hash
        elif raw_password:
            password_hash = generate_password_hash(raw_password)
        else:
            password_hash = generate_password_hash("12345678")

        user = User(
            id=user_id,
            full_name=full_name,
            email=email.lower(),
            password_hash=password_hash,
            role=safe_get(row, "role", "buyer") or "buyer",
            city=safe_get(row, "city"),
        )
        db.session.add(user)

    db.session.commit()
    print("Seeded users.")

def seed_properties():
    path = csv_path("properties.csv")
    df = pd.read_csv(path)

    # Normalize headers
    df.columns = [str(col).strip().lower() for col in df.columns]

    print(f"Seeding properties from {path.name}...")
    print("Normalized columns found in properties.csv:")
    print(df.columns.tolist())

    def row_get(row, *possible_keys, default=None):
        for key in possible_keys:
            key = str(key).strip().lower()
            if key in row.index:
                value = clean_value(row[key], None)
                if value is not None:
                    return value
        return default

    for _, row in df.iterrows():

        property_id = to_int(row_get(row, "property_id", "id"))
        title = row_get(row, "title")
        listing_type = row_get(row, "listing_type")
        property_type = row_get(row, "property_type")
        city = row_get(row, "city", "property_city", "location_city")
        district = row_get(row, "district", "region", "area")
        neighborhood = row_get(row, "neighborhood", "neighbor", "neighbourhood", "zone")

        if not title or not listing_type or not property_type or not city:
            print("Skipping property row due to missing required fields:")
            print(row.to_dict())
            continue

        area_sqm = to_float(row_get(row, "area_sqm", "area")) or 0.0
        price_usd = to_float(row_get(row, "price_usd", "price")) or 0.0

        bedrooms = to_int(row_get(row, "bedrooms"), 0) or 0
        bathrooms = to_int(row_get(row, "bathrooms"), 0) or 0

        parking_spaces = to_int(row_get(row, "parking_spaces", "parking"), 0) or 0
        balcony = to_bool(row_get(row, "has_balcony", "balcony"), False)
        elevator = to_bool(row_get(row, "has_elevator", "elevator"), False)

        generator_available = to_bool(row_get(row, "generator_available", "generator"), False)
        heating = to_bool(row_get(row, "heating"), False)
        ac = to_bool(row_get(row, "ac"), False)

        solar_ready = to_bool(row_get(row, "solar_ready"), False)

        price_per_sqm = to_float(row_get(row, "price_per_sqm"))
        if price_per_sqm is None and area_sqm > 0:
            price_per_sqm = price_usd / area_sqm

        listed_at = to_datetime_safe(
            row_get(row, "listed_date", "created_at"),
            datetime.utcnow()
        )

        prop = Property(
            id=property_id,
            title=title,
            listing_type=listing_type,
            property_type=property_type,
            city=city,
            district=district,
            neighborhood=neighborhood,
            address=row_get(row, "address"),
            duration=row.get("duration") if row.get("listing_type") == "Rent" else None,
            price_usd=price_usd,
            area_sqm=area_sqm,
            bedrooms=bedrooms,
            bathrooms=bathrooms,

            floor_number=to_int(row_get(row, "floor_number")),
            total_floors=to_int(row_get(row, "total_floors")),
            furnished=to_bool(row_get(row, "furnished"), False),

            parking_spaces=parking_spaces,
            balcony=balcony,
            elevator=elevator,

            generator_available=generator_available,
            water_supply=row_get(row, "water_supply"),
            heating=heating,
            ac=ac,
            solar_ready=solar_ready,

            latitude=to_float(row_get(row, "latitude")),
            longitude=to_float(row_get(row, "longitude")),

            description=row_get(row, "description"),

            fair_price_min=to_float(row_get(row, "fair_price_min")),
            fair_price_max=to_float(row_get(row, "fair_price_max")),
            price_per_sqm=price_per_sqm,

            year_built=to_int(row_get(row, "year_built")),
            status=row_get(row, "status", default="available") or "available",

            created_at=listed_at,

            # ✅ Lebanon Living
            state_electricity=row_get(row, "state_electricity"),
            generator_cap=row_get(row, "generator_cap"),
            city_water=row_get(row, "city_water"),
            tank_capacity=row_get(row, "tank_capacity"),
            filtration=row_get(row, "filtration"),

            downtown_distance=row_get(row, "downtown_distance"),
            traffic_level=row_get(row, "traffic_level"),
            quiet_hours=row_get(row, "quiet_hours"),

            # ✅ Pricing insights
            market_position=row_get(row, "market_position"),
            pricing_reason=row_get(row, "pricing_reason"),
            price_history=row_get(row, "price_history"),

            # ✅ Time / macro / market features
            feature_listing_year=to_int(row_get(row, "feature_listing_year")) or (listed_at.year if listed_at else None),
            feature_listing_month=to_int(row_get(row, "feature_listing_month")) or (listed_at.month if listed_at else None),
            feature_listing_quarter=to_int(row_get(row, "feature_listing_quarter")) or (((listed_at.month - 1) // 3) + 1 if listed_at else None),

            macro_inflation_cpi_pct=to_float(row_get(row, "macro_inflation_cpi_pct")),
            macro_real_gdp_growth_pct=to_float(row_get(row, "macro_real_gdp_growth_pct")),
            macro_debt_to_gdp_pct=to_float(row_get(row, "macro_debt_to_gdp_pct")),
            macro_current_account_pct_gdp=to_float(row_get(row, "macro_current_account_pct_gdp")),
            macro_source_label=row_get(row, "macro_source_label"),

            market_city_listing_count=to_int(row_get(row, "market_city_listing_count")),
            market_city_avg_ppsqm=to_float(row_get(row, "market_city_avg_ppsqm")),
            market_city_median_ppsqm=to_float(row_get(row, "market_city_median_ppsqm")),

            market_district_listing_count=to_int(row_get(row, "market_district_listing_count")),
            market_district_avg_ppsqm=to_float(row_get(row, "market_district_avg_ppsqm")),
            market_district_median_ppsqm=to_float(row_get(row, "market_district_median_ppsqm")),

            market_city_type_listing_count=to_int(row_get(row, "market_city_type_listing_count")),
            market_city_type_avg_ppsqm=to_float(row_get(row, "market_city_type_avg_ppsqm")),
            market_city_type_median_ppsqm=to_float(row_get(row, "market_city_type_median_ppsqm")),

            market_district_type_listing_count=to_int(row_get(row, "market_district_type_listing_count")),
            market_district_type_avg_ppsqm=to_float(row_get(row, "market_district_type_avg_ppsqm")),
            market_district_type_median_ppsqm=to_float(row_get(row, "market_district_type_median_ppsqm")),

            market_neighborhood_listing_count=to_int(row_get(row, "market_neighborhood_listing_count")),
            market_neighborhood_avg_ppsqm=to_float(row_get(row, "market_neighborhood_avg_ppsqm")),
            market_neighborhood_median_ppsqm=to_float(row_get(row, "market_neighborhood_median_ppsqm")),

            market_local_avg_ppsqm=to_float(row_get(row, "market_local_avg_ppsqm")),
            market_local_listing_count=to_int(row_get(row, "market_local_listing_count")),
            market_feature_source=row_get(row, "market_feature_source"),

            feature_downtown_distance_km_proxy=to_float(row_get(row, "feature_downtown_distance_km_proxy")),
            feature_traffic_score=to_float(row_get(row, "feature_traffic_score")),
            feature_quiet_score=to_float(row_get(row, "feature_quiet_score")),

            agent_id=to_int(row_get(row, "agent_id")),
        )

        db.session.add(prop)

        if property_id and property_id <= 5:
            print(f"DEBUG property {property_id}: {title}, {city}, {price_usd}")

    db.session.commit()
    print("Properties seeded successfully.")

    db.session.commit()
    print("Seeded properties.")


def seed_property_images():
    path = csv_path("property_images.csv")
    df = pd.read_csv(path)

    print(f"Seeding property images from {path.name}...")

    for _, row in df.iterrows():
        image_id = to_int(safe_get(row, "image_id"))
        property_id = to_int(safe_get(row, "property_id"))

        image_url = (
            safe_get(row, "image_url")
            or safe_get(row, "image_path")
            or safe_get(row, "url")
            or safe_get(row, "filename")
        )

        if property_id is None or not image_url:
            print(f"Skipping image row due to missing property_id/image_url: {row.to_dict()}")
            continue

        image = PropertyImage(
            id=image_id,
            property_id=property_id,
            image_url=image_url,
            is_primary=to_bool(safe_get(row, "is_primary"), False),
        )
        db.session.add(image)

    db.session.commit()
    print("Seeded property images.")


def seed_favorites():
    path = csv_path("favorites.csv")
    df = pd.read_csv(path)

    print(f"Seeding favorites from {path.name}...")

    for _, row in df.iterrows():
        favorite_id = to_int(safe_get(row, "favorite_id"))
        user_id = to_int(safe_get(row, "user_id"))
        property_id = to_int(safe_get(row, "property_id"))

        if user_id is None or property_id is None:
            print(f"Skipping favorite row due to missing user_id/property_id: {row.to_dict()}")
            continue

        favorite = Favorite(
            id=favorite_id,
            user_id=user_id,
            property_id=property_id,
        )
        db.session.add(favorite)

    db.session.commit()
    print("Seeded favorites.")


def main():
    ensure_required_files()

    with app.app_context():
        print("seed.py started")
        print("Inside app context")

        db.drop_all()
        db.create_all()
        print("Old data deleted and tables recreated")

        seed_agents()
        seed_users()
        seed_properties()
        seed_property_images()
        seed_favorites()

        print("Database seeded successfully.")


if __name__ == "__main__":
    main()