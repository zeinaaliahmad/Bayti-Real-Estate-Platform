from __future__ import annotations

import json
import os
import uuid
from datetime import datetime

import joblib
import pandas as pd
from flask import Flask, flash, jsonify, redirect, render_template, request, session, url_for
from flask_login import LoginManager, UserMixin, current_user, login_required, login_user, logout_user
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import or_
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

from services.chatbot_engine import ChatbotEngine

from flask import Flask, flash, jsonify, redirect, render_template, request, url_for
from flask_login import (
    LoginManager,
    UserMixin,
    current_user,
    login_required,
    login_user,
    logout_user,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

SALE_MODEL_PATH = os.path.join(MODEL_DIR, "price_model.joblib")
SALE_MODEL_METADATA_PATH = os.path.join(MODEL_DIR, "price_model_metadata.json")

RENT_MODEL_PATH = os.path.join(MODEL_DIR, "rent_price_model.joblib")
RENT_MODEL_METADATA_PATH = os.path.join(MODEL_DIR, "rent_price_model_metadata.json")

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev_secret_key_change_this")
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///bayti.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["UPLOAD_FOLDER"] = os.path.join("static", "uploads", "sell")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

bot = ChatbotEngine(
    data_path=os.path.join(MODEL_DIR, "properties_with_embeddings.pkl")
)

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

db = SQLAlchemy(app)


def load_model_bundle(model_path: str, metadata_path: str, default_mae: float, default_r2: float) -> dict:
    model = joblib.load(model_path) if os.path.exists(model_path) else None
    metadata = {}
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception:
            metadata = {}

    return {
        "model": model,
        "metadata": metadata,
        "mae": float(metadata.get("metrics", {}).get("mae", default_mae)),
        "r2": float(metadata.get("metrics", {}).get("r2", default_r2)),
    }


SALE_MODEL_BUNDLE = load_model_bundle(
    SALE_MODEL_PATH,
    SALE_MODEL_METADATA_PATH,
    default_mae=259152.50,
    default_r2=0.71,
)

RENT_MODEL_BUNDLE = load_model_bundle(
    RENT_MODEL_PATH,
    RENT_MODEL_METADATA_PATH,
    default_mae=500.00,
    default_r2=0.65,
)


def get_model_bundle_for_goal(goal: str) -> tuple[str, dict]:
    listing_mode = "sale" if str(goal).strip().lower() == "sell" else "rent"
    bundle = SALE_MODEL_BUNDLE if listing_mode == "sale" else RENT_MODEL_BUNDLE
    return listing_mode, bundle


# -----------------------------
# Models
# -----------------------------
class PriceAbuseReport(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    listing_link = db.Column(db.String(500))
    property_title = db.Column(db.String(200), nullable=False)
    location = db.Column(db.String(200), nullable=False)
    listed_price = db.Column(db.Float, nullable=False)
    expected_price = db.Column(db.Float)
    report_type = db.Column(db.String(100), nullable=False)
    contact_email = db.Column(db.String(150))
    details = db.Column(db.Text, nullable=False)
    evidence = db.Column(db.Text)
    created_at = db.Column(db.DateTime, server_default=db.func.now())


class Agent(db.Model):
    __tablename__ = "agents"

    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(150), nullable=False)
    agency_name = db.Column(db.String(150))
    phone = db.Column(db.String(50))
    email = db.Column(db.String(120))
    whatsapp = db.Column(db.String(50))
    rating = db.Column(db.Float, default=0.0)
    years_experience = db.Column(db.Integer, default=0)
    languages = db.Column(db.String(200))
    city = db.Column(db.String(100))
    district = db.Column(db.String(100))
    services = db.Column(db.String(100))
    is_verified = db.Column(db.Boolean, default=False)
    photo_url = db.Column(db.String(255))
    bio = db.Column(db.Text)
    total_listings = db.Column(db.Integer, default=0)


class Property(db.Model):
    __tablename__ = "properties"

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    listing_type = db.Column(db.String(20), nullable=False)
    property_type = db.Column(db.String(50), nullable=False)
    city = db.Column(db.String(100), nullable=False)
    district = db.Column(db.String(100))
    neighborhood = db.Column(db.String(100))
    address = db.Column(db.String(255))
    duration = db.Column(db.String(20), nullable=True)
    price_usd = db.Column(db.Float, nullable=False)
    area_sqm = db.Column(db.Float, nullable=False)
    bedrooms = db.Column(db.Integer, default=0)
    bathrooms = db.Column(db.Integer, default=0)
    floor_number = db.Column(db.Integer)
    total_floors = db.Column(db.Integer)
    furnished = db.Column(db.Boolean, default=False)
    parking_spaces = db.Column(db.Integer, default=0)
    balcony = db.Column(db.Boolean, default=False)
    elevator = db.Column(db.Boolean, default=False)
    generator_available = db.Column(db.Boolean, default=False)
    water_supply = db.Column(db.String(100))
    heating = db.Column(db.Boolean, default=False)
    ac = db.Column(db.Boolean, default=False)
    solar_ready = db.Column(db.Boolean, default=False)

    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)

    description = db.Column(db.Text)
    fair_price_min = db.Column(db.Float)
    fair_price_max = db.Column(db.Float)
    price_per_sqm = db.Column(db.Float)
    year_built = db.Column(db.Integer)
    status = db.Column(db.String(30), default="available")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    state_electricity = db.Column(db.String(100))
    generator_cap = db.Column(db.String(100))
    solar_ready_text = db.Column(db.String(100))
    city_water = db.Column(db.String(100))
    tank_capacity = db.Column(db.String(100))
    filtration = db.Column(db.String(100))
    downtown_distance = db.Column(db.String(100))
    traffic_level = db.Column(db.String(100))
    quiet_hours = db.Column(db.String(100))

    market_position = db.Column(db.String(50))
    pricing_reason = db.Column(db.Text)
    price_history = db.Column(db.Text)

    feature_listing_year = db.Column(db.Integer)
    feature_listing_month = db.Column(db.Integer)
    feature_listing_quarter = db.Column(db.Integer)

    macro_inflation_cpi_pct = db.Column(db.Float)
    macro_real_gdp_growth_pct = db.Column(db.Float)
    macro_debt_to_gdp_pct = db.Column(db.Float)
    macro_current_account_pct_gdp = db.Column(db.Float)
    macro_source_label = db.Column(db.String(255))

    market_city_listing_count = db.Column(db.Integer)
    market_city_avg_ppsqm = db.Column(db.Float)
    market_city_median_ppsqm = db.Column(db.Float)

    market_district_listing_count = db.Column(db.Integer)
    market_district_avg_ppsqm = db.Column(db.Float)
    market_district_median_ppsqm = db.Column(db.Float)

    market_city_type_listing_count = db.Column(db.Integer)
    market_city_type_avg_ppsqm = db.Column(db.Float)
    market_city_type_median_ppsqm = db.Column(db.Float)

    market_district_type_listing_count = db.Column(db.Integer)
    market_district_type_avg_ppsqm = db.Column(db.Float)
    market_district_type_median_ppsqm = db.Column(db.Float)

    market_neighborhood_listing_count = db.Column(db.Integer)
    market_neighborhood_avg_ppsqm = db.Column(db.Float)
    market_neighborhood_median_ppsqm = db.Column(db.Float)

    market_local_avg_ppsqm = db.Column(db.Float)
    market_local_listing_count = db.Column(db.Integer)
    market_feature_source = db.Column(db.String(255))

    feature_downtown_distance_km_proxy = db.Column(db.Float)
    feature_traffic_score = db.Column(db.Float)
    feature_quiet_score = db.Column(db.Float)

    agent_id = db.Column(db.Integer, db.ForeignKey("agents.id"), nullable=True)

    images = db.relationship(
        "PropertyImage",
        backref="property",
        lazy=True,
        cascade="all, delete-orphan",
    )
    favorites = db.relationship(
        "Favorite",
        backref="property",
        lazy=True,
        cascade="all, delete-orphan",
    )


class ContactMessage(db.Model):
    __tablename__ = "contact_messages"

    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), nullable=False)
    subject = db.Column(db.String(200), nullable=False)
    message = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)


class PropertyImage(db.Model):
    __tablename__ = "property_images"

    id = db.Column(db.Integer, primary_key=True)
    property_id = db.Column(db.Integer, db.ForeignKey("properties.id"), nullable=False)
    image_url = db.Column(db.String(255), nullable=False)
    is_primary = db.Column(db.Boolean, default=False)


login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"


class User(UserMixin, db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(50), default="buyer")
    city = db.Column(db.String(100))

    favorites = db.relationship(
        "Favorite",
        backref="user",
        lazy=True,
        cascade="all, delete-orphan",
    )


class Favorite(db.Model):
    __tablename__ = "favorites"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    property_id = db.Column(db.Integer, db.ForeignKey("properties.id"), nullable=False)


class AgentMessage(db.Model):
    __tablename__ = "agent_messages"

    id = db.Column(db.Integer, primary_key=True)
    sender_user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    agent_id = db.Column(db.Integer, db.ForeignKey("agents.id"), nullable=False)
    property_id = db.Column(db.Integer, db.ForeignKey("properties.id"), nullable=False)
    full_name = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), nullable=False)
    phone = db.Column(db.String(50))
    message = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, server_default=db.func.now())


class SellDraft(db.Model):
    __tablename__ = "sell_drafts"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    goal = db.Column(db.String(50), nullable=False, default="sell")
    location = db.Column(db.String(200), nullable=False)
    city = db.Column(db.String(100))
    district = db.Column(db.String(100))
    property_type = db.Column(db.String(50), nullable=False)
    size_sqm = db.Column(db.Float, nullable=False)
    overall_condition = db.Column(db.String(100), nullable=False)
    solar_system = db.Column(db.Boolean, default=False)
    water_security = db.Column(db.String(100))
    electricity = db.Column(db.String(100))
    estimated_low = db.Column(db.Float)
    estimated_high = db.Column(db.Float)
    suggested_price = db.Column(db.Float)
    confidence_score = db.Column(db.Float)
    status = db.Column(db.String(20), default="draft")
    created_at = db.Column(db.DateTime, server_default=db.func.now())
    updated_at = db.Column(db.DateTime, server_default=db.func.now(), onupdate=db.func.now())


class SellDraftImage(db.Model):
    __tablename__ = "sell_draft_images"

    id = db.Column(db.Integer, primary_key=True)
    draft_id = db.Column(db.Integer, db.ForeignKey("sell_drafts.id"), nullable=False)
    image_url = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, server_default=db.func.now())


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


# -----------------------------
# Helpers
# -----------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def save_uploaded_files(files):
    saved_paths = []
    for file in files:
        if not file or file.filename == "":
            continue
        if allowed_file(file.filename):
            ext = file.filename.rsplit(".", 1)[1].lower()
            unique_name = f"{uuid.uuid4().hex}.{ext}"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(unique_name))
            file.save(filepath)
            saved_paths.append(filepath.replace("\\", "/"))
    return saved_paths


def extract_city_district(location_text):
    if not location_text:
        return None, None
    parts = [p.strip() for p in location_text.split(",") if p.strip()]
    if len(parts) == 1:
        return parts[0], None
    return parts[0], parts[1]


def parse_yes_no(value, default=False):
    if value is None:
        return default
    return str(value).strip().lower() in {"yes", "true", "1", "on"}


def parse_optional_int(form, key, default=None):
    value = form.get(key, "")
    if value is None or str(value).strip() == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_form_bool_for_template(value, default=False):
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"yes", "true", "1", "on"}


MACRO_BY_YEAR = {
    2024: {
        "macro_inflation_cpi_pct": 45.2,
        "macro_real_gdp_growth_pct": -7.1,
        "macro_debt_to_gdp_pct": 176.5,
        "macro_current_account_pct_gdp": -22.2,
        "macro_source_label": "World Bank Lebanon indicators / Lebanon Economic Monitor Spring 2025",
    },
    2025: {
        "macro_inflation_cpi_pct": 14.6,
        "macro_real_gdp_growth_pct": 3.5,
        "macro_debt_to_gdp_pct": 155.2,
        "macro_current_account_pct_gdp": -15.8,
        "macro_source_label": "World Bank Macro Poverty Outlook April 2026",
    },
    2026: {
        "macro_inflation_cpi_pct": 14.6,
        "macro_real_gdp_growth_pct": 3.5,
        "macro_debt_to_gdp_pct": 155.2,
        "macro_current_account_pct_gdp": -15.8,
        "macro_source_label": "Carry-forward of latest available 2025 World Bank estimate",
    },
}


def map_distance_proxy(value):
    mapping = {
        "Very Near": 1.0,
        "Near": 3.0,
        "Medium": 7.0,
        "Far": 12.0,
    }
    return mapping.get((value or "").strip(), None)


def map_traffic_score(value):
    mapping = {"Low": 1.0, "Medium": 2.0, "High": 3.0}
    return mapping.get((value or "").strip(), None)


def map_quiet_score(value):
    mapping = {
        "Busy Area": 1.0,
        "Moderate": 2.0,
        "Very Quiet": 3.0,
        "10 PM - 7 AM": 3.0,
    }
    return mapping.get((value or "").strip(), None)


def safe_ppsqm(price_usd, area_sqm):
    if not price_usd or not area_sqm or area_sqm <= 0:
        return None
    return float(price_usd) / float(area_sqm)


def get_group_stats(query, price_per_sqm_getter):
    rows = query.all()
    values = [price_per_sqm_getter(p) for p in rows]
    values = [v for v in values if v is not None]
    if not values:
        return {"count": 0, "avg": None, "median": None}

    values_sorted = sorted(values)
    n = len(values_sorted)
    if n % 2 == 1:
        median_val = values_sorted[n // 2]
    else:
        median_val = (values_sorted[n // 2 - 1] + values_sorted[n // 2]) / 2

    return {
        "count": len(values_sorted),
        "avg": sum(values_sorted) / len(values_sorted),
        "median": median_val,
    }


def build_market_and_macro_snapshot(city, district, neighborhood, property_type, listing_type, listed_dt=None):
    today = listed_dt or datetime.utcnow()
    year = today.year
    month = today.month
    quarter = ((month - 1) // 3) + 1

    price_getter = lambda p: safe_ppsqm(p.price_usd, p.area_sqm)
    base_query = Property.query.filter(Property.listing_type.ilike(listing_type))

    city_stats = get_group_stats(base_query.filter(Property.city.ilike(city)), price_getter) if city else {"count": 0, "avg": None, "median": None}
    district_stats = get_group_stats(base_query.filter(Property.district.ilike(district)), price_getter) if district else {"count": 0, "avg": None, "median": None}
    city_type_stats = get_group_stats(
        base_query.filter(Property.city.ilike(city), Property.property_type.ilike(property_type)), price_getter
    ) if city and property_type else {"count": 0, "avg": None, "median": None}
    district_type_stats = get_group_stats(
        base_query.filter(Property.district.ilike(district), Property.property_type.ilike(property_type)), price_getter
    ) if district and property_type else {"count": 0, "avg": None, "median": None}
    neighborhood_stats = get_group_stats(base_query.filter(Property.neighborhood.ilike(neighborhood)), price_getter) if neighborhood else {"count": 0, "avg": None, "median": None}

    local_avg = district_type_stats["avg"] or city_type_stats["avg"] or district_stats["avg"] or city_stats["avg"]
    local_count = district_type_stats["count"] or city_type_stats["count"] or district_stats["count"] or city_stats["count"] or 0
    macro = MACRO_BY_YEAR.get(year, MACRO_BY_YEAR[2025])

    return {
        "feature_listing_year": year,
        "feature_listing_month": month,
        "feature_listing_quarter": quarter,
        "macro_inflation_cpi_pct": macro["macro_inflation_cpi_pct"],
        "macro_real_gdp_growth_pct": macro["macro_real_gdp_growth_pct"],
        "macro_debt_to_gdp_pct": macro["macro_debt_to_gdp_pct"],
        "macro_current_account_pct_gdp": macro["macro_current_account_pct_gdp"],
        "macro_source_label": macro["macro_source_label"],
        "market_city_listing_count": city_stats["count"],
        "market_city_avg_ppsqm": city_stats["avg"],
        "market_city_median_ppsqm": city_stats["median"],
        "market_district_listing_count": district_stats["count"],
        "market_district_avg_ppsqm": district_stats["avg"],
        "market_district_median_ppsqm": district_stats["median"],
        "market_city_type_listing_count": city_type_stats["count"],
        "market_city_type_avg_ppsqm": city_type_stats["avg"],
        "market_city_type_median_ppsqm": city_type_stats["median"],
        "market_district_type_listing_count": district_type_stats["count"],
        "market_district_type_avg_ppsqm": district_type_stats["avg"],
        "market_district_type_median_ppsqm": district_type_stats["median"],
        "market_neighborhood_listing_count": neighborhood_stats["count"],
        "market_neighborhood_avg_ppsqm": neighborhood_stats["avg"],
        "market_neighborhood_median_ppsqm": neighborhood_stats["median"],
        "market_local_avg_ppsqm": local_avg,
        "market_local_listing_count": local_count,
        "market_feature_source": "Derived from Bayti internal listings",
        "feature_downtown_distance_km_proxy": None,
        "feature_traffic_score": None,
        "feature_quiet_score": None,
    }


def get_condition_factor(condition):
    mapping = {
        "Brand New / Under Construction": 1.10,
        "Excellent": 1.05,
        "Good": 1.00,
        "Needs Renovation": 0.88,
    }
    return mapping.get(condition, 1.00)


def compute_confidence(similar_count):
    if similar_count >= 10:
        return 94
    if similar_count >= 7:
        return 90
    if similar_count >= 4:
        return 84
    return 72


def find_comparables(city, district, property_type, size_sqm, listing_type="sale", limit=8):
    query = Property.query.filter(
        Property.listing_type == listing_type,
        Property.property_type.ilike(property_type),
    )

    if city:
        query = query.filter(Property.city.ilike(city))
    if district:
        query = query.filter(Property.district.ilike(district))

    min_area = size_sqm * 0.75
    max_area = size_sqm * 1.25

    query = query.filter(
        Property.area_sqm >= min_area,
        Property.area_sqm <= max_area,
        Property.price_usd.isnot(None),
        Property.area_sqm.isnot(None),
        Property.area_sqm > 0,
    )

    comps = query.limit(limit).all()

    if len(comps) < 3 and city:
        query = Property.query.filter(
            Property.listing_type == listing_type,
            Property.property_type.ilike(property_type),
            Property.city.ilike(city),
            Property.area_sqm >= min_area,
            Property.area_sqm <= max_area,
            Property.price_usd.isnot(None),
            Property.area_sqm.isnot(None),
            Property.area_sqm > 0,
        )
        comps = query.limit(limit).all()

    return comps


def generate_ai_summary(form_data, valuation_result):
    city = form_data.get("city") or "this area"
    solar_note = "with solar support" if form_data.get("solar_system") else "without solar support"
    goal = str(form_data.get("goal", "sell")).strip().lower()

    if goal == "rent":
        return (
            f"Based on the trained rental model and local comparable {form_data.get('property_type', 'properties').lower()} listings in {city}, "
            f"your estimated fair rental price is around ${valuation_result['suggested_price']:,}. "
            f"The expected range is ${valuation_result['low_estimate']:,} to ${valuation_result['high_estimate']:,}. "
            f"This estimate reflects a {form_data.get('overall_condition', 'good')} property, {solar_note}, "
            f"and local living-standard adjustments for electricity, water, and neighborhood context."
        )

    return (
        f"Based on the trained sale model and local comparable {form_data.get('property_type', 'properties').lower()} listings in {city}, "
        f"your estimated fair property value is around ${valuation_result['suggested_price']:,}. "
        f"The current range is ${valuation_result['low_estimate']:,} to ${valuation_result['high_estimate']:,}. "
        f"This estimate reflects a {form_data.get('overall_condition', 'good')} property, {solar_note}, "
        f"and local living-standard adjustments for electricity, water, and neighborhood context."
    )


def get_primary_image(property_obj):
    property_type = (property_obj.property_type or "").strip().lower()
    property_id = property_obj.id or 1

    if "apartment" in property_type:
        folder, prefix, total_images = "apartments", "ap", 37
    elif "villa" in property_type:
        folder, prefix, total_images = "villas", "v", 16
    elif "office" in property_type:
        folder, prefix, total_images = "offices", "o", 17
    elif "land" in property_type:
        folder, prefix, total_images = "land", "l", 7
    elif "shop" in property_type:
        folder, prefix, total_images = "shops", "s", 16
    elif "penthouse" in property_type:
        folder, prefix, total_images = "penthouses", "p", 7
    elif "townhouse" in property_type:
        folder, prefix, total_images = "townhouses", "t", 7
    elif "duplex" in property_type:
        folder, prefix, total_images = "duplex", "dup", 17
    elif "studio" in property_type:
        folder, prefix, total_images = "studios", "st", 11
    else:
        return url_for("static", filename="images/default.jpg")

    image_number = ((property_id - 1) % total_images) + 1
    return url_for("static", filename=f"images/{folder}/{prefix}{image_number}.jpg")


def get_agent_image(agent_obj):
    agent_id = agent_obj.id or 1
    return url_for("static", filename=f"images/agents/agent{agent_id}.jpg")


def build_ml_input_for_prediction(
    listing_type,
    location,
    property_type,
    size_sqm,
    overall_condition,
    bedrooms,
    bathrooms,
    year_built,
    parking_spaces,
    floor_number,
    total_floors,
    furnished,
    elevator,
    balcony,
    heating,
    ac,
    solar_system,
    water_security,
    electricity,
    generator_cap,
    tank_capacity,
    filtration,
    traffic_level,
    quiet_hours,
):
    city, district = extract_city_district(location)
    snapshot = build_market_and_macro_snapshot(
        city=city,
        district=district,
        neighborhood=None,
        property_type=property_type,
        listing_type=listing_type,
        listed_dt=datetime.utcnow(),
    )

    snapshot["feature_traffic_score"] = map_traffic_score(traffic_level)
    snapshot["feature_quiet_score"] = map_quiet_score(quiet_hours)

    row = {
        "property_type": property_type,
        "city": city or "",
        "district": district or "",
        "neighborhood": "",
        "state_electricity": electricity or "",
        "generator_cap": generator_cap or "",
        "city_water": water_security or "",
        "tank_capacity": tank_capacity or "",
        "filtration": filtration or "",
        "traffic_level": traffic_level or "",
        "quiet_hours": quiet_hours or "",
        "market_feature_source": snapshot.get("market_feature_source", ""),
        "macro_source_label": snapshot.get("macro_source_label", ""),
        "status": "available",
        "area_sqm": float(size_sqm) if size_sqm else 0,
        "bedrooms": bedrooms if bedrooms is not None else 0,
        "bathrooms": bathrooms if bathrooms is not None else 1,
        "floor_number": floor_number,
        "total_floors": total_floors,
        "parking_spaces": parking_spaces if parking_spaces is not None else 0,
        "latitude": None,
        "longitude": None,
        "year_built": year_built,
        "feature_listing_year": snapshot.get("feature_listing_year"),
        "feature_listing_month": snapshot.get("feature_listing_month"),
        "feature_listing_quarter": snapshot.get("feature_listing_quarter"),
        "macro_inflation_cpi_pct": snapshot.get("macro_inflation_cpi_pct"),
        "macro_real_gdp_growth_pct": snapshot.get("macro_real_gdp_growth_pct"),
        "macro_debt_to_gdp_pct": snapshot.get("macro_debt_to_gdp_pct"),
        "macro_current_account_pct_gdp": snapshot.get("macro_current_account_pct_gdp"),
        "market_city_listing_count": snapshot.get("market_city_listing_count"),
        "market_city_avg_ppsqm": snapshot.get("market_city_avg_ppsqm"),
        "market_city_median_ppsqm": snapshot.get("market_city_median_ppsqm"),
        "market_district_listing_count": snapshot.get("market_district_listing_count"),
        "market_district_avg_ppsqm": snapshot.get("market_district_avg_ppsqm"),
        "market_district_median_ppsqm": snapshot.get("market_district_median_ppsqm"),
        "market_city_type_listing_count": snapshot.get("market_city_type_listing_count"),
        "market_city_type_avg_ppsqm": snapshot.get("market_city_type_avg_ppsqm"),
        "market_city_type_median_ppsqm": snapshot.get("market_city_type_median_ppsqm"),
        "market_district_type_listing_count": snapshot.get("market_district_type_listing_count"),
        "market_district_type_avg_ppsqm": snapshot.get("market_district_type_avg_ppsqm"),
        "market_district_type_median_ppsqm": snapshot.get("market_district_type_median_ppsqm"),
        "market_neighborhood_listing_count": snapshot.get("market_neighborhood_listing_count"),
        "market_neighborhood_avg_ppsqm": snapshot.get("market_neighborhood_avg_ppsqm"),
        "market_neighborhood_median_ppsqm": snapshot.get("market_neighborhood_median_ppsqm"),
        "market_local_avg_ppsqm": snapshot.get("market_local_avg_ppsqm"),
        "market_local_listing_count": snapshot.get("market_local_listing_count"),
        "feature_downtown_distance_km_proxy": snapshot.get("feature_downtown_distance_km_proxy"),
        "feature_traffic_score": snapshot.get("feature_traffic_score"),
        "feature_quiet_score": snapshot.get("feature_quiet_score"),
        "furnished": int(bool(furnished)),
        "balcony": int(bool(balcony)),
        "elevator": int(bool(elevator)),
        "heating": int(bool(heating)),
        "ac": int(bool(ac)),
        "solar_ready": int(bool(solar_system)),
    }

    return pd.DataFrame([row]), city, district, snapshot


def predict_price_ml(goal="sell", **kwargs):
    listing_type, model_bundle = get_model_bundle_for_goal(goal)
    model = model_bundle["model"]
    model_mae = model_bundle["mae"]
    model_r2 = model_bundle["r2"]

    if model is None:
        raise RuntimeError(
            f"Trained {listing_type} model not found. Please run train_price_model.py first."
        )

    input_df, city, district, snapshot = build_ml_input_for_prediction(
        listing_type=listing_type,
        **kwargs,
    )
    predicted_price = float(model.predict(input_df)[0])

    size_sqm = kwargs.get("size_sqm")
    overall_condition = kwargs.get("overall_condition")
    property_type = kwargs.get("property_type")

    local_avg_ppsqm = snapshot.get("market_local_avg_ppsqm")
    predicted_ppsqm = predicted_price / size_sqm if size_sqm else None

    low_estimate = max(0, round(predicted_price - model_mae))
    high_estimate = round(predicted_price + model_mae)
    confidence_score = max(50, min(95, round(model_r2 * 100)))

    if local_avg_ppsqm and predicted_ppsqm:
        if predicted_ppsqm > 1.10 * local_avg_ppsqm:
            trend = "Above local average"
        elif predicted_ppsqm < 0.90 * local_avg_ppsqm:
            trend = "Below local average"
        else:
            trend = "Near local average"
    else:
        trend = f"Estimated from trained {listing_type} ML model"

    comps = find_comparables(city, district, property_type, size_sqm, listing_type=listing_type)
    comp_cards = []
    for c in comps[:5]:
        comp_cards.append(
            {
                "title": c.title,
                "price": c.price_usd,
                "city": c.city,
                "district": c.district,
                "area_sqm": c.area_sqm,
                "price_per_sqm": round(c.price_usd / c.area_sqm, 2) if c.area_sqm else None,
            }
        )

    valuation = {
        "low_estimate": low_estimate,
        "high_estimate": high_estimate,
        "suggested_price": round(predicted_price),
        "confidence_score": confidence_score,
        "explanation": {
            "local_comps_avg_ppsqm": round(local_avg_ppsqm, 2) if local_avg_ppsqm else 0,
            "condition_factor": get_condition_factor(overall_condition),
            "market_trend": trend,
            "num_comps": len(comps),
            "listing_type": listing_type,
        },
        "comparables": comp_cards,
    }
    return valuation, city, district, snapshot, listing_type


@app.context_processor
def inject_helpers():
    return dict(get_primary_image=get_primary_image, get_agent_image=get_agent_image)


# -----------------------------
# Routes
# -----------------------------
@app.route("/trust-certification")
def trust_certification():
    return render_template("trust_certification.html")


@app.route("/ai-insights-guide")
def ai_insights_guide():
    return render_template("ai_insights_guide.html")


@app.route("/terms-of-service")
def terms_of_service():
    return render_template("terms_of_service.html")


@app.route("/privacy-policy")
def privacy_policy():
    return render_template("privacy_policy.html")


@app.route("/cookies-settings")
def cookies_settings():
    return render_template("cookies_settings.html")


@app.route("/contact-us", methods=["GET", "POST"])
def contact_us():
    if request.method == "POST":
        full_name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        subject = request.form.get("subject", "").strip()
        message = request.form.get("message", "").strip()

        if not full_name or not email or not subject or not message:
            flash("Please fill in all fields.", "error")
            return redirect(url_for("contact_us"))

        db.session.add(ContactMessage(full_name=full_name, email=email, subject=subject, message=message))
        db.session.commit()
        flash("Your message has been sent successfully.", "success")
        return redirect(url_for("contact_us"))

    return render_template("contact_us.html")


@app.route("/test")
def test():
    properties = Property.query.limit(10).all()
    return "<br>".join(
        [f"{p.title} | city={p.city} | district={p.district} | neighborhood={p.neighborhood} | area={p.area_sqm}" for p in properties]
    )


@app.route("/test-agents")
def test_agents():
    agents = Agent.query.all()
    return "<br>".join(
        [f"{a.full_name} | {a.city} | {a.district} | {a.languages} | {a.services} | verified={a.is_verified}" for a in agents]
    ) or "No agents in database"


@app.route("/")
def home():
    properties = Property.query.order_by(Property.created_at.desc(), Property.id.desc()).limit(8).all()
    agents = Agent.query.order_by(Agent.rating.desc(), Agent.full_name.asc()).limit(5).all()
    return render_template("home.html", properties=properties, agents=agents, get_primary_image=get_primary_image)


@app.route("/api/rent-map")
def rent_map_data():
    city = request.args.get("city", type=str)
    district = request.args.get("district", type=str)
    min_price = request.args.get("min_price", type=float)
    max_price = request.args.get("max_price", type=float)
    bedrooms = request.args.get("bedrooms", type=int)

    query = Property.query.filter(Property.listing_type == "rent")
    if city:
        query = query.filter(Property.city.ilike(f"%{city}%"))
    if district:
        query = query.filter(Property.district.ilike(f"%{district}%"))
    if min_price is not None:
        query = query.filter(Property.price_usd >= min_price)
    if max_price is not None:
        query = query.filter(Property.price_usd <= max_price)
    if bedrooms is not None:
        query = query.filter(Property.bedrooms >= bedrooms)

    query = query.filter(Property.latitude.isnot(None), Property.longitude.isnot(None))

    return jsonify(
        [
            {
                "id": p.id,
                "title": p.title,
                "price_usd": p.price_usd,
                "city": p.city,
                "district": p.district,
                "bedrooms": p.bedrooms,
                "bathrooms": p.bathrooms,
                "area_sqm": p.area_sqm,
                "latitude": p.latitude,
                "longitude": p.longitude,
                "url": url_for("property_details", property_id=p.id),
            }
            for p in query.all()
        ]
    )


@app.route("/buy")
def buy():
    property_type = request.args.get("type")
    city = request.args.get("city")
    district = request.args.get("district")
    min_price = request.args.get("min_price", type=float)
    max_price = request.args.get("max_price", type=float)
    min_area = request.args.get("min_area", type=float)
    max_area = request.args.get("max_area", type=float)
    bedrooms = request.args.get("bedrooms")
    bathrooms = request.args.get("bathrooms")
    furnished = request.args.get("furnished")
    balcony = request.args.get("balcony")
    elevator = request.args.get("elevator")
    parking = request.args.get("parking")
    generator = request.args.get("generator")
    ac = request.args.get("ac")
    heating = request.args.get("heating")
    solar_ready = request.args.get("solar_ready")
    sort = request.args.get("sort", "newest")
    page = request.args.get("page", 1, type=int)

    query = Property.query.filter(Property.listing_type.ilike("%sale%"))

    if city and city.strip():
        location_text = city.strip()
        query = query.filter(
            or_(
                Property.city.ilike(f"%{location_text}%"),
                Property.district.ilike(f"%{location_text}%"),
                Property.neighborhood.ilike(f"%{location_text}%"),
            )
        )
    if district and district.strip():
        query = query.filter(Property.district.ilike(f"%{district.strip()}%"))
    if property_type and property_type != "any":
        query = query.filter(Property.property_type.ilike(f"%{property_type}%"))
    if min_price is not None:
        query = query.filter(Property.price_usd >= min_price)
    if max_price is not None:
        query = query.filter(Property.price_usd <= max_price)
    if min_area is not None:
        query = query.filter(Property.area_sqm >= min_area)
    if max_area is not None:
        query = query.filter(Property.area_sqm <= max_area)

    if bedrooms and bedrooms != "any":
        if bedrooms == "studio":
            query = query.filter(Property.bedrooms == 0)
        elif bedrooms == "5+":
            query = query.filter(Property.bedrooms >= 5)
        else:
            query = query.filter(Property.bedrooms == int(bedrooms))

    if bathrooms and bathrooms != "any":
        if bathrooms == "5+":
            query = query.filter(Property.bathrooms >= 5)
        else:
            query = query.filter(Property.bathrooms == int(bathrooms))

    if furnished == "1":
        query = query.filter(Property.furnished.is_(True))
    if balcony == "1":
        query = query.filter(Property.balcony.is_(True))
    if elevator == "1":
        query = query.filter(Property.elevator.is_(True))
    if parking == "1":
        query = query.filter(Property.parking_spaces > 0)
    if generator == "1":
        query = query.filter(Property.generator_available.is_(True))
    if ac == "1":
        query = query.filter(Property.ac.is_(True))
    if heating == "1":
        query = query.filter(Property.heating.is_(True))
    if solar_ready == "1":
        query = query.filter(Property.solar_ready.is_(True))

    if sort == "newest":
        query = query.order_by(Property.id.desc())
    elif sort == "oldest":
        query = query.order_by(Property.id.asc())
    elif sort == "price_low":
        query = query.order_by(Property.price_usd.asc())
    elif sort == "price_high":
        query = query.order_by(Property.price_usd.desc())
    elif sort == "area_low":
        query = query.order_by(Property.area_sqm.asc())
    elif sort == "area_high":
        query = query.order_by(Property.area_sqm.desc())
    else:
        query = query.order_by(Property.id.desc())

    pagination = query.paginate(page=page, per_page=8, error_out=False)

    favorite_ids = set()
    favorite_map = {}
    if current_user.is_authenticated:
        user_favorites = Favorite.query.filter_by(user_id=current_user.id).all()
        favorite_ids = {int(f.property_id) for f in user_favorites}
        favorite_map = {int(f.property_id): f.id for f in user_favorites}

    return render_template(
        "buy.html",
        properties=pagination.items,
        pagination=pagination,
        favorite_ids=favorite_ids,
        favorite_map=favorite_map,
        selected_type=property_type,
        selected_city=city,
        selected_district=district,
        min_price=min_price,
        max_price=max_price,
        min_area=min_area,
        max_area=max_area,
        selected_bedrooms=bedrooms,
        selected_bathrooms=bathrooms,
        furnished=furnished,
        balcony=balcony,
        elevator=elevator,
        parking=parking,
        generator=generator,
        ac=ac,
        heating=heating,
        solar_ready=solar_ready,
        selected_sort=sort,
        get_primary_image=get_primary_image,
    )


@app.route("/rent")
def rent():
    selected_city = request.args.get("city", "").strip()
    selected_type = request.args.get("type", "").strip()
    selected_bedrooms = request.args.get("bedrooms", "").strip()
    selected_bathrooms = request.args.get("bathrooms", "").strip()
    sort_by = request.args.get("sort_by", "newest")
    min_price = request.args.get("min_price", type=float)
    max_price = request.args.get("max_price", type=float)
    generator = request.args.get("generator")
    balcony = request.args.get("balcony")
    elevator = request.args.get("elevator")
    furnished = request.args.get("furnished")
    parking = request.args.get("parking")
    ac = request.args.get("ac")
    heating = request.args.get("heating")
    solar_ready = request.args.get("solar_ready")
    selected_duration = request.args.get("duration", "")

    query = Property.query.filter(Property.listing_type.ilike("%rent%"))
    if selected_city:
        query = query.filter(Property.city.ilike(f"%{selected_city}%"))
    if selected_type:
        query = query.filter(Property.property_type.ilike(f"%{selected_type}%"))
    if min_price is not None:
        query = query.filter(Property.price_usd >= min_price)
    if max_price is not None:
        query = query.filter(Property.price_usd <= max_price)
    if generator == "1":
        query = query.filter(Property.generator_available.is_(True))
    if balcony == "1":
        query = query.filter(Property.balcony.is_(True))
    if elevator == "1":
        query = query.filter(Property.elevator.is_(True))
    if furnished == "1":
        query = query.filter(Property.furnished.is_(True))
    if parking == "1":
        query = query.filter(Property.parking_spaces > 0)
    if ac == "1":
        query = query.filter(Property.ac.is_(True))
    if heating == "1":
        query = query.filter(Property.heating.is_(True))
    if solar_ready == "1":
        query = query.filter(Property.solar_ready.is_(True))

    if selected_bedrooms and selected_bedrooms != "any":
        if selected_bedrooms == "studio":
            query = query.filter(Property.bedrooms == 0)
        elif selected_bedrooms == "5+":
            query = query.filter(Property.bedrooms >= 5)
        else:
            query = query.filter(Property.bedrooms == int(selected_bedrooms))

    if selected_bathrooms and selected_bathrooms != "any":
        if selected_bathrooms == "5+":
            query = query.filter(Property.bathrooms >= 5)
        else:
            query = query.filter(Property.bathrooms == int(selected_bathrooms))

    if sort_by == "price_low_high":
        query = query.order_by(Property.price_usd.asc())
    elif sort_by == "price_high_low":
        query = query.order_by(Property.price_usd.desc())
    elif sort_by == "area_low_high":
        query = query.order_by(Property.area_sqm.asc())
    elif sort_by == "area_high_low":
        query = query.order_by(Property.area_sqm.desc())
    else:
        query = query.order_by(Property.id.desc())
        
    if selected_duration:
        query = query.filter(Property.duration == selected_duration)
    
    properties = query.all()
    favorite_ids = set()
    favorite_map = {}
    if current_user.is_authenticated:
        favorites = Favorite.query.filter_by(user_id=current_user.id).all()
        favorite_ids = {fav.property_id for fav in favorites}
        favorite_map = {fav.property_id: fav.id for fav in favorites}

    return render_template(
        "rent.html",
        properties=properties,
        favorite_ids=favorite_ids,
        favorite_map=favorite_map,
        selected_city=selected_city,
        selected_type=selected_type,
        selected_bedrooms=selected_bedrooms,
        selected_bathrooms=selected_bathrooms,
        min_price=min_price,
        max_price=max_price,
        generator=generator,
        balcony=balcony,
        elevator=elevator,
        furnished=furnished,
        parking=parking,
        ac=ac,
        heating=heating,
        solar_ready=solar_ready,
        sort_by=sort_by,
        selected_duration=selected_duration,
    )


@app.route("/property/<int:property_id>")
def property_details(property_id):
    property_obj = Property.query.get_or_404(property_id)
    property_images = PropertyImage.query.filter_by(property_id=property_obj.id).all()
    agent = Agent.query.get(property_obj.agent_id) if property_obj.agent_id else None

    favorite_ids = set()
    favorite_map = {}
    if current_user.is_authenticated:
        favorites = Favorite.query.filter_by(user_id=current_user.id).all()
        favorite_ids = {fav.property_id for fav in favorites}
        favorite_map = {fav.property_id: fav.id for fav in favorites}

    similar_properties = get_similar_properties(property_obj, limit=3)
    insight = compute_market_insight(property_obj)

    nearby_places = []
    if property_obj.neighborhood:
        nearby_places.append(f"Near {property_obj.neighborhood}")
    if property_obj.district:
        nearby_places.append(f"{property_obj.district} district")
    if property_obj.city:
        nearby_places.append(f"{property_obj.city} center")

    return render_template(
        "property-details.html",
        property=property_obj,
        property_images=property_images,
        agent=agent,
        favorite_ids=favorite_ids,
        favorite_map=favorite_map,
        similar_properties=similar_properties,
        nearby_places=nearby_places,
        price_history=insight["price_history"],
        computed_market_position=insight["market_position"],
        computed_pricing_reason=insight["pricing_reason"],
        current_ppsqm=insight["current_ppsqm"],
        avg_ppsqm=insight["avg_ppsqm"],
    )


@app.route("/sell", methods=["GET", "POST"])
@login_required
def sell():
    if request.method == "POST":
        action = request.form.get("action", "estimate")
        goal = request.form.get("goal", "sell").strip().lower()
        listing_mode, model_bundle = get_model_bundle_for_goal(goal)

        if model_bundle["model"] is None:
            flash(
                f"The {listing_mode} ML model was not found. Please train the model first.",
                "error",
            )
            return redirect(url_for("sell"))

        location = request.form.get("location", "").strip()
        property_type = request.form.get("property_type", "").strip()
        size_sqm = request.form.get("size_sqm", type=float)
        overall_condition = request.form.get("overall_condition", "").strip()

        bedrooms = parse_optional_int(request.form, "bedrooms", 0)
        bathrooms = parse_optional_int(request.form, "bathrooms", 1)
        year_built = parse_optional_int(request.form, "year_built")
        parking_spaces = parse_optional_int(request.form, "parking_spaces", 0)
        floor_number = parse_optional_int(request.form, "floor_number")
        total_floors = parse_optional_int(request.form, "total_floors")

        furnished = parse_yes_no(request.form.get("furnished"), False)
        elevator = parse_yes_no(request.form.get("elevator"), False)
        balcony = parse_yes_no(request.form.get("balcony"), False)
        heating = parse_yes_no(request.form.get("heating"), False)
        ac = parse_yes_no(request.form.get("ac"), False)
        solar_system = parse_yes_no(request.form.get("solar_system"), True)

        water_security = request.form.get("water_security", "").strip()
        electricity = request.form.get("electricity", "").strip()
        generator_cap = request.form.get("generator_cap", "").strip()
        tank_capacity = request.form.get("tank_capacity", "").strip()
        filtration = request.form.get("filtration", "").strip()
        traffic_level = request.form.get("traffic_level", "").strip()
        quiet_hours = request.form.get("quiet_hours", "").strip()

        if not location or not property_type or not size_sqm or not overall_condition:
            flash("Please fill in all required fields.", "error")
            return redirect(url_for("sell"))

        valuation, city, district, snapshot, listing_type = predict_price_ml(
            goal=goal,
            location=location,
            property_type=property_type,
            size_sqm=size_sqm,
            overall_condition=overall_condition,
            bedrooms=bedrooms,
            bathrooms=bathrooms,
            year_built=year_built,
            parking_spaces=parking_spaces,
            floor_number=floor_number,
            total_floors=total_floors,
            furnished=furnished,
            elevator=elevator,
            balcony=balcony,
            heating=heating,
            ac=ac,
            solar_system=solar_system,
            water_security=water_security,
            electricity=electricity,
            generator_cap=generator_cap,
            tank_capacity=tank_capacity,
            filtration=filtration,
            traffic_level=traffic_level,
            quiet_hours=quiet_hours,
        )

        uploaded_files = request.files.getlist("images")
        saved_images = save_uploaded_files(uploaded_files)

        draft = SellDraft(
            user_id=current_user.id,
            goal=goal,
            location=location,
            city=city,
            district=district,
            property_type=property_type,
            size_sqm=size_sqm,
            overall_condition=overall_condition,
            solar_system=solar_system,
            water_security=water_security,
            electricity=electricity,
            estimated_low=valuation["low_estimate"],
            estimated_high=valuation["high_estimate"],
            suggested_price=valuation["suggested_price"],
            confidence_score=valuation["confidence_score"],
            status="draft" if action == "save_draft" else "published",
        )
        db.session.add(draft)
        db.session.flush()

        for img_path in saved_images:
            db.session.add(SellDraftImage(draft_id=draft.id, image_url=img_path))

        if action == "publish":
            local_avg_ppsqm = snapshot["market_local_avg_ppsqm"] or 0.0
            predicted_ppsqm = valuation["suggested_price"] / size_sqm if size_sqm else None
            generator_available = bool(generator_cap) or ("generator" in electricity.lower())

            new_property = Property(
                title=f"{property_type} in {location}",
                listing_type=listing_type,
                property_type=property_type,
                city=city or "",
                district=district or "",
                neighborhood=None,
                price_usd=valuation["suggested_price"],
                area_sqm=size_sqm,
                bedrooms=bedrooms or 0,
                bathrooms=bathrooms or 0,
                floor_number=floor_number,
                total_floors=total_floors,
                furnished=furnished,
                parking_spaces=parking_spaces or 0,
                balcony=balcony,
                elevator=elevator,
                generator_available=generator_available,
                heating=heating,
                ac=ac,
                solar_ready=solar_system,
                price_per_sqm=predicted_ppsqm,
                year_built=year_built,
                state_electricity=electricity or None,
                generator_cap=generator_cap or None,
                solar_ready_text="Yes" if solar_system else "No",
                city_water=water_security or None,
                tank_capacity=tank_capacity or None,
                filtration=filtration or None,
                traffic_level=traffic_level or None,
                quiet_hours=quiet_hours or None,
                market_position="Fair" if local_avg_ppsqm == 0 else (
                    "Above Market" if predicted_ppsqm and predicted_ppsqm > 1.10 * local_avg_ppsqm
                    else "Below Market" if predicted_ppsqm and predicted_ppsqm < 0.90 * local_avg_ppsqm
                    else "Fair"
                ),
                pricing_reason=(
                    f"Auto-generated from Bayti {listing_type} ML estimate using local market aggregates. Local average is ${local_avg_ppsqm:,.0f}/sqm."
                    if local_avg_ppsqm
                    else f"Auto-generated from Bayti {listing_type} ML estimate. No strong local aggregate available yet."
                ),
                description=generate_ai_summary(
                    {
                        "goal": goal,
                        "city": city,
                        "property_type": property_type,
                        "overall_condition": overall_condition,
                        "solar_system": solar_system,
                    },
                    valuation,
                ),
                feature_listing_year=snapshot["feature_listing_year"],
                feature_listing_month=snapshot["feature_listing_month"],
                feature_listing_quarter=snapshot["feature_listing_quarter"],
                macro_inflation_cpi_pct=snapshot["macro_inflation_cpi_pct"],
                macro_real_gdp_growth_pct=snapshot["macro_real_gdp_growth_pct"],
                macro_debt_to_gdp_pct=snapshot["macro_debt_to_gdp_pct"],
                macro_current_account_pct_gdp=snapshot["macro_current_account_pct_gdp"],
                macro_source_label=snapshot["macro_source_label"],
                market_city_listing_count=snapshot["market_city_listing_count"],
                market_city_avg_ppsqm=snapshot["market_city_avg_ppsqm"],
                market_city_median_ppsqm=snapshot["market_city_median_ppsqm"],
                market_district_listing_count=snapshot["market_district_listing_count"],
                market_district_avg_ppsqm=snapshot["market_district_avg_ppsqm"],
                market_district_median_ppsqm=snapshot["market_district_median_ppsqm"],
                market_city_type_listing_count=snapshot["market_city_type_listing_count"],
                market_city_type_avg_ppsqm=snapshot["market_city_type_avg_ppsqm"],
                market_city_type_median_ppsqm=snapshot["market_city_type_median_ppsqm"],
                market_district_type_listing_count=snapshot["market_district_type_listing_count"],
                market_district_type_avg_ppsqm=snapshot["market_district_type_avg_ppsqm"],
                market_district_type_median_ppsqm=snapshot["market_district_type_median_ppsqm"],
                market_neighborhood_listing_count=snapshot["market_neighborhood_listing_count"],
                market_neighborhood_avg_ppsqm=snapshot["market_neighborhood_avg_ppsqm"],
                market_neighborhood_median_ppsqm=snapshot["market_neighborhood_median_ppsqm"],
                market_local_avg_ppsqm=snapshot["market_local_avg_ppsqm"],
                market_local_listing_count=snapshot["market_local_listing_count"],
                market_feature_source=snapshot["market_feature_source"],
                feature_downtown_distance_km_proxy=snapshot["feature_downtown_distance_km_proxy"],
                feature_traffic_score=snapshot["feature_traffic_score"],
                feature_quiet_score=snapshot["feature_quiet_score"],
            )
            db.session.add(new_property)

        db.session.commit()

        ai_summary = generate_ai_summary(
            {
                "goal": goal,
                "city": city,
                "property_type": property_type,
                "overall_condition": overall_condition,
                "solar_system": solar_system,
            },
            valuation,
        )

        flash(
            "Draft saved successfully." if action == "save_draft" else
            "Listing published successfully." if action == "publish" else
            "Estimate generated successfully.",
            "success",
        )

        return render_template(
            "sell.html",
            valuation=valuation,
            ai_summary=ai_summary,
            comparables=valuation["comparables"],
            form_data={
                "goal": goal,
                "location": location,
                "property_type": property_type,
                "size_sqm": size_sqm,
                "overall_condition": overall_condition,
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "year_built": year_built,
                "parking_spaces": parking_spaces,
                "floor_number": floor_number,
                "total_floors": total_floors,
                "furnished": furnished,
                "elevator": elevator,
                "balcony": balcony,
                "heating": heating,
                "ac": ac,
                "solar_system": solar_system,
                "water_security": water_security,
                "electricity": electricity,
                "generator_cap": generator_cap,
                "tank_capacity": tank_capacity,
                "filtration": filtration,
                "traffic_level": traffic_level,
                "quiet_hours": quiet_hours,
            },
            uploaded_images=saved_images,
        )

    return render_template(
        "sell.html",
        valuation=None,
        ai_summary=None,
        comparables=[],
        form_data={},
        uploaded_images=[],
    )


@app.route("/find-agent")
def find_agent():
    location = request.args.get("location", "").strip()
    service = request.args.get("service", "").strip()
    language = request.args.get("language", "").strip()
    verified_only = request.args.get("verified_only", "").strip()

    query = Agent.query
    if location:
        query = query.filter(or_(Agent.city.ilike(f"%{location}%"), Agent.district.ilike(f"%{location}%")))
    if service:
        query = query.filter(Agent.services.ilike(f"%{service}%"))
    if language:
        query = query.filter(Agent.languages.ilike(f"%{language}%"))
    if verified_only == "1":
        query = query.filter(Agent.is_verified.is_(True))

    filtered_agents = query.order_by(Agent.rating.desc(), Agent.years_experience.desc()).all()
    top_agents = Agent.query.order_by(Agent.rating.desc(), Agent.years_experience.desc()).limit(3).all()

    return render_template(
        "find_agent.html",
        agents=filtered_agents,
        top_agents=top_agents,
        selected_location=location,
        selected_service=service,
        selected_language=language,
        selected_verified=verified_only,
    )


@app.route("/agent/<int:agent_id>")
def agent_detail(agent_id):
    agent = Agent.query.get_or_404(agent_id)
    return render_template("agent_detail.html", agent=agent)


def compute_market_insight(property_obj):
    if not property_obj.area_sqm or property_obj.area_sqm <= 0 or not property_obj.price_usd:
        return {
            "market_position": "fair",
            "pricing_reason": "Not enough data to evaluate this price.",
            "current_ppsqm": None,
            "avg_ppsqm": None,
            "price_history": [],
        }

    current_ppsqm = property_obj.price_usd / property_obj.area_sqm
    comparables = Property.query.filter(
        Property.id != property_obj.id,
        Property.listing_type == property_obj.listing_type,
        Property.property_type == property_obj.property_type,
        Property.city == property_obj.city,
        Property.area_sqm.isnot(None),
        Property.area_sqm > 0,
        Property.price_usd.isnot(None),
        Property.price_usd > 0,
    ).all()

    comparable_ppsqm = [comp.price_usd / comp.area_sqm for comp in comparables if comp.area_sqm and comp.price_usd]

    if not comparable_ppsqm:
        return {
            "market_position": "fair",
            "pricing_reason": f"No comparable {property_obj.property_type.lower()} listings were found in {property_obj.city}.",
            "current_ppsqm": round(current_ppsqm, 2),
            "avg_ppsqm": None,
            "price_history": [],
        }

    avg_ppsqm = sum(comparable_ppsqm) / len(comparable_ppsqm)
    ratio = current_ppsqm / avg_ppsqm if avg_ppsqm > 0 else 1

    if ratio < 0.90:
        market_position = "undervalued"
    elif ratio > 1.10:
        market_position = "overpriced"
    else:
        market_position = "fair"

    pct_diff = abs((current_ppsqm - avg_ppsqm) / avg_ppsqm) * 100 if avg_ppsqm > 0 else 0
    condition_text = "current"
    neighborhood_text = property_obj.neighborhood or property_obj.district or property_obj.city

    if market_position == "undervalued":
        pricing_reason = (
            f"Priced about {pct_diff:.0f}% below comparable listings in {neighborhood_text} based on price per sqm, "
            f"{property_obj.property_type.lower()} type, {condition_text} condition, {property_obj.area_sqm:.0f} sqm size, and current market patterns."
        )
    elif market_position == "overpriced":
        pricing_reason = (
            f"Priced about {pct_diff:.0f}% above comparable listings in {neighborhood_text} based on price per sqm, "
            f"{property_obj.property_type.lower()} type, {condition_text} condition, {property_obj.area_sqm:.0f} sqm size, and current market patterns."
        )
    else:
        pricing_reason = (
            f"Priced within the expected market range in {neighborhood_text} based on price per sqm, "
            f"{property_obj.property_type.lower()} type, {condition_text} condition, and recent comparable listings."
        )

    price_history = [round(property_obj.price_usd * 0.93), round(property_obj.price_usd * 0.96), round(property_obj.price_usd * 0.98), round(property_obj.price_usd)]

    return {
        "market_position": market_position,
        "pricing_reason": pricing_reason,
        "current_ppsqm": round(current_ppsqm, 2),
        "avg_ppsqm": round(avg_ppsqm, 2),
        "price_history": price_history,
    }


@app.route("/report-abuse", methods=["GET", "POST"])
def report_abuse():
    if request.method == "POST":
        flash("Your report has been submitted successfully.", "success")
        return redirect(url_for("report_abuse"))
    return render_template("report-abuse.html")


@app.route("/contact-agent/<int:property_id>", methods=["POST"])
@login_required
def contact_agent(property_id):
    property_obj = Property.query.get_or_404(property_id)
    full_name = request.form.get("full_name", "").strip()
    email = request.form.get("email", "").strip()
    phone = request.form.get("phone", "").strip()
    message = request.form.get("message", "").strip()

    if not full_name or not email or not message:
        flash("Please fill in name, email, and message.", "error")
        return redirect(url_for("property_details", property_id=property_id) + "#contact-agent-box")

    db.session.add(
        AgentMessage(
            sender_user_id=current_user.id,
            agent_id=property_obj.agent_id,
            property_id=property_obj.id,
            full_name=full_name,
            email=email,
            phone=phone,
            message=message,
        )
    )
    db.session.commit()
    flash("Your message has been sent to the agent.", "success")
    return redirect(url_for("property_details", property_id=property_id) + "#contact-agent-box")


@app.route("/contact-agent-general", methods=["POST"])
def contact_agent_general():
    message = request.form.get("message", "").strip()
    agent_name = request.form.get("agent_name", "").strip()
    if not message or not agent_name:
        flash("Please fill in all required fields.", "error")
        return redirect(url_for("find_agent"))
    flash(f"Your message to {agent_name} has been sent.", "success")
    return redirect(url_for("find_agent"))


def get_similar_properties(target, limit=3):
    candidates = Property.query.filter(
        Property.id != target.id,
        Property.listing_type == target.listing_type,
        Property.status == "available",
    ).all()

    scored = []
    for p in candidates:
        score = 0
        if p.property_type == target.property_type:
            score += 30
        if p.city == target.city:
            score += 20
        if p.district and target.district and p.district == target.district:
            score += 10
        if p.bedrooms == target.bedrooms:
            score += 10
        if p.bathrooms == target.bathrooms:
            score += 8
        if p.area_sqm and target.area_sqm:
            area_diff_ratio = abs(p.area_sqm - target.area_sqm) / max(target.area_sqm, 1)
            score += max(0, 12 - area_diff_ratio * 20)
        if p.price_usd and target.price_usd:
            price_diff_ratio = abs(p.price_usd - target.price_usd) / max(target.price_usd, 1)
            score += max(0, 20 - price_diff_ratio * 30)
        scored.append((score, p))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scored[:limit]]


@app.route("/favorites")
@login_required
def favorites():
    favorite_items = Favorite.query.filter_by(user_id=current_user.id).order_by(Favorite.id.desc()).all()
    return render_template("favorites.html", favorites=favorite_items, get_primary_image=get_primary_image)


@app.route("/add-favorite/<int:property_id>", methods=["POST"])
@login_required
def add_favorite(property_id):
    property_obj = Property.query.get_or_404(property_id)
    existing_favorite = Favorite.query.filter_by(user_id=current_user.id, property_id=property_obj.id).first()
    if not existing_favorite:
        db.session.add(Favorite(user_id=current_user.id, property_id=property_obj.id))
        db.session.commit()

    next_url = request.args.get("next") or request.form.get("next")
    if next_url:
        return redirect(next_url)
    return redirect(request.referrer or url_for("home"))


@app.route("/remove-favorite/<int:favorite_id>", methods=["POST"])
@login_required
def remove_favorite(favorite_id):
    favorite = Favorite.query.get_or_404(favorite_id)
    if favorite.user_id != current_user.id:
        flash("Unauthorized action.", "error")
        return redirect(url_for("favorites"))

    db.session.delete(favorite)
    db.session.commit()

    next_url = request.args.get("next") or request.form.get("next")
    if next_url:
        return redirect(next_url)
    return redirect(request.referrer or url_for("favorites"))


@app.route("/dashboard")
@login_required
def dashboard():
    user_favorites_count = Favorite.query.filter_by(user_id=current_user.id).count()
    return render_template("dashboard.html", user=current_user, user_favorites_count=user_favorites_count)


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not full_name or not email or not password or not confirm_password:
            flash("Please fill in all fields.", "error")
            return redirect(url_for("signup"))
        if password != confirm_password:
            flash("Passwords do not match.", "error")
            return redirect(url_for("signup"))
        if User.query.filter_by(email=email).first():
            flash("An account with this email already exists.", "error")
            return redirect(url_for("signup"))

        db.session.add(User(full_name=full_name, email=email, password_hash=generate_password_hash(password)))
        db.session.commit()
        flash("Account created successfully. Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("sign_up.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        if not email or not password:
            flash("Please fill in all fields.", "error")
            return redirect(url_for("login"))

        user = User.query.filter_by(email=email).first()
        if not user or not check_password_hash(user.password_hash, password):
            flash("Invalid email or password.", "error")
            return redirect(url_for("login"))

        login_user(user)
        flash("Logged in successfully.", "success")
        return redirect(url_for("dashboard"))

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "success")
    return redirect(url_for("home"))

@app.route("/ai-assistant")
def ai_assistant():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return render_template("ai_assistant.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        if "session_id" not in session:
            session["session_id"] = str(uuid.uuid4())

        data = request.get_json(silent=True) or {}
        user_message = (data.get("message") or "").strip()

        result = bot.process(user_message, session_id=session["session_id"])

        return jsonify(result)

    except Exception as e:
        print("CHAT ERROR:", e)
        return jsonify({
            "reply_html": f"<p>Backend error: {str(e)}</p>",
            "listings": [],
            "suggestions": [
                "Apartment near AUB under $800",
                "Studio for rent in Hamra",
                "2-bedroom in Achrafieh for sale"
            ],
            "market_insight": None,
            "mode": "error"
        }), 500


@app.route("/api/chat/clear", methods=["POST"])
def clear_chat():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    bot.clear_history(session["session_id"])
    return jsonify({"ok": True})

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)