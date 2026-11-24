import html
import os
import pickle
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

load_dotenv()


@dataclass
class SearchFilters:
    listing_type: Optional[str] = None
    property_type: Optional[str] = None
    location: Optional[str] = None
    max_price: Optional[float] = None
    min_price: Optional[float] = None
    min_bedrooms: Optional[int] = None
    max_bedrooms: Optional[int] = None
    min_bathrooms: Optional[int] = None
    max_bathrooms: Optional[int] = None
    min_area: Optional[float] = None
    max_area: Optional[float] = None
    raw_text: str = ""


class ChatbotEngine:
    """
    Bayti conversational engine.

    What it does:
    - Uses a hosted Google model (Gemma via the Gemini API) for natural conversation.
    - Grounds property answers in your own listings dataset.
    - Falls back gracefully if the API is unavailable.
    - Returns structured JSON-ready data so the frontend can render cards.
    """

    def __init__(self, data_path: str = "models/properties_with_embeddings.pkl"):
        self.data_path = data_path
        self.df = self._load_properties(data_path)
        self.chat_histories: Dict[str, List[Dict[str, Any]]] = {}
        self.last_searches: Dict[str, Dict[str, Any]] = {}

        try:
            self.embedding_model = SentenceTransformer(
                os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
                local_files_only=True
            )
            self.use_embeddings = True
            print("✅ Local embedding model loaded.")
        except Exception as exc:
            print(f"⚠️ Embedding model unavailable locally: {exc}")
            print("⚠️ Using keyword search fallback instead.")
            self.embedding_model = None
            self.use_embeddings = False

        # Official Google GenAI SDK for Gemini API / hosted Gemma.
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.client = None
        self.use_llm = False
        self.model_name = os.getenv("BAYTI_LLM_MODEL", "gemma-4-26b-a4b-it")

        if self.api_key:
            try:
                from google import genai

                self.client = genai.Client(api_key=self.api_key)
                self.use_llm = True
                print(f"✅ Connected to Google GenAI model: {self.model_name}")
            except Exception as exc:
                print(f"⚠️ Could not initialize Google GenAI client: {exc}")
        else:
            print("⚠️ GEMINI_API_KEY not found. Bayti will use local fallback responses.")

        self.system_prompt = """
You are Bayti, an intelligent, warm, polished AI assistant for a Lebanese real estate platform.

Your role:
1. Hold natural conversations like a strong general-purpose assistant.
2. Help users search for homes, rentals, offices, land, and other listings in Lebanon.
3. When listing context is provided, only use those listing details. Never invent facts about specific properties.
4. When the question is general and not about a property search, answer normally and helpfully.
5. When the request is vague, ask only the most useful follow-up question(s).
6. Be concise, fluent, and natural. Avoid robotic replies.
7. Never claim you searched listings if no listing context is provided.
8. If the user asks for pricing fairness or comparisons, reason from the provided listing context only.
""".strip()

    # ---------------------------
    # Public API
    # ---------------------------
    def clear_history(self, session_id: str = "default") -> None:
        self.chat_histories[session_id] = []
        self.last_searches.pop(session_id, None)

    def process(self, user_input: str, session_id: str = "default") -> Dict[str, Any]:
        if not user_input or not user_input.strip():
            return {
                "reply_html": "<p>Please type a message.</p>",
                "listings": [],
                "suggestions": self._default_suggestions(),
                "market_insight": None,
                "mode": "chat",
            }

        user_input = user_input.strip()
        self.chat_histories.setdefault(session_id, [])

        intent = self._detect_intent(user_input, session_id=session_id)
        merged_message = self._merge_with_previous_search_if_needed(user_input, session_id)
        filters = self._extract_filters(merged_message)

        listings: List[Dict[str, Any]] = []
        market_insight = None
        mode = "chat"

        if intent in {"property_search", "property_followup", "mixed"}:
            candidates = self._retrieve_properties(merged_message, filters, top_k=24)
            listings = self._rows_to_cards(candidates.head(3))
            market_insight = self._build_market_insight(candidates)
            mode = "search"
            self.last_searches[session_id] = {
                "message": merged_message,
                "filters": filters,
                "results_count": len(candidates),
            }
            reply_html = self._generate_search_reply(
                user_input=user_input,
                full_query=merged_message,
                results_df=candidates.head(100),
                session_id=session_id,
            )
        else:
            reply_html = self._generate_general_reply(user_input=user_input, session_id=session_id)

        payload = {
            "reply_html": reply_html,
            "listings": listings,
            "suggestions": self._suggestions_for_turn(intent, listings),
            "market_insight": market_insight,
            "mode": mode,
        }

        self.chat_histories[session_id].append(
            {
                "user": user_input,
                "assistant": re.sub(r"<[^>]+>", "", reply_html),
                "mode": mode,
            }
        )
        self.chat_histories[session_id] = self.chat_histories[session_id][-12:]

        return payload

    # ---------------------------
    # Data loading
    # ---------------------------
    def _load_properties(self, data_path: str) -> pd.DataFrame:
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Could not find property dataset at '{data_path}'. "
                "Make sure your embeddings file exists."
            )

        with open(data_path, "rb") as handle:
            df = pickle.load(handle)

        if not isinstance(df, pd.DataFrame):
            raise ValueError("The embeddings file must contain a pandas DataFrame.")

        df = df.copy()
        df.columns = [str(col).strip() for col in df.columns]

        # Standardize common columns expected by the assistant.
        rename_map = {
            "type": "property_type",
            "listingType": "listing_type",
            "price_usd": "price",
            "area": "area_sqm",
            "beds": "bedrooms",
            "baths": "bathrooms",
        }
        for old, new in rename_map.items():
            if old in df.columns and new not in df.columns:
                df.rename(columns={old: new}, inplace=True)

        if "id" not in df.columns:
            df["id"] = np.arange(1, len(df) + 1)

        for col in ["price", "bedrooms", "bathrooms", "area_sqm"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Build listing text if missing.
        if "listing_text" not in df.columns:
            text_cols = [
                c
                for c in [
                    "title",
                    "property_type",
                    "listing_type",
                    "city",
                    "district",
                    "neighborhood",
                    "description",
                ]
                if c in df.columns
            ]
            df["listing_text"] = df[text_cols].fillna("").astype(str).agg(" ".join, axis=1)

        return df

    # ---------------------------
    # Intent / parsing
    # ---------------------------
    def _detect_intent(self, text: str, session_id: str = "default") -> str:
        t = text.lower().strip()

        greetings = {
            "hi", "hello", "hey", "good morning", "good afternoon", "good evening", "marhaba", "salam"
        }
        if t in greetings:
            return "general_chat"

        followup_words = {
            "cheaper", "more", "show more", "only studios", "only apartments", "with parking",
            "near aub", "sea view", "bigger", "smaller", "under", "above", "more expensive",
        }
        if any(word in t for word in followup_words) and session_id in self.last_searches:
            return "property_followup"

        property_keywords = [
            "apartment", "studio", "villa", "house", "office", "shop", "land", "property", "listing",
            "rent", "rental", "buy", "sale", "bedroom", "bathroom", "sqm", "m2", "price", "budget",
            "beirut", "hamra", "achrafieh", "jounieh", "tripoli", "saida", "tyre", "aub",
        ]
        general_keywords = [
            "explain", "what is", "how does", "why", "write", "email", "summary", "code", "python"
        ]

        property_score = sum(1 for kw in property_keywords if kw in t)
        general_score = sum(1 for kw in general_keywords if kw in t)

        if property_score >= 1 and general_score >= 1:
            return "mixed"
        if property_score >= 1:
            return "property_search"
        return "general_chat"

    def _merge_with_previous_search_if_needed(self, text: str, session_id: str) -> str:
        lowered = text.lower().strip()
        if session_id not in self.last_searches:
            return text

        short_followups = [
            "cheaper", "more", "show more", "only studios", "with parking", "near aub",
            "sea view", "bigger", "smaller", "2 bedrooms", "3 bedrooms", "under 800", "under $800",
        ]
        if any(token in lowered for token in short_followups):
            previous = self.last_searches[session_id]["message"]
            return f"{previous}. Additional preference: {text}"
        return text

    def _extract_filters(self, text: str) -> SearchFilters:
        t = text.lower()
        filters = SearchFilters(raw_text=text)

        listing_map = {
            "rent": ["rent", "rental", "monthly"],
            "sale": ["sale", "buy", "purchase", "for sale"],
        }
        for label, words in listing_map.items():
            if any(word in t for word in words):
                filters.listing_type = label
                break

        property_types = ["apartment", "studio", "villa", "house", "office", "shop", "land", "duplex", "chalet"]
        for ptype in property_types:
            if ptype in t:
                filters.property_type = ptype
                break

        # Price parsing
        price_under = re.search(r"(?:under|below|less than|max(?:imum)?|up to)\s*\$?\s*([\d,]+)", t)
        price_over = re.search(r"(?:over|above|more than|min(?:imum)?|at least)\s*\$?\s*([\d,]+)", t)
        if price_under:
            filters.max_price = float(price_under.group(1).replace(",", ""))
        if price_over:
            filters.min_price = float(price_over.group(1).replace(",", ""))

        # Bedrooms / bathrooms / area
        bed_match = re.search(r"(\d+)\s*(?:bed|beds|bedroom|bedrooms|bd)\b", t)
        bath_match = re.search(r"(\d+)\s*(?:bath|baths|bathroom|bathrooms|ba)\b", t)
        area_match = re.search(r"(\d+)\s*(?:sqm|m2|m²)", t)
        if bed_match:
            filters.min_bedrooms = int(bed_match.group(1))
        if bath_match:
            filters.min_bathrooms = int(bath_match.group(1))
        if area_match:
            filters.min_area = float(area_match.group(1))

        # Location extraction from actual dataset values when possible.
        location_columns = [c for c in ["city", "district", "neighborhood"] if c in self.df.columns]
        possible_locations = set()
        for col in location_columns:
            values = self.df[col].dropna().astype(str).str.strip().str.lower().unique().tolist()
            possible_locations.update(v for v in values if 2 <= len(v) <= 40)

        special_aliases = {
            "aub": "hamra",
            "ras beirut": "ras beirut",
            "bliss": "bliss",
        }
        for alias, canonical in special_aliases.items():
            if alias in t:
                filters.location = canonical
                return filters

        for loc in sorted(possible_locations, key=len, reverse=True):
            if loc and loc in t:
                filters.location = loc
                break

        return filters

    # ---------------------------
    # Retrieval
    # ---------------------------
    def _retrieve_properties(self, query: str, filters: SearchFilters, top_k: int = 24) -> pd.DataFrame:
        df = self.df.copy()

        # Semantic ranking first.
        if "embedding" in df.columns:
            query_embedding = self.embedding_model.encode(query)
            property_embeddings = np.vstack(df["embedding"].values)
            similarities = cosine_similarity([query_embedding], property_embeddings)[0]
            df["similarity"] = similarities
        else:
            query_terms = set(re.findall(r"\w+", query.lower()))
            df["similarity"] = df["listing_text"].fillna("").astype(str).str.lower().apply(
                lambda x: len(query_terms.intersection(set(re.findall(r"\w+", x))))
            )

        if filters.listing_type and "listing_type" in df.columns:
            df = df[df["listing_type"].astype(str).str.lower().str.contains(filters.listing_type, na=False)]

        if filters.property_type and "property_type" in df.columns:
            property_aliases = {
                "apartment": ["apartment", "flat", "condo", "duplex", "residential", "home"],
                "house": ["house", "villa", "townhouse", "duplex", "residential", "home"],
                "studio": ["studio", "apartment", "flat", "residential"],
                "villa": ["villa", "house", "residential"],
                "office": ["office", "commercial"],
                "shop": ["shop", "retail", "commercial"],
                "land": ["land", "parcel", "plot"],
            }

            allowed_types = property_aliases.get(filters.property_type, [filters.property_type])

            type_mask = pd.Series(False, index=df.index)
            for allowed in allowed_types:
                type_mask = type_mask | df["property_type"].astype(str).str.lower().str.contains(allowed, na=False)

            df = df[type_mask]

        if filters.location:
            location_mask = pd.Series(False, index=df.index)
            for col in ["city", "district", "neighborhood", "title", "listing_text"]:
                if col in df.columns:
                    location_mask = location_mask | df[col].astype(str).str.lower().str.contains(filters.location, na=False)
            df = df[location_mask]

        if filters.max_price is not None and "price" in df.columns:
            df = df[df["price"].fillna(np.inf) <= filters.max_price]
        if filters.min_price is not None and "price" in df.columns:
            df = df[df["price"].fillna(-np.inf) >= filters.min_price]
        if filters.min_bedrooms is not None and "bedrooms" in df.columns:
            df = df[df["bedrooms"].fillna(-np.inf) >= filters.min_bedrooms]
        if filters.max_bedrooms is not None and "bedrooms" in df.columns:
            df = df[df["bedrooms"].fillna(np.inf) <= filters.max_bedrooms]
        if filters.min_bathrooms is not None and "bathrooms" in df.columns:
            df = df[df["bathrooms"].fillna(-np.inf) >= filters.min_bathrooms]
        if filters.min_area is not None and "area_sqm" in df.columns:
            df = df[df["area_sqm"].fillna(-np.inf) >= filters.min_area]

        if df.empty:
            return df

        if "price" in df.columns:
            df = df.assign(price_numeric=pd.to_numeric(df["price"], errors="coerce"))
        return df.sort_values("similarity", ascending=False).head(top_k)

    def _format_context(self, df: pd.DataFrame) -> str:
        if df.empty:
            return "No matching listings found from the dataset."

        blocks = []
        for _, row in df.iterrows():
            blocks.append(
                "\n".join(
                    [
                        f"Title: {row.get('title', 'N/A')}",
                        f"Listing type: {row.get('listing_type', 'N/A')}",
                        f"Property type: {row.get('property_type', 'N/A')}",
                        f"City: {row.get('city', 'N/A')}",
                        f"District: {row.get('district', 'N/A')}",
                        f"Neighborhood: {row.get('neighborhood', 'N/A')}",
                        f"Bedrooms: {row.get('bedrooms', 'N/A')}",
                        f"Bathrooms: {row.get('bathrooms', 'N/A')}",
                        f"Area sqm: {row.get('area_sqm', 'N/A')}",
                        f"Price: {row.get('price', 'N/A')}",
                    ]
                )
            )
        return "\n---\n".join(blocks)

    # ---------------------------
    # LLM replies
    # ---------------------------
    def _generate_general_reply(self, user_input: str, session_id: str) -> str:
        if self.use_llm:
            prompt = f"""
{self.system_prompt}

Conversation history:
{self._history_as_text(session_id)}

User message:
{user_input}

Reply naturally. Use plain text only.
""".strip()
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                )
                text = getattr(response, "text", None)
                if text:
                    return self._text_to_html(text)
            except Exception as exc:
                print(f"Gemma/Gemini general reply error: {exc}")

        return self._text_to_html(
            "I’m here to help with both general questions and property searches. "
            "You can ask me anything, or tell me the city, budget, and type of home you want."
        )

    def _generate_search_reply(
        self,
        user_input: str,
        full_query: str,
        results_df: pd.DataFrame,
        session_id: str,
    ) -> str:
        if self.use_llm:
            prompt = f"""
{self.system_prompt}

Conversation history:
{self._history_as_text(session_id)}

User request:
{user_input}

Expanded search request:
{full_query}

Listing context from the user's dataset:
{self._format_context(results_df)}

Instructions:
- Answer as a smart real-estate assistant.
- If there are relevant matches, summarize the strongest 2-3 options clearly and compare them.
- If the matches are imperfect, say that honestly.
- Never invent amenities, views, distances, or prices.
- End with one short useful follow-up question when appropriate.
- Use plain text only.
""".strip()
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                )
                text = getattr(response, "text", None)
                if text:
                    return self._text_to_html(text)
            except Exception as exc:
                print(f"Gemma/Gemini search reply error: {exc}")

        if results_df.empty:
            return self._text_to_html(
                "I couldn’t find a strong match in your current listings. Try changing the area, budget, or property type."
            )

        top_titles = results_df["title"].fillna("Property").astype(str).tolist()[:3]
        reply = "I found a few listings that look relevant: " + ", ".join(top_titles) + "."
        reply += " Tell me if you want cheaper options, a specific area, or more bedrooms."
        return self._text_to_html(reply)

    # ---------------------------
    # Frontend payload helpers
    # ---------------------------
    def _rows_to_cards(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        cards: List[Dict[str, Any]] = []
        if df.empty:
            return cards

        for _, row in df.iterrows():
            title = str(row.get("title", "Property"))
            city = str(row.get("city", "")) if pd.notna(row.get("city", np.nan)) else ""
            district = str(row.get("district", "")) if pd.notna(row.get("district", np.nan)) else ""
            location = ", ".join([part for part in [district, city] if part]) or "Lebanon"

            price = row.get("price", None)
            if pd.notna(price):
                try:
                    price_value = float(price)
                    price_label = f"${price_value:,.0f}"
                except Exception:
                    price_label = str(price)
            else:
                price_label = "Price on request"

            listing_type = str(row.get("listing_type", "")).strip().lower()
            if listing_type == "rent":
                price_label += "/mo"

            badges = ["Verified"]
            if row.get("similarity", 0) >= 0.55:
                badges.append("Strong Match")

            meta = []
            bedrooms = row.get("bedrooms", None)
            bathrooms = row.get("bathrooms", None)
            area_sqm = row.get("area_sqm", None)
            if pd.notna(bedrooms):
                meta.append(f"{int(float(bedrooms))} Bed")
            if pd.notna(bathrooms):
                meta.append(f"{int(float(bathrooms))} Bath")
            if pd.notna(area_sqm):
                meta.append(f"{int(float(area_sqm))} m²")

            cards.append(
                {
                    "id": int(row.get("property_id", 0)),
                    "title": title,
                    "location": location,
                    "price": price_label,
                    "subtitle": str(row.get("property_type", "Property")).title(),
                    "badges": badges,
                    "meta": meta,
                }
            )
        return cards

    def _build_market_insight(self, df: pd.DataFrame) -> Optional[Dict[str, str]]:
        if df.empty or "price" not in df.columns:
            return None

        prices = pd.to_numeric(df["price"], errors="coerce").dropna()
        if prices.empty:
            return None

        median_price = prices.median()
        min_price = prices.min()
        max_price = prices.max()

        return {
            "title": "Market insight",
            "body": (
                f"In these matched listings, prices range from ${min_price:,.0f} to ${max_price:,.0f}, "
                f"with a median of ${median_price:,.0f}."
            ),
            "cta": "Refine by budget",
        }

    def _default_suggestions(self) -> List[str]:
        return [
            "Apartment near AUB under $800",
            "2-bedroom in Achrafieh for sale",
            "Studio for rent in Hamra",
            "Explain the Lebanese rental market",
        ]

    def _suggestions_for_turn(self, intent, listings):
        if intent in {"property_search", "property_followup", "mixed"} and listings:
            return [
            "Show cheaper options",
            "Only 2 bedrooms",
            "Only rent listings",
            "Near AUB",
            ]
        return []

    # ---------------------------
    # Formatting helpers
    # ---------------------------
    def _history_as_text(self, session_id: str) -> str:
        history = self.chat_histories.get(session_id, [])[-6:]
        if not history:
            return "No previous conversation."
        lines = []
        for item in history:
            lines.append(f"User: {item['user']}")
            lines.append(f"Assistant: {item['assistant']}")
        return "\n".join(lines)

    def _text_to_html(self, text: str) -> str:
        safe = html.escape(text.strip())
        if not safe:
            return "<p>I’m here to help.</p>"

        lines = [line.strip() for line in safe.splitlines() if line.strip()]
        if not lines:
            return f"<p>{safe}</p>"

        html_parts: List[str] = []
        in_list = False
        for line in lines:
            bullet_match = re.match(r"^[-*•]\s+(.*)", line)
            numbered_match = re.match(r"^\d+\.\s+(.*)", line)

            if bullet_match or numbered_match:
                item = bullet_match.group(1) if bullet_match else numbered_match.group(1)
                if not in_list:
                    html_parts.append("<ul>")
                    in_list = True
                html_parts.append(f"<li>{item}</li>")
            else:
                if in_list:
                    html_parts.append("</ul>")
                    in_list = False
                html_parts.append(f"<p>{line}</p>")

        if in_list:
            html_parts.append("</ul>")

        return "".join(html_parts)
