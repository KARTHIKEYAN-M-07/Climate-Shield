"""Localization dictionaries for English and Tamil across Dashboard v2."""

TRANSLATIONS = {
    "en": {
        "title": "ClimateShield AI: River-Flow Forecasting & Early-Warning Prototype for Tamil Nadu",
        "subtitle": "Academic Decision-Support Prototype | Open-Meteo GloFAS Modelled River Discharge (CC BY 4.0)",
        "warning_notice": (
            "⚠️ **ACADEMIC PROTOTYPE NOTICE**: This dashboard is an educational decision-support prototype. "
            "It is **NOT** an official flood warning system. Follow official warnings issued by "
            "the Central Water Commission (CWC), IMD, and Tamil Nadu SDMA."
        ),
        "ten_sec_title": "⏱️ 10-Second Executive Summary",
        "station": "Monitoring Station",
        "horizon": "Forecast Horizon",
        "model": "ML Model",
        "risk_category": "Risk Category",
        "expected_peak": "Expected Flow / Peak",
        "interval_fan": "Prediction Interval (Fan Chart)",
        "confidence_ood": "Confidence & Data Freshness",
        "why_forecast": "Primary Hydrologic Rationale",
        "tab_overview": "Overview",
        "tab_rain_risk": "Rainfall Risk",
        "tab_district": "District Comparison",
        "tab_station_comp": "Station Comparison",
        "tab_forecast": "Forecast (Fan Chart)",
        "tab_risk": "Risk Decision",
        "tab_explain": "Explain",
        "tab_models": "ML Model Comparison",
        "tab_training_status": "Training Status",
        "tab_backtest": "Historical Backtest",
        "tab_verification": "Verification & Drift",
        "tab_case_studies": "Case Studies",
        "tab_quality": "Data Quality",
        "tab_about": "About & Limitations",
        "download_csv": "📥 Download Forecast CSV",
        "download_json": "📥 Download Model Scores JSON",
        "download_latency": "📥 Download Latency Benchmark JSON",
        "measured_latency": "Measured Latency",
        "ood_clean": "IN-DISTRIBUTION (Normal Confidence)",
        "ood_alert": "OUT-OF-DISTRIBUTION (Low Confidence)",
        "fresh": "Data Fresh (<48h)",
        "stale": "Data Stale (>48h)",
    },
    "ta": {
        "title": "ClimateShield AI: தமிழ்நாடு நதி படுகை வெள்ள முன்னறிவிப்பு அமைப்பு மற்றும் ஆரம்ப எச்சரிக்கை முன்மாதிரி (Pending Review)",
        "subtitle": "கல்விசார் முடிவு ஆதரவு முன்மாதிரி | Open-Meteo GloFAS தரவு (Pending Review)",
        "warning_notice": (
            "⚠️ **அறிவிப்பு (Pending Review)**: இந்த டாஷ்போர்டு கல்விசார் முடிவு ஆதரவுக்கான ஒரு முன்மாதிரி மட்டுமே. "
            "இது அதிகாரப்பூர்வ வெள்ள எச்சரிக்கை அமைப்பு அல்ல. CWC, IMD மற்றும் தமிழ்நாடு SDMA ஆகியவற்றின் "
            "அதிகாரப்பூர்வ எச்சரிக்கைகளைப் பின்பற்றவும்."
        ),
        "ten_sec_title": "⏱️ 10-வினாடி முக்கியச் சுருக்கம்",
        "station": "கண்காணிப்பு நிலையம்",
        "horizon": "முன்னறிவிப்பு காலம்",
        "model": "முன்கணிப்பு மாதிரி",
        "risk_category": "ஆபத்து நிலை",
        "expected_peak": "எதிர்பார்க்கப்படும் நீர் வெளியேற்றம்",
        "interval_fan": "முன்னறிவிப்பு இடைவெளி (விசிறி விளக்கப்படம்)",
        "confidence_ood": "நம்பகத்தன்மை & தரவு புதுப்பிப்பு",
        "why_forecast": "முக்கிய நீரியல் காரணம்",
        "tab_overview": "கண்ணோட்டம்",
        "tab_rain_risk": "மழைப்பொழிவு ஆபத்து",
        "tab_district": "மாவட்ட ஒப்பீடு",
        "tab_station_comp": "நிலைய ஒப்பீடு",
        "tab_forecast": "முன்னறிவிப்பு (விசிறி படம்)",
        "tab_risk": "ஆபத்து முடிவு",
        "tab_explain": "விளக்கம் (Explain)",
        "tab_models": "மாதிரிகள் ஒப்பீடு",
        "tab_training_status": "பயிற்சி நிலை (Pending Review)",
        "tab_backtest": "வரலாற்று மறுஆய்வு (Pending Review)",
        "tab_verification": "சரிபார்ப்பு மற்றும் நகர்வு",
        "tab_case_studies": "நிகழ்வு ஆய்வுகள் (Case Studies)",
        "tab_quality": "தரவு தரம்",
        "tab_about": "பற்றி & வரம்புகள்",
        "download_csv": "📥 முன்னறிவிப்பு CSV பதிவிறக்கம்",
        "download_json": "📥 மாதிரி மதிப்பெண்கள் JSON பதிவிறக்கம்",
        "download_latency": "📥 தாமத அளவீட்டு JSON பதிவிறக்கம்",
        "measured_latency": "அளவிடப்பட்ட தாமதம்",
        "ood_clean": "இயல்பான வரம்பில் (சாதாரண நம்பிக்கை)",
        "ood_alert": "வரம்புக்கு வெளியே (குறைந்த நம்பிக்கை)",
        "fresh": "புதிய தரவு (<48 மணி)",
        "stale": "பழைய தரவு (>48 மணி)",
    },
}


def get_text(lang: str, key: str) -> str:
    """Retrieve localized string with English fallback."""
    return TRANSLATIONS.get(lang, TRANSLATIONS["en"]).get(key, TRANSLATIONS["en"].get(key, key))
