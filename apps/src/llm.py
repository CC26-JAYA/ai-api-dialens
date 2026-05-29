import os
from openai import OpenAI
from dotenv import load_dotenv

from .prompt import SYSTEM_PROMPT

load_dotenv()


FEATURE_METADATA = {
    "HighBP": {
        "label": "riwayat tekanan darah tinggi",
        "value_type": "yes_no",
    },
    "GenHlth": {
        "label": "penilaian kesehatan umum",
        "value_type": "general_health",
    },
    "HighChol": {
        "label": "riwayat kolesterol tinggi",
        "value_type": "yes_no",
    },
    "Age_BMI_Risk": {
        "label": "kombinasi usia dan BMI",
        "value_type": "age_bmi",
    },
    "CholCheck": {
        "label": "pemeriksaan kolesterol rutin",
        "value_type": "yes_no",
    },
    "HvyAlcoholConsump": {
        "label": "konsumsi alkohol berat",
        "value_type": "yes_no",
    },
    "BMI": {
        "label": "indeks massa tubuh/BMI",
        "value_type": "bmi",
    },
    "PhysActivity": {
        "label": "aktivitas fisik",
        "value_type": "activity",
    },
    "Smoker": {
        "label": "kebiasaan merokok",
        "value_type": "yes_no",
    },
}


def _yes_no(value):
    if value is None:
        return "tidak tersedia"
    try:
        return "ya" if int(value) == 1 else "tidak"
    except (TypeError, ValueError):
        return "tidak tersedia"


def _format_bmi(value):
    if value is None:
        return "tidak tersedia"
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return "tidak tersedia"


def _format_feature_value(feature, patient_data):
    metadata = FEATURE_METADATA.get(feature, {})
    value_type = metadata.get("value_type")

    if value_type == "yes_no":
        return _yes_no(patient_data.get(feature))
    if value_type == "activity":
        value = patient_data.get(feature)
        if value is None:
            return "tidak tersedia"
        try:
            return "aktif" if int(value) == 1 else "tidak aktif"
        except (TypeError, ValueError):
            return "tidak tersedia"
    if value_type == "bmi":
        return _format_bmi(patient_data.get("BMI"))
    if value_type == "general_health":
        value = patient_data.get("GenHlth")
        if value is None:
            return "tidak tersedia"
        labels = {
            1: "sangat baik",
            2: "baik sekali",
            3: "baik",
            4: "cukup",
            5: "kurang baik",
        }
        try:
            numeric_value = int(value)
        except (TypeError, ValueError):
            return "tidak tersedia"
        label = labels.get(numeric_value, "tidak tersedia")
        return f"{numeric_value} dari 5 ({label})"
    if value_type == "age_bmi":
        age = patient_data.get("Age")
        age_text = f"grup usia CDC {age}" if age is not None else "grup usia CDC tidak tersedia"
        return f"{age_text}; BMI {_format_bmi(patient_data.get('BMI'))}"

    value = patient_data.get(feature)
    return str(value) if value is not None else "tidak tersedia"


def _format_factor_context(patient_data, top_factors, explanation_method=None):
    is_fallback = explanation_method == "standardized_input_fallback"
    source = (
        "Faktor dominan berdasarkan input pengguna yang sudah distandarisasi "
        "(fallback, bukan SHAP)"
        if is_fallback
        else "Faktor yang paling memengaruhi prediksi model berdasarkan analisis SHAP"
    )

    lines = []
    for factor in top_factors:
        feature = factor.get("feature")
        metadata = FEATURE_METADATA.get(feature, {})
        label = metadata.get("label", feature)
        value = _format_feature_value(feature, patient_data)
        direction = (
            "mendorong prediksi risiko lebih tinggi"
            if factor.get("direction") == "risk"
            else "mendorong prediksi risiko lebih rendah"
        )
        lines.append(f"- {label}: {value}, menjadi faktor yang {direction}.")

    if not lines:
        return f"{source}: tidak tersedia."

    return f"{source}:\n" + "\n".join(lines)


def _fallback(risk_level):
    msgs = {
        "Low": (
            "Hasil skrining AI DiaLens menunjukkan risiko rendah. Ini bukan diagnosis medis. "
            "Pertahankan kebiasaan baik dan coba minggu ini: (1) tetap aktif bergerak minimal 30 menit/hari, "
            "(2) pilih makanan tinggi serat, (3) batasi minuman manis. "
            "Konsultasikan ke dokter bila muncul keluhan seperti sering haus, sering buang air kecil, "
            "mudah lelah, atau ada riwayat keluarga diabetes. Teruskan langkah baik ini."
        ),
        "Moderate": (
            "Hasil skrining AI DiaLens menunjukkan risiko sedang. Ini bukan diagnosis medis. "
            "Coba minggu ini: (1) kurangi minuman manis dan porsi karbohidrat olahan, "
            "(2) jalan kaki 30 menit hampir setiap hari, (3) rencanakan cek gula darah di fasilitas kesehatan. "
            "Temui dokter lebih cepat bila sering haus, sering buang air kecil, penglihatan buram, "
            "atau berat badan turun tanpa sebab. Perubahan kecil tetap berarti."
        ),
        "High": (
            "Hasil skrining AI DiaLens menunjukkan risiko tinggi. Ini bukan diagnosis medis, "
            "tetapi sebaiknya ditindaklanjuti. Minggu ini: (1) buat janji ke dokter untuk evaluasi dan pemeriksaan gula darah, "
            "(2) catat pola makan serta kurangi makanan/minuman tinggi gula, "
            "(3) mulai aktivitas ringan yang aman sesuai kondisi tubuh. "
            "Cari pertolongan medis segera bila sangat lemas, sangat haus, sering buang air kecil, "
            "atau penglihatan mendadak buram. Anda sudah mengambil langkah awal yang penting."
        ),
    }
    return msgs.get(risk_level, msgs["Moderate"])


def generate_recommendation(
    patient_data,
    prob,
    risk_level,
    top_factors,
    api_key=None,
    explanation_method=None,
):
    """
    Generate rekomendasi personal lewat OpenRouter.
    Kalo ga ada API key, balik ke rule-based fallback.
    """
    api_key = api_key if api_key is not None else os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        return _fallback(risk_level)

    factor_context = _format_factor_context(
        patient_data,
        top_factors,
        explanation_method,
    )

    prompt = (
        f"Pengguna aplikasi DiaLens mendapat hasil skrining diabetes:\n"
        f"- Probabilitas risiko: {prob:.1%}\n"
        f"- Level risiko: {risk_level}\n"
        f"- Metode interpretasi: {explanation_method or 'tidak tersedia'}\n"
        f"- Konteks faktor utama:\n{factor_context}\n"
        f"- Ringkasan input pengguna: BMI {_format_bmi(patient_data.get('BMI'))}, "
        f"Tekanan darah tinggi: {_yes_no(patient_data.get('HighBP'))}, "
        f"Kolesterol: {_yes_no(patient_data.get('HighChol'))}, "
        f"Merokok: {_yes_no(patient_data.get('Smoker'))}, "
        f"Aktivitas fisik: {_format_feature_value('PhysActivity', patient_data)}, "
        f"Grup usia CDC: {patient_data.get('Age', 'tidak tersedia')}.\n\n"
        f"Buat rencana aksi personal dalam Bahasa Indonesia maksimal 200 kata. "
        f"Jangan menambah data yang tidak diberikan."
    )

    try:
        client = OpenAI(
            api_key=api_key,
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            timeout=15.0,
        )
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b:free",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"API error: {e}, pake fallback")
        return _fallback(risk_level)
