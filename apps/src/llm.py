import os
from openai import OpenAI
from src.prompt import SYSTEM_PROMPT

API_KEY = os.getenv("ANTHROPIC_API_KEY", "")


def _fallback(risk_level):
    msgs = {
        "Low": (
            "Risiko Anda saat ini rendah — teruskan kebiasaan baik ini! "
            "Minggu ini: (1) jaga rutinitas olahraga minimal 30 menit/hari, "
            "(2) perbanyak sayur dan buah rendah gula, (3) kurangi minuman manis dan makanan olahan. "
            "Skrining ulang tiap 1 tahun meski hasilnya bagus. "
            "Temui dokter jika muncul gejala baru seperti sering haus atau mudah lelah. Semangat!"
        ),
        "Moderate": (
            "Risiko sedang — tapi masih sangat bisa dicegah dengan perubahan gaya hidup. "
            "Coba lakukan ini minggu ini: (1) ganti nasi putih dengan nasi merah atau ubi, "
            "(2) mulai jalan kaki 30 menit setiap hari, (3) cek gula darah puasa ke puskesmas terdekat. "
            "Temui dokter kalau ada gejala seperti sering haus, buang air kecil berlebihan, atau penglihatan buram. "
            "Perubahan kecil sekarang berdampak besar ke depannya!"
        ),
        "High": (
            "Risiko tinggi — ini sinyal penting untuk segera bertindak, jangan ditunda. "
            "Yang perlu dilakukan secepatnya: (1) buat janji ke dokter untuk cek HbA1c dan gula darah lengkap, "
            "(2) mulai catat pola makan harian dan hindari makanan manis dan bertepung, "
            "(3) pantau tekanan darah dan kolesterol jika ada riwayat keduanya. "
            "Langsung ke IGD jika merasakan haus ekstrem, lemas parah, atau pandangan tiba-tiba kabur. "
            "Diabetes yang ketahuan lebih awal jauh lebih mudah dikelola — Anda sudah mengambil langkah yang tepat!"
        ),
    }
    return msgs.get(risk_level, msgs["Moderate"])


def generate_recommendation(
    patient_data, prob, risk_level, top_factors, api_key=API_KEY
):
    """
    Generate rekomendasi personal lewat Claude API.
    Kalo ga ada API key, balik ke rule-based fallback.
    """
    if not api_key:
        return _fallback(risk_level)

    shap_context = "; ".join(
        [
            f"{f['feature']} ({'naikin' if f['direction'] == 'risk' else 'nurunin'} risiko)"
            for f in top_factors
        ]
    )

    prompt = (
        f"Pengguna aplikasi DiaLens mendapat hasil skrining diabetes:\n"
        f"- Probabilitas risiko: {prob:.1%}\n"
        f"- Level risiko: {risk_level}\n"
        f"- Faktor terpenting dari analisis AI: {shap_context}\n"
        f"- BMI: {patient_data.get('BMI', '?'):.1f}, "
        f"Tekanan darah tinggi: {'Ya' if patient_data.get('HighBP') else 'Tidak'}, "
        f"Kolesterol: {'Ya' if patient_data.get('HighChol') else 'Tidak'}, "
        f"Merokok: {'Ya' if patient_data.get('Smoker') else 'Tidak'}, "
        f"Aktif fisik: {'Ya' if patient_data.get('PhysActivity') else 'Tidak'}\n\n"
        f"Buat rencana aksi personal dalam Bahasa Indonesia, max 200 kata:\n"
        f"1. Interpretasi hasil (1-2 kalimat, tidak menakut-nakuti)\n"
        f"2. 3 langkah konkret yang bisa dilakukan minggu ini\n"
        f"3. Kapan harus ke dokter\n"
        f"4. Kalimat penutup yang menyemangati\n"
        f"Ingatkan ini bukan diagnosis medis."
    )

    try:
        client = OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
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
