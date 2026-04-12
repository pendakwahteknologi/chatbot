"""
Agency Configuration — JKST (Jabatan Kehakiman Syariah Terengganu)
==================================================================
"""

AGENCY_ID = "jkst"
AGENCY_NAME = "Jabatan Kehakiman Syariah Terengganu"
AGENCY_NAME_EN = "Terengganu Shariah Judiciary Department"
AGENCY_ACRONYM = "JKST"
AGENCY_WEBSITE = "https://syariah.terengganu.gov.my"

CONTACT_ADDRESS = "Tingkat 5, Bangunan Mahkamah Syariah, Jalan Sultan Mohamad, 21100 Kuala Terengganu"
CONTACT_PHONE = "09-623 2323"
CONTACT_FAX = "09-624 1510"
CONTACT_EMAIL = "jkstr@esyariah.gov.my"
CONTACT_HOURS = "Ahad-Rabu: 8:00 AM - 4:00 PM, Khamis: 8:00 AM - 3:00 PM"

INTERNAL_KEYWORDS = [
    "jkst", "jkstr", "jabatan kehakiman syariah terengganu", "mahkamah syariah",
    "syariah", "syarie", "islam", "hukum syarak", "fatwa", "muamalat",
    "mahkamah", "bicara", "perbicaraan", "kes", "pendaftaran", "guaman",
    "tuntutan", "rayuan", "permohonan", "keputusan mahkamah",
    "perkahwinan", "kahwin", "nikah", "cerai", "perceraian", "talak",
    "fasakh", "khuluk", "nafkah", "hadhanah", "mut'ah", "harta sepencarian",
    "sulh", "kaunseling", "rundingan", "pendamai", "mediasi",
    "dokumen", "borang", "sijil", "surat", "prosedur", "sop", "arahan",
    "bahagian", "jabatan", "unit", "pengurusan", "pentadbiran",
    "cuti", "tuntutan", "elaun", "gaji", "kakitangan",
    "sokongan keluarga", "penyelidikan", "rekod", "teknologi maklumat",
]

EXTERNAL_KEYWORDS = [
    "cuaca", "jadual", "tarikh",
    "perbandingan", "statistik", "malaysia",
    "undang-undang sivil", "akta am", "peraturan kerajaan",
]

NEWS_KEYWORDS = [
    "aktiviti terkini", "aktiviti jkst", "aktiviti terbaru",
    "berita terkini", "berita jkst", "berita terbaru",
    "program terkini", "program jkst",
    "perkembangan terkini", "perkembangan jkst",
]

NEWS_URLS = [
    "https://syariah.terengganu.gov.my/index.php/arkib2",
    "https://syariah.terengganu.gov.my/index.php/arkib2/berita-semasa",
]

NEWS_BASE_URL = "https://syariah.terengganu.gov.my"
WEB_SEARCH_PREFIX = "mahkamah syariah terengganu JKST"

WEBSITE_LIVE_PAGES = [
    ("https://syariah.terengganu.gov.my/index.php/profil/perutusan-kps", "Pengenalan & Perutusan"),
    ("https://syariah.terengganu.gov.my/index.php/profil/dasar", "Visi, Misi & Objektif"),
    ("https://syariah.terengganu.gov.my/index.php/profil/carta-organisasi-2023", "Carta Organisasi"),
    ("https://syariah.terengganu.gov.my/index.php/profil/struktur-organisasi", "Struktur Organisasi"),
    ("https://syariah.terengganu.gov.my/index.php/bahagian-suk/bahagian-sulh", "Bahagian Sulh"),
    ("https://syariah.terengganu.gov.my/index.php/bahagian-suk/seksyen-bahagian-sokongan-keluarga-sbsk", "Sokongan Keluarga"),
    ("https://syariah.terengganu.gov.my/index.php/hubungi-kami/ap-2", "Borang Perkhidmatan"),
    ("https://syariah.terengganu.gov.my/index.php/muat-turun/alamat-mahkamah-rendah-syariah-daerah-daerah", "Alamat Mahkamah"),
    ("https://syariah.terengganu.gov.my/index.php/muat-turun/waktu-operasi-jkstr", "Waktu Operasi"),
    ("https://syariah.terengganu.gov.my/index.php/soalan-lazim/umum", "Soalan Lazim Umum"),
    ("https://syariah.terengganu.gov.my/index.php/soalan-lazim/mahkamah-tinggi-syariah", "Soalan Lazim Mahkamah Tinggi"),
    ("https://syariah.terengganu.gov.my/index.php/soalan-lazim/mahkamah-rendah-syariah", "Soalan Lazim Mahkamah Rendah"),
]

WEBSITE_KEYWORD_MAPPING = {
    ("profil", "sejarah", "pengenalan", "latar belakang", "tentang", "perutusan"): [0],
    ("visi", "misi", "objektif", "matlamat", "dasar"): [1],
    ("carta", "organisasi", "struktur", "pegawai"): [2, 3],
    ("sulh", "mediasi", "pengantaraan"): [4],
    ("sokongan keluarga", "bsk", "nafkah"): [5],
    ("perkhidmatan", "khidmat", "borang"): [6],
    ("hubungi", "alamat", "telefon", "lokasi", "mahkamah"): [7, 8],
    ("waktu", "operasi", "jam", "buka"): [8],
    ("soalan", "lazim", "faq"): [9, 10, 11],
}

CHROMA_COLLECTION_NAME = f"{AGENCY_ID}_knowledge"

INSTALL_DIR = f"/opt/{AGENCY_ID}-ai"
FRONTEND_DIR = f"/var/www/{AGENCY_ID}-ai/public"
SERVICE_NAME = f"{AGENCY_ID}-ai"
PORT = 8001
CHROMA_DB_DIR = f"{INSTALL_DIR}/chroma_db"
KNOWLEDGE_DIR = f"{INSTALL_DIR}/knowledge"
DOCUMENTS_DIR = f"{INSTALL_DIR}/documents"
LOG_DIR = f"{INSTALL_DIR}/logs"
HF_CACHE_DIR = f"{INSTALL_DIR}/.hf_cache"

SYSTEM_PROMPT = f"""Kamu adalah pegawai khidmat pelanggan {AGENCY_ACRONYM} ({AGENCY_NAME}) yang mesra dan berpengetahuan. Jawab seperti manusia biasa yang sedang berbual — bukan robot.

PERATURAN PENTING:
1. JANGAN SEKALI-KALI guna emoji, emotikon, atau simbol hiasan.
2. Tulis secara semula jadi seperti manusia berbual — ringkas, jelas, mesra.
3. Jangan guna ayat pembuka klise. Jangan ulang soalan pengguna. Terus jawab.
4. Guna "kamu/anda" bukan "tuan/puan" kecuali konteks rasmi.

RUJUKAN DAN SUMBER:
- Jika jawapan berdasarkan dokumen tertentu, WAJIB sertakan rujukan yang boleh diklik.
- Format: [Nama Dokumen](URL_PENUH) — guna URL tepat dari konteks.
- Kalau ada fail boleh muat turun, bagi terus pautan.
- Jangan reka URL. Guna HANYA URL yang ada dalam konteks.

BATASAN:
- Jangan beri nasihat undang-undang, fatwa, atau tafsiran hukum syarak.
- Kalau tak pasti, cakap terus terang.

HUBUNGAN {AGENCY_ACRONYM}:
{CONTACT_ADDRESS}
{CONTACT_HOURS} | Tel: {CONTACT_PHONE} | {CONTACT_EMAIL}
Web: {AGENCY_WEBSITE}"""

ULTRA_SYSTEM_PROMPT = f"""Kamu pegawai khidmat pelanggan {AGENCY_ACRONYM} yang mesra. Jawab macam manusia biasa berbual — bukan robot.

CARA JAWAB:
- JANGAN guna emoji atau emotikon langsung.
- Tulis semula jadi, macam kawan yang berpengetahuan tolong jelaskan.
- Jangan ulang soalan. Jangan guna ayat pembuka klise. Terus jawab.
- Tulis jawapan akhir sahaja — jangan tunjuk proses berfikir.

RUJUKAN:
- WAJIB sertakan rujukan yang boleh diklik jika jawapan dari dokumen tertentu.
- Format: [Nama Dokumen](URL) — guna URL tepat dari konteks.
- Jangan reka URL. Guna hanya URL dari konteks yang diberi.

BATASAN:
- Jangan beri nasihat undang-undang atau fatwa. Suruh jumpa pegawai mahkamah.
- Kalau tak pasti, cakap terus terang.

HUBUNGAN {AGENCY_ACRONYM}:
{CONTACT_ADDRESS}
{CONTACT_HOURS} | Tel: {CONTACT_PHONE} | {CONTACT_EMAIL}"""
